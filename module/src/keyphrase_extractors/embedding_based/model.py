import itertools
import re
from logging import Logger

import numpy as np
from keybert._maxsum import max_sum_distance
from keybert._mmr import mmr
from nltk import RegexpParser
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from spacy.language import Language
from spacy.tokens.doc import Doc

from ..utils import to_original_expression
from .data import SentenceEmbeddingBasedExtractionConfig


EmbeddingArray = NDArray[np.float32 | np.int8 | np.uint8]


class JapanesePhraseRankingModel:
    def __init__(
        self,
        model: SentenceTransformer,
        text_processor: Language,
        batchsize: int,
        use_prompt: bool,
        stop_words: set[str],
        show_progress_bar: bool,
        config: SentenceEmbeddingBasedExtractionConfig,
        count_vectorizer: CountVectorizer | None,
        logger: Logger | None = None,
    ):
        self.logger = logger

        # Embedding model
        self.model = model
        self.batchsize = batchsize
        self.use_prompt = use_prompt
        self.show_progress_bar = show_progress_bar

        # Initialize a tokenizer
        self.text_processor = text_processor
        self.stop_words = stop_words

        # Others
        self.config = config
        if (
            (not self.config.grammar_phrasing)
            and (count_vectorizer is None)
            and (self.config.ngram_range is not None)
        ):
            self.count_vectorizer = CountVectorizer(
                tokenizer=str.split,
                token_pattern=None,
                lowercase=False,
                stop_words=list(self.stop_words),
                ngram_range=self.config.ngram_range,
                max_df=1.0,
                min_df=1,
            )
        else:
            self.count_vectorizer = count_vectorizer

    def _words_to_phrases(self, words: list[str], words_pos: list[str]) -> set[str]:
        if self.logger:
            self.logger.debug("Phasing based on grammer")
            self.logger.debug(f"Grammer: {self.config.grammar}")
        grammar_parser = RegexpParser(self.config.grammar)
        tuples = [(str(i), words_pos[i]) for i in range(len(words))]
        tree = grammar_parser.parse(tuples)

        candidates: set[str] = set()
        np_indices: set[int] = set()
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                leaves = subtree.leaves()

                first = int(leaves[0][0])
                last = int(leaves[-1][0])
                phrase = " ".join(words[first : last + 1]).strip()
                candidates.add(phrase)

                for leaf in leaves:
                    idx_str, _ = leaf
                    idx = int(idx_str)
                    np_indices.add(idx)

        for i, (word, pos) in enumerate(zip(words, words_pos, strict=True)):
            word = word.strip()
            if (
                (i not in np_indices)
                and (pos in self.config.pos_filter)
                and (word not in self.stop_words)
                and word
            ):
                candidates.add(word)

        return candidates

    def _words_to_ngrams(self, words: list[str]) -> set[str]:
        if self.logger:
            self.logger.debug("Phasing based on N-gram")
            self.logger.debug(f"N-gram range: {self.config.ngram_range}")

        if self.count_vectorizer:
            vector = self.count_vectorizer.transform([" ".join(words)])
        else:
            raise ValueError("CountVectorizer is not initialized.")
        vector_array = vector.toarray()[0]
        non_zero_indices = vector_array.nonzero()[0]
        candidates: set[str] = set(self.ngram_vocab[non_zero_indices])
        return candidates

    def _tokenize_text(self, text: str, grammar_phrasing: bool = True) -> list[str]:
        if self.logger:
            self.logger.debug(f"Tokenize: {text}")
        doc: Doc = self.text_processor(text)

        words = [token.text for token in doc]
        words_pos = [token.pos_ for token in doc]

        if grammar_phrasing:
            tokens = self._words_to_phrases(words=words, words_pos=words_pos)
        else:
            tokens = self._words_to_ngrams(words=words)
        tokens = [
            to_original_expression(original_text=text, phrase=_token)
            for _token in tokens
        ]
        return tokens

    def _extract_key_contents(
        self,
        anchor_embed: EmbeddingArray,
        candidate_embeds: EmbeddingArray,
        candidate_strs: list[str],
        top_n: int,
        nr_candidates: int,
        threshold: float | None,
        use_mmr: bool,
        use_maxsum: bool,
        diversity: float,
        use_masked_distance: bool,
    ) -> list[tuple[str, float]]:
        if self.logger:
            self.logger.debug("Extract key contents")

        selected: list[tuple[str, float]]
        if use_masked_distance:
            if self.logger:
                self.logger.debug("Mode: using masked distance")
            distances = cosine_distances(anchor_embed, candidate_embeds)
            top_n = min(top_n, candidate_embeds.shape[0])
            selected_idx = np.argpartition(distances[0], -top_n)[-top_n:]
            selected_idx = selected_idx[np.argsort(distances[0][selected_idx])]
            selected = [
                (candidate_strs[i], float(distances[0][i]))
                for i in reversed(selected_idx)
            ]
        elif use_mmr:
            if self.logger:
                self.logger.debug("Mode: MMR")
            selected = mmr(
                anchor_embed, candidate_embeds, candidate_strs, top_n, diversity
            )
        elif use_maxsum:
            if self.logger:
                self.logger.debug("Mode: Max-Sum")
            selected = max_sum_distance(
                anchor_embed,
                candidate_embeds,
                candidate_strs,
                top_n,
                nr_candidates,
            )
        else:
            if self.logger:
                self.logger.debug("Mode: normal")
            similarities = cosine_similarity(anchor_embed, candidate_embeds)
            top_n = min(top_n, candidate_embeds.shape[0])
            selected_idx = np.argpartition(similarities[0], -top_n)[-top_n:]
            selected_idx = selected_idx[np.argsort(similarities[0][selected_idx])]
            selected = [
                (candidate_strs[i], float(similarities[0][i]))
                for i in reversed(selected_idx)
            ]

        if threshold is not None:
            if self.logger:
                self.logger.debug(f"Threshold: {threshold}")
            selected = [item for item in selected if item[1] >= threshold]

        return selected

    def _mask_text(self, source_text: str, target: str) -> str:
        mask = " ".join(["[MASK]"] * (len(target) // 2 + 1))
        pattern = re.compile(
            r"\s*" + re.escape(source_text) + r"\s*",
            re.IGNORECASE,
        )
        masked_text = pattern.sub(mask, source_text)
        return masked_text

    def _add_source_text(self, source_text: str, target: str) -> str:
        _target = re.sub(r"\s+", " ", target).strip()
        return f"次の本文における「{_target}」の意味\n本文：\n{source_text.strip()}"

    def _extract_sentences(
        self, docs: list[str], sentences: list[list[str]]
    ) -> list[list[tuple[str, float]]]:
        if self.logger:
            self.logger.info("Extract the key sentences")

        # ドキュメントのベクトル化
        doc_embeddings: EmbeddingArray
        if self.use_prompt:
            doc_embeddings = self.model.encode(  # type: ignore
                sentences=docs,
                prompt_name="passage",
                batch_size=self.batchsize,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
        else:
            doc_embeddings = self.model.encode(  # type: ignore
                sentences=docs,
                batch_size=self.batchsize,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )

        # 各文のベクトル化
        if self.config.use_masked_distance:
            embedding_target_sentences = [
                [
                    self._mask_text(source_text=_doc, target=_sent)
                    for _sent in _sentences
                ]
                for _doc, _sentences in zip(docs, sentences, strict=True)
            ]
        elif self.config.add_source_text:
            embedding_target_sentences = [
                [
                    self._add_source_text(source_text=_doc, target=_sent)
                    for _sent in _sentences
                ]
                for _doc, _sentences in zip(docs, sentences, strict=True)
            ]
        else:
            embedding_target_sentences = sentences

        sentence_embeddings: list[EmbeddingArray]
        if self.use_prompt:
            sentence_embeddings = [
                self.model.encode(  # type: ignore
                    sentences=_sentences,
                    prompt_name="query",
                    batch_size=self.batchsize,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                for _sentences in embedding_target_sentences
            ]
        else:
            sentence_embeddings = [
                self.model.encode(  # type: ignore
                    sentences=_sentences,
                    batch_size=self.batchsize,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                for _sentences in embedding_target_sentences
            ]

        key_sentences: list[list[tuple[str, float]]] = []

        for chunk_idx, (_sent_embeds, _sentences) in enumerate(
            zip(sentence_embeddings, sentences, strict=True)
        ):
            _doc_embed: EmbeddingArray = doc_embeddings[chunk_idx].reshape(1, -1)
            try:
                _key_sentences = self._extract_key_contents(
                    anchor_embed=_doc_embed,
                    candidate_embeds=_sent_embeds,
                    candidate_strs=_sentences,
                    top_n=self.config.max_filtered_sentences,
                    nr_candidates=self.config.nr_candidates,
                    threshold=self.config.threshold,
                    use_mmr=self.config.use_mmr,
                    use_maxsum=self.config.use_maxsum,
                    diversity=self.config.diversity,
                    use_masked_distance=self.config.use_masked_distance,
                )
            except ValueError:
                _key_sentences = []

            key_sentences.append(_key_sentences)

        return key_sentences

    def _extract_phrases(
        self, sentences: list[list[str]], phrases: list[list[list[str]]]
    ) -> list[list[list[tuple[str, float]]]]:
        if self.logger:
            self.logger.info("Extract the keyphrases")

        # 文のベクトル化
        sentence_embeddings: list[EmbeddingArray]
        if self.use_prompt:
            sentence_embeddings = [
                self.model.encode(  # type: ignore
                    sentences=_sentences,
                    prompt_name="passage",
                    batch_size=self.batchsize,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                for _sentences in sentences
            ]
        else:
            sentence_embeddings = [
                self.model.encode(  # type: ignore
                    sentences=_sentences,
                    batch_size=self.batchsize,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                for _sentences in sentences
            ]

        # フレーズのベクトル化
        if self.config.use_masked_distance:
            embedding_target_phrases = [
                [
                    [
                        self._mask_text(source_text=_sent, target=_phrase)
                        for _phrase in _phrase_set
                    ]
                    for _sent, _phrase_set in zip(_sentences, _phrases, strict=True)
                ]
                for _sentences, _phrases in zip(sentences, phrases, strict=True)
            ]
        elif self.config.add_source_text:
            embedding_target_phrases = [
                [
                    [
                        self._add_source_text(source_text=_sent, target=_phrase)
                        for _phrase in _phrase_set
                    ]
                    for _sent, _phrase_set in zip(_sentences, _phrases, strict=True)
                ]
                for _sentences, _phrases in zip(sentences, phrases, strict=True)
            ]
        else:
            embedding_target_phrases = phrases

        phrase_embeddings: list[list[EmbeddingArray]]
        if self.use_prompt:
            phrase_embeddings = [
                [
                    self.model.encode(  # type: ignore
                        sentences=_phrases_one_sentence,
                        prompt_name="query",
                        batch_size=self.batchsize,
                        show_progress_bar=self.show_progress_bar,
                        convert_to_numpy=True,
                    )
                    for _phrases_one_sentence in _phrases
                ]
                for _phrases in embedding_target_phrases
            ]
        else:
            phrase_embeddings = [
                [
                    self.model.encode(  # type: ignore
                        sentences=_phrases_one_sentence,
                        batch_size=self.batchsize,
                        show_progress_bar=self.show_progress_bar,
                    )
                    for _phrases_one_sentence in _phrases
                ]
                for _phrases in embedding_target_phrases
            ]

        key_phrase: list[list[list[tuple[str, float]]]] = []

        for _sentence_embeds, _phrase_embeds_list, _phrases_list in zip(
            sentence_embeddings, phrase_embeddings, phrases, strict=True
        ):
            _key_phrase_chunk: list[list[tuple[str, float]]] = []

            for sent_idx, (_phrase_embeds, _phrases) in enumerate(
                zip(_phrase_embeds_list, _phrases_list, strict=True)
            ):
                _sentence_embed: EmbeddingArray = _sentence_embeds[sent_idx].reshape(
                    1, -1
                )
                try:
                    _key_phrases = self._extract_key_contents(
                        anchor_embed=_sentence_embed,
                        candidate_embeds=_phrase_embeds,
                        candidate_strs=_phrases,
                        top_n=self.config.max_filtered_phrases,
                        nr_candidates=self.config.nr_candidates,
                        threshold=self.config.threshold,
                        use_mmr=self.config.use_mmr,
                        use_maxsum=self.config.use_maxsum,
                        diversity=self.config.diversity,
                        use_masked_distance=self.config.use_masked_distance,
                    )
                except ValueError:
                    _key_phrases = []

                _key_phrase_chunk.append(_key_phrases)

            key_phrase.append(_key_phrase_chunk)

        return key_phrase

    def _reciprocal_rank_fusion(
        self,
        sentence_similarities: list[float],
        key_phrases: list[list[tuple[str, float]]],
    ) -> list[tuple[str, float]]:
        if self.logger:
            self.logger.info("Merge score using reciprocal rank fusion")
        # Calcurate the ranks of sentences
        sentence_ranks: EmbeddingArray = (
            np.array(sentence_similarities).argsort()[::-1] + 1
        )

        # Calcurate the ranks of phrases
        phrases: list[str] = []
        rrf_scores: list[float] = []
        for _sent_rank, _phrases_and_scores in zip(
            sentence_ranks, key_phrases, strict=True
        ):
            phrases += [
                _phrase_and_score[0] for _phrase_and_score in _phrases_and_scores
            ]
            _phrase_ranks: EmbeddingArray = (
                np.array(
                    [_phrase_and_score[1] for _phrase_and_score in _phrases_and_scores]
                ).argsort()[::-1]
                + 1
            )
            rrf_scores += [
                (1 / (_sent_rank + self.config.rrf_k))
                + (1 / (_phrase_rank + self.config.rrf_k))
                for _phrase_rank in _phrase_ranks
            ]

        return sorted(
            zip(phrases, rrf_scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

    def _hybrid_similarity_sort(
        self,
        sentence_similarities: list[float],
        key_phrases: list[list[tuple[str, float]]],
        alpha: float = 0.5,
    ) -> list[tuple[str, float]]:
        if self.logger:
            self.logger.info("Merge score using weighted averaging")
        hybrid_scored_phrases = (
            (phrase, alpha * sentence_sim + (1 - alpha) * phrase_score)
            for sentence_sim, phrases in zip(
                sentence_similarities, key_phrases, strict=True
            )
            for phrase, phrase_score in phrases
        )
        return sorted(hybrid_scored_phrases, key=lambda x: x[1], reverse=True)

    def _concat_sentences(self, sentences: list[str]) -> list[str]:
        concated_sentences: list[str] = []
        buffer = ""
        for text in sentences:
            buffer += "\n" + text.strip()
            if len(buffer) >= self.config.minimum_characters:
                concated_sentences.append(buffer)
                buffer = ""
        if buffer:
            concated_sentences.append(buffer)
        return concated_sentences

    def _split_text_into_sentences(self, text: str) -> list[str]:
        sentences = [
            _sentence.strip()
            for _sentence in re.split(r"(?<=\n)|(?<=[。！？．])|(?<=[\.\!\?]\s)", text)
        ]
        sentences = [_sentence for _sentence in sentences if _sentence]
        return self._concat_sentences(sentences=sentences)

    def _fit_count_vectorizer(self, sentences: list[list[str]]) -> None:
        if self.logger:
            self.logger.debug("Fit CountVectorizer.")
        sentences_add_space: list[str] = [
            " ".join([token.text for token in self.text_processor(_sent)])
            for _sent in itertools.chain.from_iterable(sentences)
        ]
        if self.count_vectorizer:
            self.count_vectorizer.fit(sentences_add_space)
            self.ngram_vocab: NDArray[np.str_] = (
                self.count_vectorizer.get_feature_names_out()
            )
        else:
            raise ValueError("CountVectorizer is not initialized.")

    def extract_keyphrases(self, docs: list[str]) -> list[list[tuple[str, float]]]:
        sentences: list[list[str]] = []
        if self.logger:
            self.logger.debug("Split documents into sentences")
        for _doc in docs:
            sentences.append(self._split_text_into_sentences(text=_doc))

        if not self.config.grammar_phrasing:
            self._fit_count_vectorizer(sentences=sentences)

        if self.config.filter_sentences:
            # Extract the key sentences
            key_sentences: list[list[tuple[str, float]]] = self._extract_sentences(
                docs=docs, sentences=sentences
            )

            # Identify candidate key phrases
            if self.logger:
                self.logger.info("Identify candidate key phrases")
            sentences: list[list[str]] = []
            phrases: list[list[list[str]]] = []
            for _sentences in key_sentences:
                phrases.append(
                    [
                        list(
                            self._tokenize_text(
                                text=_sent[0],
                                grammar_phrasing=self.config.grammar_phrasing,
                            )
                        )
                        for _sent in _sentences
                    ]
                )
                sentences.append([_sent[0] for _sent in _sentences])

            if self.logger:
                self.logger.info("Extract the keyphrases")
            key_phrases: list[list[list[tuple[str, float]]]] = self._extract_phrases(
                sentences=sentences, phrases=phrases
            )

            # Merge sentence importance and phrase importance
            if self.logger:
                self.logger.info("Merge sentence importance and phrase importance")
            sentence_similarities: list[list[float]] = [
                [_sent[1] for _sent in _sentences] for _sentences in key_sentences
            ]
            if self.config.use_rrf_sorting:
                sorting_function = self._reciprocal_rank_fusion
            else:
                sorting_function = self._hybrid_similarity_sort
            sorted_keyphrases: list[list[tuple[str, float]]] = [
                sorting_function(
                    sentence_similarities=_sentence_similarities,
                    key_phrases=_key_phrases,
                )
                for _sentence_similarities, _key_phrases in zip(
                    sentence_similarities, key_phrases, strict=True
                )
            ]
        else:
            docs_nested: list[list[str]] = [[_doc] for _doc in docs]

            if self.logger:
                self.logger.info("Identify candidate key phrases")
            phrases: list[list[list[str]]] = []
            for _sentences in sentences:
                _phrases: set[str] = set()
                for _sent in _sentences:
                    _phrases.update(
                        self._tokenize_text(
                            text=_sent,
                            grammar_phrasing=self.config.grammar_phrasing,
                        )
                    )
                phrases.append([list(_phrases)])

            # Extract the key phrases
            key_phrases: list[list[list[tuple[str, float]]]] = self._extract_phrases(
                sentences=docs_nested, phrases=phrases
            )

            sorted_keyphrases: list[list[tuple[str, float]]] = [
                _phrases[0] for _phrases in key_phrases
            ]

        # Remove duplicates and sort
        if self.logger:
            self.logger.info("Remove duplicates and sort")
        result_keyphrases: list[list[tuple[str, float]]] = []
        for _keyphrases in sorted_keyphrases:
            _unique_keyphrases: dict[str, float] = {}
            for _keyphrase in _keyphrases:
                if _keyphrase[0] not in _unique_keyphrases:
                    _unique_keyphrases[_keyphrase[0]] = _keyphrase[1]
            _unique_sorted_keyphrases = sorted(
                _unique_keyphrases.items(), key=lambda x: x[1], reverse=True
            )
            result_keyphrases.append(_unique_sorted_keyphrases)

        return result_keyphrases
