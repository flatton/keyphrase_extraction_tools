import re
from pathlib import Path

from keyphrase_extractors import (
    EmbeddingModel,
    EmbeddingPrompts,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import SentenceEmbeddingBasedExtractionConfig


input_text_filepath = Path("../dataset/sample/ABEJA_Techblog.md")
with input_text_filepath.open("r") as f:
    input_text = "".join(f.readlines())
input_text = re.sub(r"\[(.+?)\]\(https://[^\)]+\)", r"\1", input_text)
input_text = re.sub(r"https?://[^\s]+", "", input_text)

embedding_model_config = EmbeddingModel(
    name="hotchpotch/static-embedding-japanese",
    device="cpu",
    prompts=None,
    trust_remote_code=True,
    batchsize=32,
    show_progress_bar=False,
)
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",  # "normal", "use_maxsum", "use_mmr"
    # nr_candidates=30,
    # diversity=0.7,
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=False,
    ngram_range=(1, 1),
    pos_filter=set(),
    use_masked_distance=False,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=None,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("hotchpotch/static-embedding-japanese")
print(keyphrases.keyphrases)
print("-" * 80)


embedding_model_config = EmbeddingModel(
    name="cl-nagoya/ruri-base",
    device="mps",
    prompts=EmbeddingPrompts(query="クエリ: ", passage="文章: "),
    trust_remote_code=True,
    batchsize=32,
    show_progress_bar=False,
)
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",  # "normal", "use_maxsum", "use_mmr"
    # nr_candidates=30,
    # diversity=0.7,
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=False,
    ngram_range=(1, 1),
    pos_filter=set(),
    use_masked_distance=False,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=None,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("cl-nagoya/ruri-base")
print(keyphrases.keyphrases)
print("-" * 80)
