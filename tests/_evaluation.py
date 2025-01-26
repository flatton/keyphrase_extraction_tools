from pathlib import Path

from dotenv import load_dotenv
from keyphrase_extractors import (
    ClassicalExtractor,
    EmbeddingModel,
    EmbeddingPrompts,
    GenerationBasedExtractor,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import SentenceEmbeddingBasedExtractionConfig
from keyphrase_extractors.evaluate import EvaluationPipeline
from keyphrase_extractors.generation_based import ResponseSchema
from langrila import Agent
from langrila.openai import OpenAIClient
from pke.unsupervised import (
    YAKE,
    KPMiner,
    MultipartiteRank,
    SingleRank,
    TextRank,
    TfIdf,
    TopicRank,
)


eval_data_dirpath = Path("../dataset/evaluation")
output_dirpath = Path("../output/evaluation")
evaluation = EvaluationPipeline(
    dataset_json_path=eval_data_dirpath / "dataset.json",
    label_json_path=eval_data_dirpath / "label.json",
    k_list=[5, 10, 25],
    output_dirpath=output_dirpath,
)

# Graph-based extractor
extractor = ClassicalExtractor(
    extractor=TextRank(),
    stop_words=None,
    max_characters=10000,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={
        "window": 2,
        "pos": {"NOUN", "PROPN", "ADJ", "NUM"},
        "top_percent": None,
        "normalized": False,
    },
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="TextRank")

extractor = ClassicalExtractor(
    extractor=SingleRank(),
    stop_words=None,
    max_characters=10000,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={
        "window": 10,
        "pos": {"NOUN", "PROPN", "ADJ", "NUM"},
        "normalized": False,
    },
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="SingleRank")

extractor = ClassicalExtractor(
    extractor=TopicRank(),
    stop_words=None,
    max_characters=10000,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={
        "threshold": 0.74,
        "method": "average",
        "heuristic": None,
    },
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="TopicRank")

extractor = ClassicalExtractor(
    extractor=MultipartiteRank(),
    stop_words=None,
    max_characters=10000,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={"threshold": 0.7, "method": "average", "alpha": 1.1},
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="MultipartiteRank")

# Statistical extractor
extractor = ClassicalExtractor(
    extractor=TfIdf(),
    stop_words=None,
    max_characters=10000,
    args_candidate_selection={"n": 3},
    args_candidate_weighting={"df": None},
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="TfIdf")

extractor = ClassicalExtractor(
    extractor=KPMiner(),
    stop_words=None,
    max_characters=10000,
    args_candidate_selection={"lasf": 3, "cutoff": 400},
    args_candidate_weighting={"df": None, "sigma": 3.0, "alpha": 2.3},
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="KPMiner")

extractor = ClassicalExtractor(
    extractor=YAKE(),
    stop_words=None,
    max_characters=10000,
    args_candidate_selection={"n": 3},
    args_candidate_weighting={"window": 2, "use_stems": False},
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="YAKE")

# Embedding model-based
embedding_model_config = EmbeddingModel(
    name="cl-nagoya/ruri-base",
    device="mps",
    prompts=EmbeddingPrompts(query="クエリ: ", passage="文章: "),
    trust_remote_code=True,
    batchsize=32,
    show_progress_bar=False,
)

## ngram_range=(1, 1)
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    # max_filtered_sentences=30,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=False,
    ngram_range=(1, 1),
    pos_filter=set(),
    use_masked_distance=False,
    minimum_characters=10,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="embedding_001_1gram")


## ngram_range=(1, 4)
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    # max_filtered_sentences=30,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=False,
    ngram_range=(1, 4),
    use_masked_distance=False,
    minimum_characters=10,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="embedding_002_4gram")

## ngram_range=(1, 1), threshold=0.7
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    # max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=False,
    grammar_phrasing=False,
    ngram_range=(1, 4),
    use_masked_distance=False,
    minimum_characters=10,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="embedding_003_4gram_thres")

## ngram_range=(1, 1), diversity_mode="use_mmr", diversity=0.7
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="use_mmr",
    max_filtered_phrases=30,
    # max_filtered_sentences=30,
    diversity=0.7,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=False,
    ngram_range=(1, 4),
    use_masked_distance=False,
    minimum_characters=10,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="embedding_004_4gram_MMR")

## grammar_phrasing=True
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    # max_filtered_sentences=30,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="embedding_011_grammar")

## grammar_phrasing=True, threshold=0.7
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    # max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=False,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="embedding_012_grammar_thres")

## grammar_phrasing=True, diversity_mode="use_mmr", diversity=0.7
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="use_mmr",
    max_filtered_phrases=30,
    # max_filtered_sentences=30,
    diversity=0.7,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(extractor=extractor, output_dirname="embedding_013_grammar_MMR")

## ngram_range=(1, 4), threshold=0.7, filter_sentences=True
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=True,
    grammar_phrasing=False,
    ngram_range=(1, 4),
    use_masked_distance=False,
    minimum_characters=10,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(
    extractor=extractor, output_dirname="embedding_021_4gram_thres_filter-sentence"
)

## ngram_range=(1, 4), diversity_mode="use_mmr", diversity=0.7
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="use_mmr",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    diversity=0.7,
    threshold=None,
    filter_sentences=True,
    grammar_phrasing=False,
    ngram_range=(1, 4),
    use_masked_distance=False,
    minimum_characters=10,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(
    extractor=extractor, output_dirname="embedding_022_4gram_MMR_filter-sentence"
)

## grammar_phrasing=True, threshold=0.7, filter_sentences=True
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=True,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor, output_dirname="embedding_023_grammar_thres_filter-sentence"
)

## grammar_phrasing=True, diversity_mode="use_mmr", diversity=0.7, filter_sentences=True,
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="use_mmr",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    diversity=0.7,
    threshold=None,
    filter_sentences=True,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor, output_dirname="embedding_024_grammar_MMR_filter-sentence"
)

# grammar_phrasing=True, threshold=0.7, use_masked_distance=True
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    threshold=0.7,
    filter_sentences=False,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=True,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor, output_dirname="embedding_025_grammar_thres_masked-distance"
)

# grammar_phrasing=True, diversity_mode="use_mmr", diversity=0.7, use_masked_distance=True
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="use_mmr",
    max_filtered_phrases=30,
    diversity=0.7,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=True,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor, output_dirname="embedding_026_grammar_MMR_masked-distance"
)

# grammar_phrasing=True, threshold=0.7, add_source_text=True
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    threshold=0.7,
    filter_sentences=False,
    grammar_phrasing=True,
    ngram_range=None,
    add_source_text=True,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor, output_dirname="embedding_027_grammar_thres_prompting"
)

## grammar_phrasing=True, diversity_mode="use_mmr", diversity=0.7, add_source_text=True
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="use_mmr",
    max_filtered_phrases=30,
    diversity=0.7,
    threshold=None,
    filter_sentences=False,
    grammar_phrasing=True,
    ngram_range=None,
    add_source_text=True,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor, output_dirname="embedding_028_grammar_MMR_prompting"
)

## grammar_phrasing=True, threshold=0.7, filter_sentences=True, minimum_characters=100
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=True,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
    minimum_characters=100,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)
evaluation.run(
    extractor=extractor,
    output_dirname="embedding_029_grammar_thres_filter-sentence_min100",
)

## cl-nagoya/ruri-small, grammar_phrasing=True, threshold=0.7, filter_sentences=True, minimum_characters=100
embedding_model_config = EmbeddingModel(
    name="cl-nagoya/ruri-small",
    device="mps",
    prompts=EmbeddingPrompts(query="クエリ: ", passage="文章: "),
    trust_remote_code=True,
    batchsize=32,
    show_progress_bar=False,
)

extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=True,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
    minimum_characters=100,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor,
    output_dirname="embedding_034_grammar_thres_filter-sentence_small_min100",
)

## cl-nagoya/ruri-large, grammar_phrasing=True, threshold=0.7, filter_sentences=True, minimum_characters=100
embedding_model_config = EmbeddingModel(
    name="cl-nagoya/ruri-large",
    device="mps",
    prompts=EmbeddingPrompts(query="クエリ: ", passage="文章: "),
    trust_remote_code=True,
    batchsize=32,
    show_progress_bar=False,
)

extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=True,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
    minimum_characters=100,
)

extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
)

evaluation.run(
    extractor=extractor,
    output_dirname="embedding_034_grammar_thres_filter-sentence_small_min100",
)

env_filepath = Path("../../.env_api")
env_filepath.is_file()
load_dotenv(dotenv_path=env_filepath)

azure_openai_client = OpenAIClient(
    api_key_env_name="AZURE_API_KEY",
    api_type="azure",
    azure_api_version="2024-08-01-preview",
    azure_endpoint_env_name="AZURE_ENDPOINT",
    azure_deployment_id_env_name="AZURE_DEPLOYMENT_ID",
)

agent = Agent(
    client=azure_openai_client,
    model="gpt-4o",
    temperature=0.0,
    response_format=ResponseSchema,
)

extractor = GenerationBasedExtractor(
    agent=agent, max_characters=None, flat_output=True, use_order=False
)
evaluation.run(extractor=extractor, output_dirname="GPT4o-2024-11-20")
