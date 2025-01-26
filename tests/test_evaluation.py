import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from keyphrase_extractors import (
    EmbeddingModel,
    EmbeddingPrompts,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import SentenceEmbeddingBasedExtractionConfig
from keyphrase_extractors.evaluate import EvaluationPipeline


def setup_logger(log_file: str = "../output/logs/test.log") -> logging.Logger:
    # ロガーを作成
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.DEBUG)  # ログレベルを設定（最も詳細なDEBUG）

    # フォーマッタの設定
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # コンソールハンドラーの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # INFO以上を出力
    console_handler.setFormatter(formatter)

    # ファイルハンドラーの設定（ローテーションファイル）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # DEBUG以上を出力
    file_handler.setFormatter(formatter)

    # ハンドラーをロガーに追加
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()

eval_data_dirpath = Path("../dataset/evaluation")
output_dirpath = Path("../output/evaluation")
evaluation = EvaluationPipeline(
    dataset_json_path=eval_data_dirpath / "dataset.json",
    label_json_path=eval_data_dirpath / "label.json",
    k_list=[5, 10, 25],
    output_dirpath=output_dirpath,
    logger=logger,
)

## hotchpotch/static-embedding-japanese, grammar_phrasing=True, threshold=0.7, filter_sentences=True, minimum_characters=100
embedding_model_config = EmbeddingModel(
    name="hotchpotch/static-embedding-japanese",
    device="cpu",
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
    logger=logger,
)

evaluation.run(
    extractor=extractor,
    output_dirname="static-embedding-japanese",
)

## cl-nagoya/ruri-small, grammar_phrasing=True, threshold=0.7, filter_sentences=True, minimum_characters=100
embedding_model_config = EmbeddingModel(
    name="cl-nagoya/ruri-base",
    device="cpu",
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
    logger=logger,
)

evaluation.run(
    extractor=extractor,
    output_dirname="static-embedding-japanese",
)

## cl-nagoya/ruri-small, grammar_phrasing=True, threshold=0.7, filter_sentences=True, minimum_characters=100
embedding_model_config = EmbeddingModel(
    name="cl-nagoya/ruri-base",
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
    logger=logger,
)

evaluation.run(
    extractor=extractor,
    output_dirname="static-embedding-japanese",
)
