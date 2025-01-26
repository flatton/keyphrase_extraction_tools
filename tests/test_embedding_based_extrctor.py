import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from keyphrase_extractors import (
    EmbeddingModel,
    EmbeddingPrompts,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import SentenceEmbeddingBasedExtractionConfig


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
    logger=logger,
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
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("cl-nagoya/ruri-base")
print(keyphrases.keyphrases)
print("-" * 80)
