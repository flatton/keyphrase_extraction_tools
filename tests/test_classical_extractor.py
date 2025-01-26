import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from keyphrase_extractors import ClassicalExtractor
from pke.unsupervised import (
    YAKE,
    KPMiner,
    MultipartiteRank,
    SingleRank,
    TextRank,
    TfIdf,
    TopicRank,
)


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


extractor = ClassicalExtractor(
    extractor=TextRank(),
    stop_words=None,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={
        "window": 2,
        "pos": {"NOUN", "PROPN", "ADJ", "NUM"},
        "top_percent": None,
        "normalized": False,
    },
    flat_output=True,
    use_order=False,
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("TextRank")
print(keyphrases.keyphrases)
print("-" * 80)


extractor = ClassicalExtractor(
    extractor=SingleRank(),
    stop_words=None,
    max_characters=None,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={
        "window": 10,
        "pos": {"NOUN", "PROPN", "ADJ", "NUM"},
        "normalized": False,
    },
    flat_output=True,
    use_order=False,
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("SingleRank")
print(keyphrases.keyphrases)
print("-" * 80)


extractor = ClassicalExtractor(
    extractor=TopicRank(),
    stop_words=None,
    max_characters=None,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={
        "threshold": 0.74,
        "method": "average",
        "heuristic": None,
    },
    flat_output=True,
    use_order=False,
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("TopicRank")
print(keyphrases.keyphrases)
print("-" * 80)


extractor = ClassicalExtractor(
    extractor=MultipartiteRank(),
    stop_words=None,
    max_characters=None,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={"threshold": 0.7, "method": "average", "alpha": 1.1},
    flat_output=True,
    use_order=False,
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("MultipartiteRank")
print(keyphrases.keyphrases)
print("-" * 80)

extractor = ClassicalExtractor(
    extractor=TfIdf(),
    stop_words=None,
    max_characters=None,
    args_candidate_selection={"n": 3},
    args_candidate_weighting={"df": None},
    flat_output=True,
    use_order=False,
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("TfIdf")
print(keyphrases.keyphrases)
print("-" * 80)

extractor = ClassicalExtractor(
    extractor=KPMiner(),
    stop_words=None,
    max_characters=None,
    args_candidate_selection={"lasf": 3, "cutoff": 400},
    args_candidate_weighting={"df": None, "sigma": 3.0, "alpha": 2.3},
    flat_output=True,
    use_order=False,
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("KPMiner")
print(keyphrases.keyphrases)
print("-" * 80)

extractor = ClassicalExtractor(
    extractor=YAKE(),
    stop_words=None,
    max_characters=None,
    args_candidate_selection={"n": 3},
    args_candidate_weighting={"window": 2, "use_stems": False},
    flat_output=True,
    use_order=False,
    logger=logger,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("YAKE")
print(keyphrases.keyphrases)
print("-" * 80)
