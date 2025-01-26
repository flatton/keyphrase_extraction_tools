import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from keyphrase_extractors import GenerationBasedExtractor
from keyphrase_extractors.generation_based import ResponseSchema
from langrila import Agent
from langrila.openai import OpenAIClient


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


env_filepath = Path("../../.env_api")
load_dotenv(dotenv_path=env_filepath)

azure_openai_client = OpenAIClient(
    api_key_env_name="AZURE_API_KEY",
    api_type="azure",
    azure_api_version="2024-08-01-preview",
    azure_endpoint_env_name="AZURE_ENDPOINT",
    azure_deployment_env_name="AZURE_DEPLOYMENT",
)

agent = Agent(
    client=azure_openai_client,
    model="gpt-4o",
    temperature=0.0,
    response_format=ResponseSchema,
)

extractor = GenerationBasedExtractor(agent=agent, logger=logger)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("GPT-4o")
print(keyphrases.keyphrases)
print("-" * 80)
