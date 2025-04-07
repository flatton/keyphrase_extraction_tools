## 目次
- [概要](#概要)
- [インストール](#インストール)
- [使い方](#使い方)
- [License](#License)
- [Citation](#Citation)

<a name="概要"/></a>
## 概要
日本語のテキストに対してキーフレーズ抽出を行うためのツールを提供しています。
キーフレーズ抽出のアプローチとして
- グラフベースおよび統計量ベースの手法（[PKE](https://github.com/boudinfl/pke) の wrapper）
- 埋め込みモデルベースのアプローチ（[KeyBERT](https://github.com/MaartenGr/KeyBERT) の拡張）
- 生成モデルベースのアプローチ（[langrila](https://github.com/taikinman/langrila) の拡張）
をサポートしています。

<a name="インストール"/></a>
## インストール
```
poetry install
```

<a name="使い方"/></a>
## 使い方
### グラフベースまたは統計量ベースの抽出器
[sample code](tests/test_classical_extractor.py)
```Python
from keyphrase_extractors import ClassicalExtractor
from pke.unsupervised import TextRank

input_text = "これはテスト用のテキストです。"

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
    logger=None,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("TextRank")
print(keyphrases.keyphrases)
print("-" * 80)
```

### 埋め込みモデルベースの抽出器
[sample code](tests/test_embedding_based_extrctor.py)
```Python
from keyphrase_extractors import (
    EmbeddingModel,
    EmbeddingPrompts,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import SentenceEmbeddingBasedExtractionConfig

input_text = "これはテスト用のテキストです。"

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
    logger=None,
)

keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("cl-nagoya/ruri-base")
print(keyphrases.keyphrases)
print("-" * 80)
```

### 生成モデルベースの抽出器
[sample code](tests/test_llm_extractor.py)
```Python
from dotenv import load_dotenv
from keyphrase_extractors import GenerationBasedExtractor
from keyphrase_extractors.generation_based import ResponseSchema
from langrila import Agent
from langrila.openai import OpenAIClient


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

extractor = GenerationBasedExtractor(agent=agent, logger=None)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("GPT-4o")
print(keyphrases.keyphrases)
print("-" * 80)
```

### 評価
[sample code](tests/test_evaluation.py.py)
```Python
from pathlib import Path

from keyphrase_extractors import (
    EmbeddingModel,
    EmbeddingPrompts,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import SentenceEmbeddingBasedExtractionConfig
from keyphrase_extractors.evaluate import EvaluationPipeline

# Initialize the evaluation pipeline
eval_data_dirpath = Path("../dataset/evaluation")
output_dirpath = Path("../output/evaluation")
evaluation = EvaluationPipeline(
    dataset_json_path=eval_data_dirpath / "dataset.json",
    label_json_path=eval_data_dirpath / "label.json",
    k_list=[5, 10, 25],
    output_dirpath=output_dirpath,
    logger=None,
)

# Initialize an extractor
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
    logger=None,
)

# Evaluate the model
evaluation.run(
    extractor=extractor,
    output_dirname="ruri-base",
)
```

<a name="License"/></a>
## License
Apache License Version 2.0
詳細は [LICENSE](LICENSE) を参照ください。

<a name="Citation"/></a>
## Citation
本ツールを引用する際は、以下の情報をご記載ください。

```
@inproceedings{japanese_keyphrase_extraction,
  title={埋め込みモデルベースの教師なしキーフレーズ抽出における長文に対する抽出精度の改善},
  author={藤原 知樹},
  booktitle={言語処理学会第31回年次大会},
  pages={4310-4315},
  year={2025}
}
```
