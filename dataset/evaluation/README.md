# データセット
- テキストの収集源：https://registry.opendata.aws/abeja-cc-ja/
- 200文字、2,000文字、20,000文字程度の日本語の文書を各33本ずつ収集
- 各文書について以下の方法でキーフレーズのアノテーションを実施

## アノテーション方法
- 各文書に対して、以下のラベルを著者一人で付与
- ラベルの種類
  - `main_topic`：その文書の主な話題となっている単語・フレーズ（データ型 - `list[str]`）
  - `angle`：その文書がどのような視点・切り口で `main_topic` について論じているかを表現している単語・フレーズ（データ型 - `list[str]`）
  - `essential_terms`：上記以外の文書の概要や重要箇所に関連性の高い単語・フレーズ、または、固有名詞、専門用語など（データ型 - `list[list[str]]`）
- アノテーション方式
  - `main_topic` および `angle` は一つの文書について一種類の単語・フレーズをアノテーション
    - 同じ意味の語句、言い換え表現、別名、省略表記なども正解とし、同じ種類の単語・フレーズを一つのリストとして保持
  - `essential_terms` は各文書について10種類前後の単語・フレーズのリストをアノテーション
    - `main_topic` および `angle` と同様、同じ種類の単語・フレーズは一つのリストにまとめており、単語・フレーズのリストのリストとして保持

# ライセンス
各種データはそれぞれ以下のライセンスの下で利用可能です。

## `dataset.json`
[Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use)

## `label.json`
Apache License Version 2.0
詳細は `/dataset/evaluation/lisence_of_label_data.txt` を参照ください。
