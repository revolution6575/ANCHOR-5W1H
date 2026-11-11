# ANCHOR-5W1H

ANCHOR-5W1H 从中文、英文和西班牙文文档中抽取事件。每个事件包含触发词、主事件标记和 WHO、WHAT、WHEN、WHERE、WHY、HOW 六个角色的原文片段。字符偏移采用左闭右开区间，`text[start:end]` 对应原文。

## 方法

语言模型先提出最多五个候选事件。针对每个触发词，用一次请求按 **WHO → WHEN → WHERE → WHAT** 的顺序生成初稿。前三项帮助区分同一文档中的事件，WHAT 从原文截取核心动作或状态，保留必要论元、否定和情态。随后根据触发词、位置、角色覆盖和事件关系评分，最多保留三个事件，并标出主事件。

mmBERT 按 512 token 分窗编码全文，相邻窗口重叠 128 token。对每个保留事件，4W 边界头提出候选片段，与句法候选和已定位的初稿片段合并。语言模型再用一次请求联合修订四个角色。WHY 和 HOW 由独立的判别头给出，使用修订后的事件表示和同一份文档编码。

## 安装

需要 Python 3.10 或更高版本。安装依赖后，按输入语种下载对应的 spaCy 模型。

```shell
python -m pip install -r requirements.txt
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

## 运行

在 `config/anchor.json` 中设置兼容 Chat Completions 的服务地址、模型名和权重路径。服务密钥可通过环境变量 `ANCHOR_API_KEY` 传入。判别器和 4W 边界头需自行训练或提供配套权重，默认目录分别为 `checkpoints/discriminator/` 和 `checkpoints/four_w/`。[训练说明](training/README.md)

```shell
python -m ceh.runner --input examples/documents.jsonl --tag example
python -m ceh.runner --corpus CEC,DEIE --limit 1 --tag sample
python -m ceh.runner --corpus all --tag all-corpora
```

自定义输入为 JSONL，每行包含 `doc_id`、`text` 和 `lang`（`zh`、`en` 或 `es`）。`--limit` 指每个语料读取的文档数。结果保存在 `outputs/raw/<tag>/`，使用相同输入和配置时可加 `--resume` 续跑。

## 数据

`datasets/` 收录十个语料，共 26,110 篇文档、82,286 个事件。可按语料名读取原文和标注。[语料来源与切分](datasets/README.md)

```python
from bench.release import records

article = next(records("DEIE"))
print(article["Id"], article["Text"])
```
