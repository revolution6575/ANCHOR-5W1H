# ANCHOR-5W1H

ANCHOR is a cluster-grounded framework for extracting 5W1H records from Chinese, English, and Spanish documents. It pairs a prompted large language model (LLM) with a small language model (SLM) built on mmBERT. Given a document, ANCHOR identifies a cluster of related events, marks the main event, and links each role to a span in the source text. An event in the cluster can supply the reason for another event or describe how it was carried out. Character offsets use an exclusive end index, so `text[start:end]` recovers the source span.

## Method

1. The LLM proposes up to five candidate events with their triggers and relations. For each trigger it can locate in the text, it drafts who, when, and where, then uses those answers and the source text to draft what in the same request. ANCHOR ranks the candidates by trigger position, predicate status, 4W coverage, and incoming event relations. It keeps up to three events and marks the highest-ranked one as the main event.
2. The SLM encodes the document once with mmBERT, using 512-token windows with a 128-token overlap. For each selected event and each of the four roles, start and end boundary heads propose evidence spans. ANCHOR merges these with spans from the syntactic parse and any draft answer found in the text.
3. The LLM refines all four roles together for each event. It sees the full document, the triggers of all selected events, and the target event’s candidate spans. For each role, it can choose a candidate, quote another source span, or leave the role empty.
4. The SLM reuses the document encoding to score evidence for why and how. Each role has its own boundary and role-scoring heads conditioned on the target trigger and the refined 4W. Each role can take one span above its own threshold or remain empty. A span can fill only one of these two roles for the same event.

## Installation

Use Python 3.10 or later. Install the dependencies and download the spaCy models for the languages you plan to process.

```shell
python -m pip install -r requirements.txt
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

## Usage

Set the Chat Completions-compatible endpoint, model name, and checkpoint paths in `config/anchor.json`. If the endpoint requires an API key, set `ANCHOR_API_KEY`. Inference requires matching weights for the discriminator and the 4W boundary heads. You can train them or supply existing checkpoints. The default directories are `checkpoints/discriminator/` and `checkpoints/four_w/`, respectively. [Training scripts](training/)

```shell
python -m ceh.runner --input examples/documents.jsonl --tag example
python -m ceh.runner --corpus CEC,DEIE --limit 1 --tag sample
python -m ceh.runner --corpus all --tag all-corpora
```

For custom input, provide a JSONL file with one document per line. Each document needs `doc_id`, `text`, and `lang` (`zh`, `en`, or `es`). The `--limit` option sets the maximum number of documents read per corpus. Results are written to `outputs/raw/<tag>/`. Use `--resume` to continue a run with the same input and configuration.

## Data

The `datasets/` directory contains ten corpora with 26,110 documents and 82,286 annotated events. Use a corpus name to read its texts and annotations. [Corpus sources and splits](datasets/README.md)

```python
from bench.release import records

article = next(records("DEIE"))
print(article["Id"], article["Text"])
```
