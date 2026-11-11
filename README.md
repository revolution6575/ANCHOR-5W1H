# ANCHOR-5W1H

ANCHOR-5W1H extracts events from Chinese, English, and Spanish documents. For each event, it identifies the trigger, marks whether the event is the main event, and extracts source spans for the WHO, WHAT, WHEN, WHERE, WHY, and HOW roles. Character offsets use an exclusive end index, so `text[start:end]` returns the original span.

## Method

The language model proposes up to five candidate events. For each trigger, it drafts four roles in a single request, in the order **WHO → WHEN → WHERE → WHAT**. The first three roles help distinguish events in the same document. For WHAT, the model selects the span that expresses the central action or state, including its necessary arguments, negation, and modality. The system scores candidates using trigger salience, position in the document, role coverage, and relations between events. It retains up to three events and marks the highest-scoring one as the main event.

mmBERT encodes the document in 512-token windows with a 128-token overlap. For each retained event, the system combines spans proposed by the 4W boundary heads with spans from the syntactic parse and draft spans located in the source text. The language model then refines all four roles together in one request per event. Separate prediction heads extract WHY and HOW using the refined event representation and the same document encoding.

## Installation

Use Python 3.10 or later. Install the dependencies and download the spaCy models for the languages you plan to process.

```shell
python -m pip install -r requirements.txt
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

## Usage

Set the Chat Completions-compatible endpoint, model name, and checkpoint paths in `config/anchor.json`. If the endpoint requires an API key, set `ANCHOR_API_KEY`. Inference requires matching weights for the discriminator and the 4W boundary heads. You can train them or supply existing checkpoints. The default directories are `checkpoints/discriminator/` and `checkpoints/four_w/`, respectively. [Training instructions](training/README.md)

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
