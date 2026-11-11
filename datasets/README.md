# ANCHOR-5W1H corpora

The ten corpora contain 26,110 documents and 82,286 events. Each `<corpus>.json.gz` file contains a UTF-8 JSON document in the `ceh-dataset-v1` format.

| Corpus | Language | Documents | Events | Source and annotation |
| --- | --- | ---: | ---: | --- |
| CEC | Chinese | 332 | 756 | Shanghai University’s corpus of Chinese emergency reports. Event triggers and arguments were mapped to 5W1H roles, then supplemented and reviewed by a model. |
| CMNEE | Chinese | 2,997 | 6,573 | Military news. Model-based annotation and review covered both event clusters and 5W1H roles. [Paper](https://arxiv.org/abs/2404.12242) |
| CNC-RECESS | English | 1,807 | 2,904 | News annotated for causal relations. The original cause and effect spans guided annotation and model review of the full text. [Paper](https://aclanthology.org/2023.ijcnlp-main.6/) |
| DEIE | Chinese | 18,936 | 69,696 | News. Human-annotated events and arguments were mapped to 5W1H roles. A model reviewed, supplemented, and revised the annotations. [Paper](https://aclanthology.org/2024.lrec-main.410/) |
| DiscourseEE | English | 312 | 598 | Medical discussions. Model review used the original posts and comments alongside the source event annotations. |
| Flares | Spanish | 1,582 | 1,582 | News. The original human 5W1H labels were converted to the common format. [FLARES task](https://sites.google.com/gcloud.ua.es/flares) |
| FrameNet | English | 13 | 18 | FrameNet 1.7 frame-semantic annotations were mapped to 5W1H roles and reviewed by a model. |
| Giveme5W1H | English | 96 | 96 | Long-form news. The original human 5W1H answers were aligned to spans in the articles. |
| OEE | Chinese | 7 | 29 | OEE-CFC financial event annotations were mapped to 5W1H roles and reviewed by a model. |
| PHEE | English | 28 | 34 | PHEE-V2 medical event annotations were mapped to 5W1H roles and reviewed by a model. |

## Record format

Each JSON file contains `schema_version` and `records`. Within a record, `Id` identifies the document, `Text` holds the source text, and `Nodes` contains the evidence spans. Events are grouped under `Event_Clusters`. Within each event, `Event_Hyperedge.Nodes` lists node IDs by role. A role can reference more than one span. The `Review` field records annotation provenance. The [corpus manifest](manifest.json) lists annotation tiers and file details.

```python
import gzip
import json

with gzip.open("datasets/CEC.json.gz", "rt", encoding="utf-8") as source:
    record = json.load(source)["records"][0]

node = record["Nodes"][0]
print(node["Node_Type"], node["Tag_Text"])
```

## Splits

The text files in `splits/` list one document ID per line. For each corpus, `natural_<corpus>_all.txt` lists every document in its original order. Corpora with predefined partitions also have separate ID lists for those partitions. DEIE has 14,972 training documents, 1,961 development documents, and 2,003 test documents. The CNC-RECESS split named `test` corresponds to the upstream development split.

The `enriched_event_*` lists include documents with at least one event that has both WHY and HOW evidence. IDs are sorted lexicographically, and each list is capped at 500 documents. The `enriched_record_legacy_*` lists apply the criterion to the whole document, so WHY and HOW may come from different events. Files named `enriched_*` without an `event_` or `record_legacy_` qualifier are aliases for the document-level lists. [Split counts](splits/manifest.json) · [Role coverage](coverage.json)

Rights to the source texts and upstream annotations belong to their respective rights holders. Flares includes a copy of the [Apache License 2.0](licenses/Flares.txt).
