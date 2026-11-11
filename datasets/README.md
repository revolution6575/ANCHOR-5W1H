# ANCHOR-5W1H 语料

十个语料共 26,110 篇文档、82,286 个事件。每个语料对应一份 `<语料名>.json.gz` 文件，解压后为 `ceh-dataset-v1` 格式。

| 语料 | 语言 | 文档 | 事件 | 来源及标注 |
| --- | --- | ---: | ---: | --- |
| CEC | 中文 | 332 | 756 | 上海大学中文突发事件语料库。由事件指示词及论元映射，模型补充和复核。 |
| CMNEE | 中文 | 2,997 | 6,573 | 军事新闻。事件簇和 5W1H 由模型标注并复核。[原始论文](https://arxiv.org/abs/2404.12242) |
| CNC-RECESS | 英文 | 1,807 | 2,904 | 因果新闻。以上游原因和结果跨度为参考，按全文标注并复核。[原始论文](https://aclanthology.org/2023.ijcnlp-main.6/) |
| DEIE | 中文 | 18,936 | 69,696 | 新闻。由上游人工事件与论元标注映射，模型复核、补充和修订。[原始论文](https://aclanthology.org/2024.lrec-main.410/) |
| DiscourseEE | 英文 | 312 | 598 | 医药讨论。参考上游事件标注，按帖子及评论原文复核。 |
| Flares | 西班牙文 | 1,582 | 1,582 | 新闻。上游人工 5W1H 标签转换为统一格式。[FLARES 任务](https://sites.google.com/gcloud.ua.es/flares) |
| FrameNet | 英文 | 13 | 18 | FrameNet 1.7 框架语义标注经映射和模型复核。 |
| Giveme5W1H | 英文 | 96 | 96 | 长篇新闻。将上游人工 5W1H 答案定位到原文。 |
| OEE | 中文 | 7 | 29 | OEE-CFC 财经事件标注经映射和模型复核。 |
| PHEE | 英文 | 28 | 34 | PHEE-V2 医药事件标注经映射和模型复核。 |

## 文件结构

解压后的 JSON 含有 `schema_version` 和 `records`。每条记录用 `Id` 标识，`Text` 保存原文，`Nodes` 保存各角色的证据片段。事件簇在 `Event_Clusters`，簇内每个事件的 `Event_Hyperedge.Nodes` 按角色列出节点 ID，允许一项角色引用多个片段。`Review` 记录标注来源。各语料的标注层级和文件信息见 [语料清单](manifest.json)。

```python
import gzip
import json

with gzip.open("datasets/CEC.json.gz", "rt", encoding="utf-8") as source:
    record = json.load(source)["records"][0]

node = record["Nodes"][0]
print(node["Node_Type"], node["Tag_Text"])
```

## 切分

`splits/` 中每行存放一个文档 ID。`natural_<语料>_all.txt` 保留全量顺序。有原始切分的语料另附对应 ID。例如 DEIE 的 train、dev、test 分别有 14,972、1,961、2,003 篇。CNC-RECESS 的发布切分 `test` 对应上游 dev。

`enriched_event_*` 选取至少一个事件同时具有 WHY 和 HOW 的文档，按 ID 排序后至多取 500 篇。`enriched_record_legacy_*` 在整篇文档层面检查这两个角色，`enriched_*` 是该历史口径的别名。各切分规模和角色覆盖率分别见 [切分清单](splits/manifest.json) 和 [覆盖统计](coverage.json)。

原文及上游标注的权利归各来源方。Flares 随附的许可文本见 [Apache License 2.0](licenses/Flares.txt)。
