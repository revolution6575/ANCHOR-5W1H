# 模型训练

训练分两步。先训练文档编码器和 WHY/HOW 判别器，再冻结编码器，训练 4W 边界头。训练池包含 722 篇文档，语料名、文档 ID、原文哈希和三折分配保存在 `pool.json`。

## 准备训练数据

先在 `config/anchor.json` 设置语言模型服务，并安装中、英、西三种 spaCy 模型。在仓库根目录运行以下命令。准备脚本按 ID 读取文档，做句法分析并向语言模型请求 S1 初稿，结果写入 `runs/training/`。如果只需重建带标注的训练集，使用 `--data-only`。

```shell
python training/prepare.py
```

## 训练权重

训练使用支持 BF16 的 CUDA 设备和本地 mmBERT-base 模型及分词器。

判别器训练 5 个 epoch，同时使用标注事件和能与标注对齐的预测事件。

```shell
python training/train_base.py --model /path/to/mmbert-base --model-name mmbert_base --epochs 5 --output runs/training/full_base
```

从 `runs/training/full_base/completed.json` 读取 `best.checkpoint`，将该路径传给 `--base-checkpoint`。4W 边界头训练 11 个 epoch。

```shell
python training/train_heads.py --base-checkpoint /path/to/trained-discriminator --epochs 11
```

训练完成后，`runs/training/heads/full/completed.json` 中的 `best.checkpoint` 指向 4W 头。推理时将判别器和 4W 头的目录分别填入 `discriminator.checkpoint` 和 `cooperative.head_checkpoint`。若 mmBERT 已迁移到其他目录，通过 `ANCHOR_ENCODER_DIR` 指定路径。
