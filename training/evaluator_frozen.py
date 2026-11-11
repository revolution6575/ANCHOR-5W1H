from __future__ import annotations
from collections import Counter
from dataclasses import dataclass

ROLES = ("WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW")
CATEGORIES = ("Ca", "Pa", "Ia", "Ma", "Sa")


@dataclass(frozen=True)
class Tag:
    text: str
    role: str
    start: int
    end: int
    uid: int


def prediction_tags(values, text):
    if values is None:
        return [], 0, 0
    if not isinstance(values, dict):
        return [], 0, 1
    unique = set()
    invalid = 0
    for role in ROLES:
        part = values.get(role, [])
        if not isinstance(part, list):
            invalid += 1
            continue
        for value in part:
            if not isinstance(value, str):
                invalid += 1
                continue
            value = value.strip()
            if value:
                unique.add((value, role))
    tags = []
    unlocated = 0
    for value, role in sorted(unique, key=lambda pair: (pair[1], pair[0])):
        pos = text.find(value)
        if pos < 0:
            pos = text.lower().find(value.lower())
        if pos < 0:
            unlocated += 1
            continue
        tags.append(
            Tag(text[pos : pos + len(value)], role, pos, pos + len(value), len(tags))
        )
    return tags, unlocated, invalid


def position_counts(real, predicted):
    counts = {role: Counter() for role in ROLES}
    real = sorted(real, key=lambda tag: (tag.start, tag.role))
    predicted = sorted(predicted, key=lambda tag: (tag.start, tag.role))

    for gold in real.copy():
        for pred in predicted.copy():
            if (
                gold.start == pred.start
                and gold.end == pred.end
                and gold.role == pred.role
            ):
                counts[gold.role]["Ca"] += 1
                real.remove(gold)
                predicted.remove(pred)
                break

    for gold in real.copy():
        partial = []
        for pred in predicted.copy():
            if (
                (pred.start >= gold.start and pred.end <= gold.end)
                or (
                    pred.start <= gold.start
                    and pred.end >= gold.start
                    and pred.end <= gold.end
                )
                or (
                    pred.start >= gold.start
                    and pred.start <= gold.end
                    and pred.end >= gold.end
                )
                or (pred.start <= gold.start and pred.end >= gold.end)
            ):
                partial.append(pred)
        partial.sort(key=lambda tag: tag.end - tag.start)
        for pred in partial:
            if pred.role == gold.role:
                counts[gold.role]["Pa"] += 1
                real.remove(gold)
                predicted.remove(pred)
                break

    for gold in real.copy():
        for pred in predicted.copy():
            if (
                gold.start == pred.start
                and gold.end == pred.end
                and gold.role != pred.role
            ):
                counts[gold.role]["Ia"] += 1
                real.remove(gold)
                predicted.remove(pred)
                break

    for gold in real:
        counts[gold.role]["Ma"] += 1
    for pred in predicted:
        counts[pred.role]["Sa"] += 1
    return counts


def overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def soft_counts(real, predicted):
    counts = {role: Counter() for role in ROLES}
    pairs = []
    for pi, pred in enumerate(predicted):
        for gi, gold in enumerate(real):
            score = overlap(pred.text, gold.text)
            if pred.role == gold.role and score >= 0.5:
                pairs.append((score, pi, gi, pred.role))
            elif pred.role != gold.role and score >= 0.8:
                pairs.append((-score, pi, gi, gold.role))
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    used_pred, used_gold = set(), set()
    for score, pi, gi, role in pairs:
        if pi in used_pred or gi in used_gold:
            continue
        counts[role]["Ca" if score >= 0.8 else "Pa" if score > 0 else "Ia"] += 1
        used_pred.add(pi)
        used_gold.add(gi)
    for gi, gold in enumerate(real):
        if gi not in used_gold:
            counts[gold.role]["Ma"] += 1
    for pi, pred in enumerate(predicted):
        if pi not in used_pred:
            counts[pred.role]["Sa"] += 1
    return counts


def metrics(counts):
    ca, pa, ia, ma, sa = (counts[k] for k in CATEGORIES)
    hit = ca + 0.5 * pa
    p = hit / (ca + pa + ia + sa) if ca + pa + ia + sa else 0.0
    r = hit / (ca + pa + ia + ma) if ca + pa + ia + ma else 0.0
    a = hit / (ca + pa + ia + ma + sa) if ca + pa + ia + ma + sa else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"Precision": p, "Recall": r, "F1": f, "ACC": a}
