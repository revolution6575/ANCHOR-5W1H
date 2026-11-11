import statistics

ROLES = ("WHY", "HOW")
MODES = ("position", "soft", "exact", "character")


def objective(scores, gold_counts):
    def quality(group, role):
        if gold_counts.get(f"{group}/{role}", 0) == 0:
            return None
        return statistics.mean(scores[f"{group}/{m}/{role}"]["F1"] for m in MODES)

    overall = [q for r in ROLES if (q := quality("ALL", r)) is not None]
    languages = [
        q
        for lang in ("zh", "en", "es")
        for r in ROLES
        if (q := quality("lang:" + lang, r)) is not None
    ]
    if not overall:
        return 0.0
    return 0.5 * statistics.mean(overall) + 0.5 * (
        statistics.mean(languages) if languages else statistics.mean(overall)
    )
