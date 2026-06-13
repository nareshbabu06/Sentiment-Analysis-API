POSITIVE_LABELS = {"1", "pos", "positive", "true", "yes", "y"}
NEGATIVE_LABELS = {"0", "neg", "negative", "false", "no", "n"}


def normalize_binary_label(label) -> int:
    normalized = str(label).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    raise ValueError(
        "Unsupported label value. Expected one of "
        f"{sorted(POSITIVE_LABELS | NEGATIVE_LABELS)} but received {label!r}."
    )


def format_sentiment_label(label) -> str:
    return "Positive" if normalize_binary_label(label) == 1 else "Negative"
