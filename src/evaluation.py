"""Model evaluation and benchmarking on the shared, independent test set.

Run ``python -m src.evaluation`` to evaluate every trained model through the
same inference pipeline the app uses and regenerate:

    reports/metrics_<model>.json
    reports/figures/confusion_matrix_<model>.png
    reports/benchmark.md
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src import config

LABEL_NAMES = [config.ID2LABEL[i] for i in range(config.NUM_CLASSES)]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Accuracy plus macro-averaged Precision / Recall / F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def report(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    return classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    title: str,
    save_path=None,
    ax=None,
):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=150)
    return ax


def save_metrics(model_name: str, metrics: dict, extra: dict | None = None) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, **metrics, **(extra or {})}
    path = config.REPORTS_DIR / f"metrics_{model_name.lower()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def benchmark_table(model_names: Sequence[str]) -> str:
    """Markdown benchmark table built from saved ``metrics_<model>.json`` files."""
    lines = [
        "| Model | Accuracy | Precision | Recall | F1-Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in model_names:
        path = config.REPORTS_DIR / f"metrics_{name.lower()}.json"
        if not path.exists():
            lines.append(f"| {name} | TBD | TBD | TBD | TBD |")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"| {name} | {m['accuracy']:.2%} | {m['precision']:.2%} "
            f"| {m['recall']:.2%} | {m['f1']:.2%} |"
        )
    return "\n".join(lines)


def evaluate_predictor(model_name: str, texts: Sequence[str], labels: Sequence[int]) -> dict:
    """Evaluate one registered model on the given texts; save metrics and figure."""
    from src.inference import get_predictor

    probs = get_predictor(model_name).predict_proba(list(texts))
    y_pred = probs.argmax(axis=1).tolist()
    metrics = compute_metrics(labels, y_pred)
    print(f"\n=== {model_name} ===\n{report(labels, y_pred)}")
    save_metrics(model_name, metrics, {"test_samples": len(labels)})
    plot_confusion_matrix(
        labels,
        y_pred,
        f"{model_name} - Test Set",
        config.FIGURES_DIR / f"confusion_matrix_{model_name.lower()}.png",
    )
    plt.close("all")
    return metrics


def main() -> None:
    from src.data_utils import prepare_splits
    from src.inference import AVAILABLE_MODELS

    matplotlib.use("Agg")  # headless: figures are only saved to disk
    test = prepare_splits()["test"]
    evaluated = []
    for name in AVAILABLE_MODELS:
        try:
            evaluate_predictor(name, test["text"].tolist(), test["label"].tolist())
            evaluated.append(name)
        except FileNotFoundError as exc:
            print(f"Skipping {name}: {exc}")

    table = benchmark_table(AVAILABLE_MODELS)
    (config.REPORTS_DIR / "benchmark.md").write_text(
        f"Test set: {len(test)} reviews (balanced, never used for training or tuning).\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    print(f"\n{table}")


if __name__ == "__main__":
    main()
