import json

import pytest

from src import config, evaluation


def test_compute_metrics_macro_average():
    metrics = evaluation.compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == pytest.approx(0.75)
    # Negative: P=1.0 R=0.5 ; Positive: P=2/3 R=1.0
    assert metrics["precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["recall"] == pytest.approx(0.75)


def test_benchmark_table_uses_saved_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    (tmp_path / "metrics_bilstm.json").write_text(
        json.dumps({"accuracy": 0.866, "precision": 0.8666, "recall": 0.866, "f1": 0.8659})
    )
    table = evaluation.benchmark_table(["BiLSTM", "DistilBERT"])
    assert "| BiLSTM | 86.60% | 86.66% | 86.60% | 86.59% |" in table
    assert "| DistilBERT | TBD | TBD | TBD | TBD |" in table
