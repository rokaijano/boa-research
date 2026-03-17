from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from boaresearch.metrics import extract_metrics
from boaresearch.schema import MetricConfig


class MetricTests(unittest.TestCase):
    def test_extract_json_regex_and_metric_file(self) -> None:
        artifact_dir = Path(tempfile.mkdtemp())
        (artifact_dir / "command_001.stdout").write_text("accuracy=0.93\n", encoding="utf-8")
        (artifact_dir / "command_001.stderr").write_text("loss=0.12\n", encoding="utf-8")
        captured = artifact_dir / "captured" / "reports"
        captured.mkdir(parents=True, exist_ok=True)
        (captured / "metrics.json").write_text(json.dumps({"accuracy": 0.91}), encoding="utf-8")
        (captured / "loss.txt").write_text("loss=0.11\n", encoding="utf-8")
        metrics = extract_metrics(
            artifact_dir=artifact_dir,
            metrics=[
                MetricConfig(name="accuracy", source="json_file", path="reports/metrics.json", json_key="accuracy"),
                MetricConfig(name="log_accuracy", source="regex", pattern=r"accuracy=([0-9.]+)", target="stdout"),
                MetricConfig(name="loss", source="metric_file", path="reports/loss.txt"),
            ],
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.91)
        self.assertAlmostEqual(metrics["log_accuracy"], 0.93)
        self.assertAlmostEqual(metrics["loss"], 0.11)


if __name__ == "__main__":
    unittest.main()
