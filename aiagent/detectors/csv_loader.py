from __future__ import annotations

import csv
from typing import List

from aiagent.detectors.schemas import ConsensusMetrics, LabeledMetricsSample
from aiagent.detectors.training import FEATURE_NAMES

REQUIRED_COLUMNS = FEATURE_NAMES


def _row_to_metrics(row: dict[str, str]) -> ConsensusMetrics:
    return {
        "block_time_sec_avg": float(row["block_time_sec_avg"]),
        "block_time_sec_std": float(row["block_time_sec_std"]),
        "fork_rate": float(row["fork_rate"]),
        "orphan_rate": float(row["orphan_rate"]),
        "reorg_depth_max": float(row["reorg_depth_max"]),
        "hashrate_concentration_top1": float(row["hashrate_concentration_top1"]),
        "hashrate_concentration_top3": float(row["hashrate_concentration_top3"]),
        "miner_entropy": float(row["miner_entropy"]),
    }


def load_metrics_samples_from_csv(csv_path: str, default_label: str = "unknown") -> List[LabeledMetricsSample]:
    samples: List[LabeledMetricsSample] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV file is missing required columns: {', '.join(missing)}")

        for row in reader:
            label = row.get("label", default_label).strip() or default_label
            samples.append(
                {
                    "metrics": _row_to_metrics(row),
                    "label": label,
                }
            )

    if not samples:
        raise ValueError("CSV file does not contain any samples.")

    return samples


def load_metrics_only_from_csv(csv_path: str) -> List[ConsensusMetrics]:
    return [sample["metrics"] for sample in load_metrics_samples_from_csv(csv_path)]
