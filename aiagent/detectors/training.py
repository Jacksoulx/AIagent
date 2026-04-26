import random
from typing import Dict, List

from aiagent.detectors.schemas import (
    FEATURE_NAMES,
    AnomalyDetectionResult,
    ConsensusMetrics,
    LabeledMetricsSample,
)


def _generate_normal_sample() -> LabeledMetricsSample:
    metrics: ConsensusMetrics = {
        "block_time_sec_avg": random.uniform(580.0, 620.0),
        "block_time_sec_std": random.uniform(20.0, 80.0),
        "fork_rate": random.uniform(0.0, 0.03),
        "orphan_rate": random.uniform(0.0, 0.03),
        "reorg_depth_max": random.uniform(0.0, 1.5),
        "hashrate_concentration_top1": random.uniform(0.18, 0.32),
        "hashrate_concentration_top3": random.uniform(0.45, 0.62),
        "miner_entropy": random.uniform(0.8, 1.2),
    }
    return {"metrics": metrics, "label": "normal"}


def _generate_abnormal_sample() -> LabeledMetricsSample:
    metrics: ConsensusMetrics = {
        "block_time_sec_avg": random.uniform(520.0, 700.0),
        "block_time_sec_std": random.uniform(90.0, 180.0),
        "fork_rate": random.uniform(0.06, 0.18),
        "orphan_rate": random.uniform(0.05, 0.14),
        "reorg_depth_max": random.uniform(3.0, 7.0),
        "hashrate_concentration_top1": random.uniform(0.4, 0.65),
        "hashrate_concentration_top3": random.uniform(0.7, 0.92),
        "miner_entropy": random.uniform(0.35, 0.78),
    }
    return {"metrics": metrics, "label": "abnormal"}


def generate_mock_training_samples(sample_count: int = 100) -> List[LabeledMetricsSample]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    normal_count = max(1, int(sample_count * 0.7))
    abnormal_count = sample_count - normal_count

    samples = [_generate_normal_sample() for _ in range(normal_count)]
    samples.extend(_generate_abnormal_sample() for _ in range(abnormal_count))
    random.shuffle(samples)
    return samples


def fit_statistical_baseline(samples: List[LabeledMetricsSample]) -> Dict[str, Dict[str, float]]:
    normal_samples = [sample for sample in samples if sample["label"] == "normal"]
    if not normal_samples:
        raise ValueError("At least one normal sample is required to fit the baseline.")

    baseline: Dict[str, Dict[str, float]] = {}
    for feature in FEATURE_NAMES:
        values = [float(sample["metrics"][feature]) for sample in normal_samples]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = max(variance ** 0.5, 1e-6)
        baseline[feature] = {"mean": mean, "std": std}
    return baseline


def score_with_statistical_baseline(
    metrics: ConsensusMetrics,
    baseline: Dict[str, Dict[str, float]],
    z_threshold: float = 2.5,
) -> AnomalyDetectionResult:
    flags: List[str] = []
    max_ratio = 0.0

    for feature in FEATURE_NAMES:
        feature_stats = baseline[feature]
        mean = feature_stats["mean"]
        std = feature_stats["std"]
        value = float(metrics[feature])
        z_score = abs(value - mean) / std
        max_ratio = max(max_ratio, z_score / z_threshold)
        if z_score >= z_threshold:
            flags.append(f"{feature} deviates strongly from normal baseline (z={z_score:.2f})")

    anomaly_score = max(0.0, min(max_ratio, 1.0))
    summary = "; ".join(flags) if flags else "Metrics remain within the learned statistical baseline."
    return {
        "anomaly_score": anomaly_score,
        "flags": flags,
        "summary": summary,
    }


def get_training_recommendations() -> Dict[str, List[str]]:
    return {
        "data": [
            "Collect longer time windows for consensus metrics to reduce noise.",
            "Label known abnormal intervals such as reorg events, fork storms, and miner concentration spikes.",
            "Track chain-specific baselines because healthy behavior differs across networks.",
        ],
        "modeling": [
            "Use the statistical baseline as an interpretable first detector.",
            "Add Isolation Forest for unsupervised anomaly scoring on tabular features.",
            "Later consider temporal models such as LSTM or autoencoders for multi-window behavior.",
        ],
    }
