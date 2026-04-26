from typing import Any, Dict, List, Tuple

FEATURE_NAMES: Tuple[str, ...] = (
    "block_time_sec_avg",
    "block_time_sec_std",
    "fork_rate",
    "orphan_rate",
    "reorg_depth_max",
    "hashrate_concentration_top1",
    "hashrate_concentration_top3",
    "miner_entropy",
)


ConsensusMetrics = Dict[str, float]
AnomalyDetectionResult = Dict[str, Any]
LabeledMetricsSample = Dict[str, Any]
MetricsArtifact = Dict[str, Any]
