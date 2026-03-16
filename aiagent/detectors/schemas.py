from typing import List, TypedDict


class ConsensusMetrics(TypedDict):
    block_time_sec_avg: float
    block_time_sec_std: float
    fork_rate: float
    orphan_rate: float
    reorg_depth_max: float
    hashrate_concentration_top1: float
    hashrate_concentration_top3: float
    miner_entropy: float


class AnomalyDetectionResult(TypedDict):
    anomaly_score: float
    flags: List[str]
    summary: str


class LabeledMetricsSample(TypedDict):
    metrics: ConsensusMetrics
    label: str
