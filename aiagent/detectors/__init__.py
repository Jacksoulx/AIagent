"""Anomaly detection and security monitoring models."""

from aiagent.detectors.schemas import (
    FEATURE_NAMES,
    AnomalyDetectionResult,
    ConsensusMetrics,
    LabeledMetricsSample,
)
from aiagent.detectors.simple_anomaly import simple_consensus_anomaly_detector
from aiagent.detectors.csv_loader import (
    load_metrics_only_from_csv,
    load_metrics_samples_from_csv,
)
from aiagent.detectors.isolation_forest_detector import (
    fit_isolation_forest,
    metrics_to_feature_vector,
    samples_to_feature_matrix,
    score_metrics_with_isolation_forest,
    score_samples_with_isolation_forest,
)
from aiagent.detectors.evaluation import evaluate_detection_results
from aiagent.detectors.training import (
    FEATURE_NAMES,
    fit_statistical_baseline,
    generate_mock_training_samples,
    get_training_recommendations,
    score_with_statistical_baseline,
)

__all__ = [
    "FEATURE_NAMES",
    "AnomalyDetectionResult",
    "ConsensusMetrics",
    "LabeledMetricsSample",
    "FEATURE_NAMES",
    "simple_consensus_anomaly_detector",
    "load_metrics_only_from_csv",
    "load_metrics_samples_from_csv",
    "fit_isolation_forest",
    "metrics_to_feature_vector",
    "samples_to_feature_matrix",
    "score_metrics_with_isolation_forest",
    "score_samples_with_isolation_forest",
    "evaluate_detection_results",
    "generate_mock_training_samples",
    "fit_statistical_baseline",
    "score_with_statistical_baseline",
    "get_training_recommendations",
]
