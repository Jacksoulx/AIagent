from typing import Any, Dict, List

from sklearn.ensemble import IsolationForest

from aiagent.detectors.schemas import (
    FEATURE_NAMES,
    AnomalyDetectionResult,
    ConsensusMetrics,
    LabeledMetricsSample,
)


def metrics_to_feature_vector(metrics: ConsensusMetrics) -> List[float]:
    return [float(metrics[feature]) for feature in FEATURE_NAMES]


def samples_to_feature_matrix(samples: List[LabeledMetricsSample]) -> List[List[float]]:
    return [metrics_to_feature_vector(sample["metrics"]) for sample in samples]


def fit_isolation_forest(
    samples: List[LabeledMetricsSample],
    contamination: float = 0.1,
    random_state: int = 42,
) -> IsolationForest:
    normal_samples = [sample for sample in samples if sample["label"] == "normal"]
    training_samples = normal_samples if normal_samples else samples
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    model.fit(samples_to_feature_matrix(training_samples))
    return model


def score_metrics_with_isolation_forest(
    metrics: ConsensusMetrics,
    model: IsolationForest,
) -> AnomalyDetectionResult:
    vector = [metrics_to_feature_vector(metrics)]
    raw_score = float(-model.score_samples(vector)[0])
    prediction = int(model.predict(vector)[0])
    anomaly_score = max(0.0, min(raw_score / 0.8, 1.0))

    flags: List[str] = []
    if prediction == -1:
        flags.append("Isolation Forest marked this window as anomalous")
    if metrics["fork_rate"] > 0.05:
        flags.append("fork_rate is elevated")
    if metrics["orphan_rate"] > 0.05:
        flags.append("orphan_rate is elevated")
    if metrics["reorg_depth_max"] >= 3:
        flags.append("reorg_depth_max indicates deep reorganizations")

    summary = "; ".join(flags) if flags else "Isolation Forest considers this sample close to the fitted normal region."
    return {
        "anomaly_score": anomaly_score,
        "flags": flags,
        "summary": summary,
    }


def score_samples_with_isolation_forest(
    samples: List[LabeledMetricsSample],
    model: IsolationForest,
) -> List[Dict[str, Any]]:
    matrix = samples_to_feature_matrix(samples)
    raw_scores = model.score_samples(matrix)
    predictions = model.predict(matrix)

    results: List[Dict[str, Any]] = []
    for sample, raw_score, prediction in zip(samples, raw_scores, predictions):
        anomaly_score = max(0.0, min(float(-raw_score) / 0.8, 1.0))
        results.append(
            {
                "label": sample["label"],
                "prediction": "anomalous" if int(prediction) == -1 else "normal_like",
                "anomaly_score": anomaly_score,
                "metrics": sample["metrics"],
            }
        )
    return results
