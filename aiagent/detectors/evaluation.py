from collections import Counter, defaultdict
from typing import Any, Dict, List


def evaluate_detection_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("No detection results were provided for evaluation.")

    total_samples = len(results)
    anomalous_predictions = sum(1 for item in results if item["prediction"] == "anomalous")
    normal_like_predictions = total_samples - anomalous_predictions

    label_counts = Counter(item["label"] for item in results)
    prediction_counts = Counter(item["prediction"] for item in results)

    by_label = defaultdict(lambda: {"total": 0, "anomalous": 0, "normal_like": 0})
    for item in results:
        label = item["label"]
        prediction = item["prediction"]
        by_label[label]["total"] += 1
        by_label[label][prediction] += 1

    per_label_detection_rate = {}
    for label, counts in by_label.items():
        total = counts["total"]
        per_label_detection_rate[label] = {
            "total": total,
            "anomalous": counts["anomalous"],
            "normal_like": counts["normal_like"],
            "anomalous_rate": counts["anomalous"] / total if total else 0.0,
        }

    return {
        "total_samples": total_samples,
        "anomalous_predictions": anomalous_predictions,
        "normal_like_predictions": normal_like_predictions,
        "anomalous_prediction_rate": anomalous_predictions / total_samples,
        "label_counts": dict(label_counts),
        "prediction_counts": dict(prediction_counts),
        "per_label_detection_rate": per_label_detection_rate,
    }
