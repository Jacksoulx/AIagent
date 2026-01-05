from typing import Dict, Any

from langchain_core.tools import tool


@tool
def simple_consensus_anomaly_detector(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Perform a simple rule-based anomaly assessment on consensus metrics.

    Expects the same keys as returned by get_mock_chain_metrics and returns
    a coarse anomaly score together with human-readable flags.
    """
    flags = []
    score = 0.0

    block_time_avg = metrics.get("block_time_sec_avg", 600.0)
    fork_rate = metrics.get("fork_rate", 0.0)
    orphan_rate = metrics.get("orphan_rate", 0.0)
    reorg_depth_max = metrics.get("reorg_depth_max", 0.0)
    top1 = metrics.get("hashrate_concentration_top1", 0.0)

    # Very rough rules, only for demonstration purposes.
    if fork_rate > 0.05:
        flags.append("elevated fork rate")
        score += 0.3

    if orphan_rate > 0.05:
        flags.append("elevated orphan rate")
        score += 0.3

    if reorg_depth_max >= 3:
        flags.append("deep recent reorgs")
        score += 0.2

    if top1 > 0.4:
        flags.append("hashrate highly concentrated in top miner")
        score += 0.2

    if block_time_avg < 550 or block_time_avg > 650:
        flags.append("block time deviates from target")
        score += 0.1

    score = min(score, 1.0)

    return {
        "anomaly_score": score,
        "flags": flags,
        "summary": (
            "; ".join(flags)
            if flags
            else "No strong anomaly indicators based on simple rules."
        ),
    }
