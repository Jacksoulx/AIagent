from typing import Dict

from langchain_core.tools import tool


@tool
def get_mock_chain_metrics(chain: str = "mockchain") -> Dict[str, float]:
    """Return mock consensus-level metrics for a blockchain.

    This is a placeholder for real node-derived metrics, useful for testing
    anomaly detection and explanation flows without connecting to a live node.
    """
    # Simple hard-coded example; you can later replace this with real data
    # from a node, database, or simulator.
    return {
        "block_time_sec_avg": 610.0,
        "block_time_sec_std": 120.0,
        "fork_rate": 0.08,
        "orphan_rate": 0.06,
        "reorg_depth_max": 4.0,
        "hashrate_concentration_top1": 0.42,
        "hashrate_concentration_top3": 0.75,
        "miner_entropy": 0.72,
    }
