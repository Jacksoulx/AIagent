import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from aiagent.blockchain.schemas import EthereumBlockRecord, MetricDetail, MetricsBuildResult
from aiagent.detectors.schemas import ConsensusMetrics, FEATURE_NAMES


def _normalized_proposer_address(block: EthereumBlockRecord) -> str:
    return (block.proposer_address or "unknown").lower()


def _validate_block_sequence_for_metrics(
    blocks: Sequence[EthereumBlockRecord],
) -> List[EthereumBlockRecord]:
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required to build Ethereum metrics.")

    ordered_blocks = sorted(blocks, key=lambda block: block.number)
    block_numbers = [block.number for block in ordered_blocks]
    if len(set(block_numbers)) != len(block_numbers):
        raise ValueError("Ethereum block window contains duplicate block numbers.")

    for block in ordered_blocks:
        if not isinstance(block.timestamp, int) or block.timestamp <= 0:
            raise ValueError(
                f"Ethereum block {block.number} has an unusable timestamp: {block.timestamp!r}."
            )

    deltas = [
        ordered_blocks[index].timestamp - ordered_blocks[index - 1].timestamp
        for index in range(1, len(ordered_blocks))
    ]
    if any(delta <= 0 for delta in deltas):
        raise ValueError(
            "Ethereum block timestamps are not strictly increasing; cannot compute usable block-time metrics."
        )

    return ordered_blocks


def compute_block_time_deltas(blocks: Sequence[EthereumBlockRecord]) -> List[int]:
    """Compute consecutive timestamp deltas from an ordered block sequence."""
    if len(blocks) < 2:
        return []

    ordered_blocks = sorted(blocks, key=lambda block: block.number)
    return [
        ordered_blocks[index].timestamp - ordered_blocks[index - 1].timestamp
        for index in range(1, len(ordered_blocks))
    ]


def compute_proposer_concentration_metrics(
    blocks: Sequence[EthereumBlockRecord],
) -> Dict[str, float]:
    """Compute proposer concentration proxies over a block window."""
    if not blocks:
        return {
            "hashrate_concentration_top1": 0.0,
            "hashrate_concentration_top3": 0.0,
            "miner_entropy": 0.0,
        }

    proposer_counts = Counter(_normalized_proposer_address(block) for block in blocks)
    total_blocks = len(blocks)
    sorted_counts = sorted(proposer_counts.values(), reverse=True)

    top1 = sorted_counts[0] / total_blocks
    top3 = sum(sorted_counts[:3]) / total_blocks

    entropy = 0.0
    for count in proposer_counts.values():
        probability = count / total_blocks
        entropy -= probability * math.log2(probability)

    return {
        "hashrate_concentration_top1": top1,
        "hashrate_concentration_top3": top3,
        "miner_entropy": entropy,
    }


def build_ethereum_metrics_report(
    blocks: Sequence[EthereumBlockRecord],
    source: str = "ethereum_json_rpc",
) -> MetricsBuildResult:
    """Build detector-compatible metrics plus metadata from canonical Ethereum blocks."""
    ordered_blocks = _validate_block_sequence_for_metrics(blocks)
    deltas = compute_block_time_deltas(ordered_blocks)
    proposer_metrics = compute_proposer_concentration_metrics(ordered_blocks)

    metrics: ConsensusMetrics = {
        "block_time_sec_avg": statistics.mean(deltas),
        "block_time_sec_std": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "fork_rate": 0.0,
        "orphan_rate": 0.0,
        "reorg_depth_max": 0.0,
        "hashrate_concentration_top1": proposer_metrics["hashrate_concentration_top1"],
        "hashrate_concentration_top3": proposer_metrics["hashrate_concentration_top3"],
        "miner_entropy": proposer_metrics["miner_entropy"],
    }

    metric_details = {
        "block_time_sec_avg": MetricDetail(
            value=metrics["block_time_sec_avg"],
            method="observed_from_canonical_block_timestamps",
            reliability="observed",
            note="Computed from consecutive canonical block timestamps.",
        ),
        "block_time_sec_std": MetricDetail(
            value=metrics["block_time_sec_std"],
            method="observed_from_canonical_block_timestamps",
            reliability="observed",
            note="Population standard deviation of consecutive canonical block timestamps.",
        ),
        "fork_rate": MetricDetail(
            value=metrics["fork_rate"],
            method="not_observable_from_canonical_rpc_window",
            reliability="placeholder",
            note="Canonical RPC block queries do not reveal transient fork branches.",
        ),
        "orphan_rate": MetricDetail(
            value=metrics["orphan_rate"],
            method="not_observable_from_canonical_rpc_window",
            reliability="placeholder",
            note="Canonical RPC block queries do not expose orphaned blocks.",
        ),
        "reorg_depth_max": MetricDetail(
            value=metrics["reorg_depth_max"],
            method="not_observable_from_canonical_rpc_window",
            reliability="placeholder",
            note="Detecting reorg depth requires chain history tracking beyond a simple canonical window.",
        ),
        "hashrate_concentration_top1": MetricDetail(
            value=metrics["hashrate_concentration_top1"],
            method="proposer_frequency_proxy_from_fee_recipient",
            reliability="proxy",
            note="Uses fee recipient / proposer address concentration as a proxy, not PoW hashrate concentration.",
        ),
        "hashrate_concentration_top3": MetricDetail(
            value=metrics["hashrate_concentration_top3"],
            method="proposer_frequency_proxy_from_fee_recipient",
            reliability="proxy",
            note="Uses top-3 fee recipient / proposer frequency as a proxy, not PoW hashrate concentration.",
        ),
        "miner_entropy": MetricDetail(
            value=metrics["miner_entropy"],
            method="proposer_frequency_proxy_from_fee_recipient",
            reliability="proxy",
            note="Shannon entropy over fee recipient / proposer addresses observed in canonical blocks.",
        ),
    }

    gas_utilization_values = [
        (block.gas_used / block.gas_limit) for block in ordered_blocks if block.gas_limit > 0
    ]
    observed_base_fees = [
        block.base_fee_per_gas
        for block in ordered_blocks
        if block.base_fee_per_gas is not None
    ]

    activity_summary: Dict[str, Any] = {
        "transaction_count_avg": statistics.mean(
            block.transaction_count for block in ordered_blocks
        ),
        "gas_used_avg": statistics.mean(block.gas_used for block in ordered_blocks),
        "gas_limit_avg": statistics.mean(block.gas_limit for block in ordered_blocks),
        "gas_utilization_avg": (
            statistics.mean(gas_utilization_values) if gas_utilization_values else 0.0
        ),
    }
    if observed_base_fees:
        activity_summary["base_fee_per_gas_avg"] = statistics.mean(observed_base_fees)

    return MetricsBuildResult(
        chain="ethereum",
        source=source,
        metrics=metrics,
        metric_details=metric_details,
        block_count=len(ordered_blocks),
        start_block=ordered_blocks[0].number,
        end_block=ordered_blocks[-1].number,
        start_timestamp=ordered_blocks[0].timestamp,
        end_timestamp=ordered_blocks[-1].timestamp,
        notes=[
            "Ethereum proposer concentration fields are proxies derived from fee recipient / proposer addresses.",
            "Fork/orphan/reorg metrics are placeholders for canonical RPC-only telemetry and should not be over-interpreted.",
        ],
        activity_summary=activity_summary,
    )


def build_rolling_ethereum_metric_reports(
    blocks: Sequence[EthereumBlockRecord],
    window_size: int = 100,
    step: Optional[int] = None,
    source: str = "ethereum_json_rpc",
) -> List[MetricsBuildResult]:
    """Build one or more rolling-window metric reports from recent Ethereum blocks."""
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required to build rolling metrics.")
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1")
    if step is not None and step <= 0:
        raise ValueError("step must be positive")

    ordered_blocks = _validate_block_sequence_for_metrics(blocks)
    effective_window_size = min(window_size, len(ordered_blocks))
    effective_step = step or max(1, effective_window_size // 2)

    if len(ordered_blocks) == effective_window_size:
        return [build_ethereum_metrics_report(ordered_blocks, source=source)]

    reports: List[MetricsBuildResult] = []
    start_index = 0
    while start_index + effective_window_size <= len(ordered_blocks):
        window = ordered_blocks[start_index : start_index + effective_window_size]
        reports.append(build_ethereum_metrics_report(window, source=source))
        start_index += effective_step

    last_window = ordered_blocks[-effective_window_size:]
    if reports[-1].end_block != last_window[-1].number:
        reports.append(build_ethereum_metrics_report(last_window, source=source))

    return reports


def metrics_report_to_csv_row(report: MetricsBuildResult) -> Dict[str, Any]:
    """Convert one metrics report into a CSV row with detector-required columns."""
    row: Dict[str, Any] = {feature: report.metrics[feature] for feature in FEATURE_NAMES}
    row.update(
        {
            "chain": report.chain,
            "start_block": report.start_block,
            "end_block": report.end_block,
            "start_timestamp": report.start_timestamp,
            "end_timestamp": report.end_timestamp,
            "source": report.source,
            "metric_notes": " | ".join(report.notes),
        }
    )
    return row
