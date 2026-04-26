"""Blockchain-related integrations: node clients, data access, simulators."""

from aiagent.blockchain.ethereum_client import EthereumClient, parse_ethereum_block
from aiagent.blockchain.metrics_builder import (
    build_ethereum_metrics_report,
    build_rolling_ethereum_metric_reports,
    metrics_report_to_csv_row,
)
from aiagent.blockchain.mock_chain import get_mock_chain_metrics
from aiagent.blockchain.schemas import EthereumBlockRecord, MetricDetail, MetricsBuildResult

__all__ = [
    "EthereumClient",
    "EthereumBlockRecord",
    "MetricDetail",
    "MetricsBuildResult",
    "build_ethereum_metrics_report",
    "build_rolling_ethereum_metric_reports",
    "metrics_report_to_csv_row",
    "get_mock_chain_metrics",
    "parse_ethereum_block",
]
