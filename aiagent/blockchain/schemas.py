from typing import Any, Dict, List, Optional

from aiagent.detectors.schemas import ConsensusMetrics


class EthereumBlockRecord:
    """Canonical block fields extracted from Ethereum JSON-RPC responses."""

    def __init__(
        self,
        number,
        hash,
        parent_hash,
        timestamp,
        proposer_address,
        gas_used,
        gas_limit,
        transaction_count,
        base_fee_per_gas,
    ):
        self.number = number
        self.hash = hash
        self.parent_hash = parent_hash
        self.timestamp = timestamp
        self.proposer_address = proposer_address
        self.gas_used = gas_used
        self.gas_limit = gas_limit
        self.transaction_count = transaction_count
        self.base_fee_per_gas = base_fee_per_gas

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "hash": self.hash,
            "parentHash": self.parent_hash,
            "timestamp": self.timestamp,
            "miner": self.proposer_address,
            "gasUsed": self.gas_used,
            "gasLimit": self.gas_limit,
            "transaction_count": self.transaction_count,
            "baseFeePerGas": self.base_fee_per_gas,
        }


class MetricDetail:
    """Describes how a metric was obtained and how reliable it is."""

    def __init__(self, value, method, reliability, note):
        self.value = value
        self.method = method
        self.reliability = reliability
        self.note = note

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "method": self.method,
            "reliability": self.reliability,
            "note": self.note,
        }


class MetricsBuildResult:
    """Combined detector-compatible metrics plus reproducibility metadata."""

    def __init__(
        self,
        chain,
        source,
        metrics,
        metric_details,
        block_count,
        start_block,
        end_block,
        start_timestamp,
        end_timestamp,
        notes=None,
        activity_summary=None,
    ):
        self.chain = chain
        self.source = source
        self.metrics = metrics
        self.metric_details = metric_details
        self.block_count = block_count
        self.start_block = start_block
        self.end_block = end_block
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.notes = list(notes or [])
        self.activity_summary = dict(activity_summary or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain": self.chain,
            "source": self.source,
            "metrics": dict(self.metrics),
            "metric_details": {
                name: detail.to_dict() for name, detail in self.metric_details.items()
            },
            "block_count": self.block_count,
            "start_block": self.start_block,
            "end_block": self.end_block,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "notes": list(self.notes),
            "activity_summary": dict(self.activity_summary),
        }
