from typing import Any, Dict, List, Optional

from aiagent.blockchain.rpc_client import (
    JsonRpcClient,
    RpcMalformedResponseError,
    RpcResponseError,
)
from aiagent.blockchain.schemas import EthereumBlockRecord
from aiagent.env_utils import get_eth_rpc_url


def _require_hex_int(value: Any, field_name: str) -> int:
    if not isinstance(value, str):
        raise RpcMalformedResponseError(
            f"Malformed Ethereum block field {field_name}: expected hex string."
        )
    try:
        return int(value, 16)
    except ValueError as exc:
        raise RpcMalformedResponseError(
            f"Malformed Ethereum block field {field_name}: invalid hex value {value!r}."
        ) from exc


def parse_ethereum_block(raw_block: Dict[str, Any]) -> EthereumBlockRecord:
    """Parse a raw `eth_getBlockByNumber` result into a typed block record."""
    required_fields = [
        "number",
        "hash",
        "parentHash",
        "timestamp",
        "gasUsed",
        "gasLimit",
        "transactions",
    ]
    missing_fields = [field for field in required_fields if field not in raw_block]
    if missing_fields:
        raise RpcMalformedResponseError(
            "Malformed Ethereum block response; missing fields: "
            + ", ".join(missing_fields)
        )

    transactions = raw_block["transactions"]
    if not isinstance(transactions, list):
        raise RpcMalformedResponseError(
            "Malformed Ethereum block response; transactions must be a list."
        )

    proposer_address = raw_block.get("miner") or raw_block.get("feeRecipient")
    if proposer_address is not None and not isinstance(proposer_address, str):
        raise RpcMalformedResponseError(
            "Malformed Ethereum block response; proposer address must be a string."
        )

    base_fee_value = raw_block.get("baseFeePerGas")
    base_fee_per_gas: Optional[int] = None
    if base_fee_value is not None:
        base_fee_per_gas = _require_hex_int(base_fee_value, "baseFeePerGas")

    block_hash = raw_block["hash"]
    parent_hash = raw_block["parentHash"]
    if not isinstance(block_hash, str) or not isinstance(parent_hash, str):
        raise RpcMalformedResponseError(
            "Malformed Ethereum block response; hash and parentHash must be strings."
        )

    return EthereumBlockRecord(
        number=_require_hex_int(raw_block["number"], "number"),
        hash=block_hash,
        parent_hash=parent_hash,
        timestamp=_require_hex_int(raw_block["timestamp"], "timestamp"),
        proposer_address=proposer_address,
        gas_used=_require_hex_int(raw_block["gasUsed"], "gasUsed"),
        gas_limit=_require_hex_int(raw_block["gasLimit"], "gasLimit"),
        transaction_count=len(transactions),
        base_fee_per_gas=base_fee_per_gas,
    )


class EthereumClient:
    """Ethereum JSON-RPC wrapper focused on recent canonical block telemetry."""

    def __init__(self, rpc_client: JsonRpcClient) -> None:
        self.rpc_client = rpc_client

    @classmethod
    def from_env(
        cls,
        timeout_sec: float = 10.0,
        max_retries: int = 2,
        retry_backoff_sec: float = 0.5,
    ) -> "EthereumClient":
        """Create a client using `ETH_RPC_URL` from the environment."""
        return cls(
            JsonRpcClient(
                get_eth_rpc_url(),
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
            )
        )

    def get_latest_block_number(self) -> int:
        """Return the latest canonical Ethereum block number."""
        result = self.rpc_client.call("eth_blockNumber")
        return _require_hex_int(result, "eth_blockNumber")

    def get_block_by_number(
        self,
        block_number: int,
        full_transactions: bool = False,
    ) -> EthereumBlockRecord:
        """Fetch one canonical block by number."""
        if block_number < 0:
            raise ValueError("block_number must be non-negative")

        result = self.rpc_client.call(
            "eth_getBlockByNumber",
            [hex(block_number), full_transactions],
        )
        if result is None:
            raise RpcResponseError(f"Ethereum block {block_number} was not found.")
        if not isinstance(result, dict):
            raise RpcMalformedResponseError(
                f"Ethereum block {block_number} response was not a JSON object."
            )

        return parse_ethereum_block(result)

    def get_recent_blocks(self, n: int = 100) -> List[EthereumBlockRecord]:
        """Fetch the latest `n` canonical blocks in ascending order."""
        if n <= 0:
            raise ValueError("n must be positive")

        latest_block = self.get_latest_block_number()
        start_block = max(0, latest_block - n + 1)
        blocks: List[EthereumBlockRecord] = []
        for block_number in range(start_block, latest_block + 1):
            blocks.append(self.get_block_by_number(block_number))
        return blocks
