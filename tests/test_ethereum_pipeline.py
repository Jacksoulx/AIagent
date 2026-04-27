import csv
import json

import pytest
import requests

from aiagent.agents import research_agent
from aiagent.agents.research_agent import collect_eth_metrics_to_csv
from aiagent.blockchain.ethereum_client import parse_ethereum_block
from aiagent.blockchain.metrics_builder import (
    build_ethereum_metrics_report,
    compute_proposer_concentration_metrics,
)
from aiagent.blockchain.rpc_client import JsonRpcClient, RpcTimeoutError
from aiagent.blockchain.schemas import EthereumBlockRecord
from aiagent.detectors.schemas import FEATURE_NAMES
from aiagent.env_utils import get_eth_rpc_url


def _make_block(
    number: int,
    timestamp: int,
    proposer: str,
    gas_used: int = 15_000_000,
    gas_limit: int = 30_000_000,
    transaction_count: int = 10,
    base_fee_per_gas=1_000_000_000,
) -> EthereumBlockRecord:
    return EthereumBlockRecord(
        number=number,
        hash=f"0x{number:064x}",
        parent_hash=f"0x{max(0, number - 1):064x}",
        timestamp=timestamp,
        proposer_address=proposer,
        gas_used=gas_used,
        gas_limit=gas_limit,
        transaction_count=transaction_count,
        base_fee_per_gas=base_fee_per_gas,
    )


class _FakeRpcResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeRpcSession:
    def __init__(self, responses=None, exception=None):
        self._responses = list(responses or [])
        self._exception = exception
        self.calls = 0

    def post(self, url, json, timeout):
        self.calls += 1
        if self._exception is not None:
            raise self._exception
        return self._responses.pop(0)


def test_parse_ethereum_block_valid_response() -> None:
    raw_block = {
        "number": "0x10",
        "hash": "0xabc",
        "parentHash": "0xdef",
        "timestamp": "0x65",
        "miner": "0x1234",
        "gasUsed": "0x5208",
        "gasLimit": "0x1c9c380",
        "transactions": ["0xtx1", "0xtx2"],
        "baseFeePerGas": "0x3b9aca00",
    }

    block = parse_ethereum_block(raw_block)

    assert block.number == 16
    assert block.hash == "0xabc"
    assert block.parent_hash == "0xdef"
    assert block.timestamp == 101
    assert block.proposer_address == "0x1234"
    assert block.gas_used == 21000
    assert block.transaction_count == 2
    assert block.base_fee_per_gas == 1_000_000_000


def test_metrics_builder_computes_block_time_average_and_std() -> None:
    blocks = [
        _make_block(100, 1000, "0xaaa"),
        _make_block(101, 1012, "0xaaa"),
        _make_block(102, 1026, "0xbbb"),
    ]

    report = build_ethereum_metrics_report(blocks)

    assert report.metrics["block_time_sec_avg"] == pytest.approx(13.0)
    assert report.metrics["block_time_sec_std"] == pytest.approx(1.0)


def test_proposer_concentration_and_entropy_are_computed_correctly() -> None:
    blocks = [
        _make_block(1, 100, "0xaaa"),
        _make_block(2, 112, "0xaaa"),
        _make_block(3, 124, "0xbbb"),
        _make_block(4, 136, "0xccc"),
    ]

    metrics = compute_proposer_concentration_metrics(blocks)

    assert metrics["hashrate_concentration_top1"] == pytest.approx(0.5)
    assert metrics["hashrate_concentration_top3"] == pytest.approx(1.0)
    assert metrics["miner_entropy"] == pytest.approx(1.5)


def test_missing_eth_rpc_url_produces_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("ETH_RPC_URL", raising=False)

    with pytest.raises(RuntimeError, match="ETH_RPC_URL"):
        get_eth_rpc_url()


def test_dry_run_report_saves_artifacts_and_skips_openai_key(
    monkeypatch,
    tmp_path,
) -> None:
    blocks = [
        _make_block(300, 1_000, "0xaaa"),
        _make_block(301, 1_012, "0xbbb"),
        _make_block(302, 1_024, "0xbbb"),
    ]
    outputs_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        research_agent,
        "_fetch_recent_eth_blocks",
        lambda n_blocks: blocks,
    )
    monkeypatch.setattr(research_agent, "OUTPUTS_DIR", str(outputs_dir))
    monkeypatch.setattr(
        research_agent,
        "get_openai_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("OPENAI_API_KEY should not be read")),
    )

    content = research_agent._run_analyze_eth_chain_command(None, 5, use_llm=False)

    assert "# Ethereum Dry-Run Report" in content
    assert "## Observed Metrics" in content
    assert "## Proxy Metrics" in content
    assert "## Unavailable Placeholder Metrics" in content
    assert "Requested 5 blocks, but only 3 canonical blocks were returned." in content
    assert "proposer / fee-recipient concentration proxy is high" in content
    assert "hashrate highly concentrated in top miner" not in content

    latest_index_path = outputs_dir / "latest_eth_run.json"
    assert latest_index_path.exists()

    latest_index = json.loads(latest_index_path.read_text(encoding="utf-8"))
    assert latest_index["chain"] == "ethereum"
    assert latest_index["n_blocks"] == 5
    assert latest_index["start_block"] == 300
    assert latest_index["end_block"] == 302
    assert latest_index["detector_label"] in {"normal_like", "anomalous"}
    assert isinstance(latest_index["anomaly_score"], float)

    for artifact_path in latest_index["artifacts"].values():
        assert artifact_path
        assert (tmp_path / artifact_path).exists() or pytest.fail(
            f"Missing artifact path: {artifact_path}"
        )

    report_path = latest_index["artifacts"]["report"]
    report_text = (tmp_path / report_path).read_text(encoding="utf-8")
    assert "## Recommended Next Telemetry Sources" in report_text


def test_eth_llm_prompt_uses_post_merge_wording() -> None:
    blocks = [
        _make_block(700, 1_000, "0xaaa"),
        _make_block(701, 1_012, "0xaaa"),
        _make_block(702, 1_024, "0xbbb"),
    ]
    metrics_report = build_ethereum_metrics_report(blocks)
    detector_result = {
        "detector": "simple_rule_based",
        "detector_label": "anomalous",
        "anomaly_score": 0.2,
        "result": {
            "anomaly_score": 0.2,
            "flags": ["proposer / fee-recipient concentration proxy is high"],
            "summary": "proposer / fee-recipient concentration proxy is high",
        },
    }

    prompt = research_agent._build_eth_chain_analysis_prompt(
        metrics_report,
        detector_result,
    )

    assert "proposer / fee-recipient concentration proxies" in prompt
    assert "entropy over proposer / fee-recipient addresses" in prompt
    assert "Do not call fee recipients miners" in prompt


def test_collect_eth_metrics_generates_rolling_windows_with_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    blocks = [
        _make_block(400 + index, 1_000 + index * 12, f"0x{index % 3:03x}")
        for index in range(10)
    ]
    output_csv_path = tmp_path / "eth_metrics.csv"

    monkeypatch.setattr(
        research_agent,
        "_fetch_recent_eth_blocks",
        lambda n_blocks: blocks,
    )

    result = collect_eth_metrics_to_csv(
        10,
        str(output_csv_path),
        window_size=4,
        step_size=2,
    )

    assert result["rows_written"] == 4
    assert result["window_size"] == 4
    assert result["step_size"] == 2

    with output_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames is not None
        for field_name in FEATURE_NAMES:
            assert field_name in reader.fieldnames
        for field_name in (
            "chain",
            "start_block",
            "end_block",
            "start_timestamp",
            "end_timestamp",
            "source",
            "metric_notes",
        ):
            assert field_name in reader.fieldnames

        rows = list(reader)
        assert len(rows) == 4
        assert rows[0]["chain"] == "ethereum"
        assert rows[0]["source"] == "ethereum_json_rpc"
        assert rows[0]["metric_notes"]


def test_collect_eth_metrics_refuses_to_overwrite_existing_file(
    monkeypatch,
    tmp_path,
) -> None:
    blocks = [
        _make_block(500, 1_000, "0xaaa"),
        _make_block(501, 1_012, "0xbbb"),
    ]
    output_csv_path = tmp_path / "eth_metrics.csv"
    output_csv_path.write_text("existing\n", encoding="utf-8")

    monkeypatch.setattr(
        research_agent,
        "_fetch_recent_eth_blocks",
        lambda n_blocks: blocks,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_eth_metrics_to_csv(2, str(output_csv_path))


def test_rpc_client_retries_http_503_and_succeeds() -> None:
    session = _FakeRpcSession(
        responses=[
            _FakeRpcResponse(503, {"error": {"code": -32000, "message": "busy"}}),
            _FakeRpcResponse(200, {"jsonrpc": "2.0", "id": 1, "result": "0x10"}),
        ]
    )
    client = JsonRpcClient(
        "https://example.invalid",
        timeout_sec=1.0,
        max_retries=2,
        retry_backoff_sec=0.0,
        session=session,
    )

    result = client.call("eth_blockNumber")

    assert result == "0x10"
    assert session.calls == 2


def test_rpc_client_retries_timeout_and_raises_clear_error() -> None:
    session = _FakeRpcSession(exception=requests.Timeout("timed out"))
    client = JsonRpcClient(
        "https://example.invalid",
        timeout_sec=1.5,
        max_retries=2,
        retry_backoff_sec=0.0,
        session=session,
    )

    with pytest.raises(RpcTimeoutError, match="after 3 attempt"):
        client.call("eth_blockNumber")

    assert session.calls == 3


def test_small_n_blocks_is_rejected_gracefully(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(research_agent, "OUTPUTS_DIR", str(tmp_path / "outputs"))

    with pytest.raises(ValueError, match="at least 2"):
        research_agent._run_analyze_eth_chain_command(None, 1, use_llm=False)

    with pytest.raises(ValueError, match="at least 2"):
        collect_eth_metrics_to_csv(1, str(tmp_path / "too_small.csv"))
