# AIagent

AIagent is a command-line AI agent prototype for blockchain consensus-security monitoring, anomaly detection, and LLM-assisted interpretation. It combines terminal interaction, detector experiments, and a real Ethereum JSON-RPC telemetry loop so the project can be cloned, tested, and demoed as a capstone research prototype.

## Features

- Terminal chat mode
- Research agent mode
- Mock chain analysis
- Mock detector training
- CSV detector workflow
- Ethereum real-chain JSON-RPC telemetry
- Dry-run mode without LLM output
- Isolation Forest detector
- Artifact saving under `outputs/`
- Rolling-window CSV collection

## Project Structure

- `aiagent/agents`
  - CLI chat and research workflows
- `aiagent/blockchain`
  - Mock-chain telemetry, Ethereum JSON-RPC client, metrics builders, and schemas
- `aiagent/detectors`
  - Rule-based detection, CSV loading, Isolation Forest, evaluation, and training helpers
- `data/`
  - Sample detector data such as `data/mock_metrics.csv`
- `tests/`
  - Regression and Ethereum pipeline tests
- `outputs/`
  - Generated at runtime for Ethereum analysis artifacts

## Installation

From the project root:

```powershell
python -m venv agent
agent\Scripts\activate
pip install -r requirements.txt
```

Direct command style with the local virtual environment:

```powershell
agent\Scripts\python.exe -m pytest
```

## Environment Variables

The project uses:

- `OPENAI_API_KEY`
- `ETH_RPC_URL`

`ETH_RPC_URL` should be a full Ethereum JSON-RPC endpoint, for example:

```text
https://eth-mainnet.g.alchemy.com/v2/<YOUR_ALCHEMY_KEY>
```

Notes:

- Do not commit real keys.
- For dry-run mode, `OPENAI_API_KEY` is not required, but `ETH_RPC_URL` is required.
- For LLM interpretation, `OPENAI_API_KEY` is required.

## Usage

Start the default CLI:

```powershell
python main.py
```

Start research mode:

```powershell
python main.py research
```

Research CLI commands:

- `analyze mock chain`
- `train mock detector`
- `analyze csv detector`
- `analyze csv detector data/mock_metrics.csv`
- `analyze eth chain dryrun 100`
- `analyze eth chain --no-llm 100`
- `collect eth metrics 500 data/eth_metrics_latest.csv`
- `analyze eth detector`
- `analyze real chain`
- `analyze eth chain 100`

## Recommended Capstone Demo Workflow

Use one normal interactive PowerShell terminal so the environment variables stay in the same session:

```powershell
cd E:\capstone\AIagent

$env:OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
$env:ETH_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/<YOUR_ALCHEMY_KEY>"

agent\Scripts\python.exe main.py research
```

Inside the CLI:

```text
analyze eth chain dryrun 100
collect eth metrics 500 data/eth_metrics_latest.csv
analyze eth detector
```

Optional:

```text
analyze eth chain 100
```

## Expected Artifacts

Successful Ethereum runs can generate:

- `outputs/latest_eth_run.json`
- `outputs/eth_blocks_*.json`
- `outputs/eth_metrics_*.json`
- `outputs/eth_detection_*.json`
- `outputs/eth_report_*.md`
- `data/eth_metrics_latest.csv`

## Metric Interpretation Caveats

The detector schema remains stable across mock and Ethereum workflows:

- `block_time_sec_avg`
- `block_time_sec_std`
- `fork_rate`
- `orphan_rate`
- `reorg_depth_max`
- `hashrate_concentration_top1`
- `hashrate_concentration_top3`
- `miner_entropy`

For Ethereum post-Merge, these schema names stay unchanged for compatibility, but the user-facing interpretation is different:

- Ethereum post-Merge does not expose PoW hashrate via canonical JSON-RPC.
- `hashrate_concentration_top1` and `hashrate_concentration_top3` are proposer / fee-recipient concentration proxies for Ethereum, not true PoW hashrate measures.
- `miner_entropy` is entropy over proposer / fee-recipient addresses for Ethereum.
- `fork_rate`, `orphan_rate`, and `reorg_depth_max` are placeholders under canonical JSON-RPC.
- Stronger conclusions require beacon-chain validator data, p2p propagation data, mempool telemetry, MEV relay or builder data, archive-node history, or node logs.

## Testing

Syntax pass:

```powershell
agent\Scripts\python.exe -m compileall aiagent
```

Test suite:

```powershell
agent\Scripts\python.exe -m pytest
```

## Known Limitations And Next Steps

- The real-chain workflow currently uses canonical RPC-only telemetry.
- `fork_rate`, `orphan_rate`, and `reorg_depth_max` are placeholders under this data model.
- The mock CSV fallback for the ML detector is useful for demo purposes but is not a valid Ethereum baseline.
- Future work should collect a longer real Ethereum baseline, integrate beacon-chain data, add p2p and mempool telemetry, and use better temporal models.
