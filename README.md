# AIagent

Command-line AI agent project skeleton for research on blockchain security, AI agents, and anomaly detection.

## Install dependencies

In the project root directory (`e:\\capstone\\AIagent`), run:

```bash
pip install -r requirements.txt
```

Make sure the following environment variable is set:

- `ETH_RPC_URL`
- `OPENAI_API_KEY`

Notes:

- `OPENAI_API_KEY` is required only for LLM-backed analysis and explanation commands.
- `analyze eth chain dryrun` and `analyze eth chain --no-llm` do not require `OPENAI_API_KEY`.
- `ETH_RPC_URL` is required for real-chain Ethereum telemetry commands.

## Run the command-line chat

In the project root directory, run:

```bash
python main.py
python main.py research
```

Then you can chat with the model directly in the terminal. Type `exit` or `quit` to end the session.

## Research agent demo commands

When running `python main.py research`, you can use the following commands.

### Existing mock and CSV demo commands

- `analyze mock chain`
  - Fetches mock consensus metrics from `aiagent.blockchain.mock_chain`
  - Runs the rule-based detector in `aiagent.detectors.simple_anomaly`
  - Asks the LLM to explain the security meaning of the results

- `train mock detector`
  - Generates labeled mock training samples
  - Fits a simple statistical baseline using normal samples
  - Scores the current mock chain metrics against that baseline
  - Asks the LLM to explain the detector behavior and training implications

- `analyze csv detector`
  - Loads labeled samples from `data/mock_metrics.csv`
  - Trains an `IsolationForest` detector on the CSV data
  - Scores both the CSV samples and the current mock chain metrics
  - Computes basic evaluation statistics over the CSV predictions
  - Asks the LLM to explain what the detector likely learned and where the dataset is limited

- `analyze csv detector <path-to-csv>`
  - Runs the same workflow on a custom CSV file path
  - Example:
    - `analyze csv detector data/mock_metrics.csv`

### New real-chain Ethereum commands

- `analyze eth chain`
  - Fetches the latest 100 Ethereum canonical blocks from `ETH_RPC_URL`
  - Builds detector-compatible metrics from the recent block window
  - Runs the rule-based detector
  - Saves raw blocks, metrics, detector output, and an LLM report under `outputs/`

- `analyze eth chain <n_blocks>`
  - Same as above, but uses a custom block window size
  - Example:
    - `analyze eth chain 300`

- `analyze eth chain dryrun`
  - Fetches recent Ethereum blocks from `ETH_RPC_URL`
  - Builds the same real-chain metrics and runs the same detector
  - Saves raw blocks, metrics, detector output, and a deterministic local Markdown report
  - Does not call the LLM and does not require `OPENAI_API_KEY`

- `analyze eth chain --no-llm <n_blocks>`
  - Same dry-run behavior with an explicit flag
  - Example:
    - `analyze eth chain --no-llm 100`

- `analyze eth detector`
  - Fetches recent Ethereum blocks
  - Builds the current metric window
  - Loads a baseline CSV if available
  - Trains an `IsolationForest` model and scores the current Ethereum window
  - Warns clearly if it falls back to `data/mock_metrics.csv`
  - Saves artifacts under `outputs/`

- `collect eth metrics <n_blocks> <output_csv_path>`
  - Fetches recent Ethereum blocks
  - Builds one or more rolling-window metric samples
  - Saves detector-compatible CSV rows plus metadata columns
  - Example:
    - `collect eth metrics 500 data/eth_metrics_latest.csv`

- `analyze real chain`
  - Alias for `analyze eth chain`

## Real-chain telemetry semantics

The real-chain pipeline keeps the detector feature schema stable:

- `block_time_sec_avg`
- `block_time_sec_std`
- `fork_rate`
- `orphan_rate`
- `reorg_depth_max`
- `hashrate_concentration_top1`
- `hashrate_concentration_top3`
- `miner_entropy`

For Ethereum canonical JSON-RPC telemetry, these fields do not all mean the same thing as they would on a PoW chain:

- Directly observed:
  - `block_time_sec_avg`
  - `block_time_sec_std`

- Proxy metrics:
  - `hashrate_concentration_top1`
  - `hashrate_concentration_top3`
  - `miner_entropy`
  - These are computed from fee recipient / proposer address frequency, not true PoW hashrate concentration.

- Placeholder metrics:
  - `fork_rate`
  - `orphan_rate`
  - `reorg_depth_max`
  - Canonical block queries do not expose these reliably, so the code stores explicit metadata that they are not observable from a simple canonical RPC window.

The LLM prompts are instructed to distinguish observed, proxy, and unavailable metrics and to recommend the next telemetry needed before making stronger security claims.

## Artifacts

Real-chain analysis runs create `outputs/` artifacts such as:

- `outputs/eth_blocks_YYYYMMDD_HHMMSS.json`
- `outputs/eth_metrics_YYYYMMDD_HHMMSS.json`
- `outputs/eth_detection_YYYYMMDD_HHMMSS.json`
- `outputs/eth_report_YYYYMMDD_HHMMSS.md`
- `outputs/latest_eth_run.json`

`outputs/latest_eth_run.json` is an index for the most recent Ethereum analysis run and includes:

- chain
- requested block count
- start and end block
- timestamp
- detector label
- anomaly score
- artifact paths

## Detector development direction

The project now includes a minimal detector research scaffold:

- `aiagent.detectors.schemas`
  - Shared typed structures for consensus metrics, labeled samples, and detection results

- `aiagent.detectors.simple_anomaly`
  - A rule-based baseline detector for quick demonstrations

- `aiagent.detectors.training`
  - Mock training sample generation
  - Statistical baseline fitting
  - Baseline-based anomaly scoring
  - Training recommendations for future ML models

- `aiagent.detectors.csv_loader`
  - Loads consensus-metric windows from CSV files into typed detector samples

- `aiagent.detectors.isolation_forest_detector`
  - Trains and scores an `IsolationForest` model for unsupervised anomaly detection

- `aiagent.detectors.evaluation`
  - Computes basic evaluation summaries such as anomalous prediction rate and per-label detection rate

Recommended next step:

- Collect longer real-chain baselines
- Add chain-specific detector training and validation
- Add richer telemetry such as beacon-chain, p2p, mempool, and reorg tracking
- Train unsupervised and temporal models on multi-window features

## CSV data format

The sample CSV file is stored at:

```text
data/mock_metrics.csv
```

Required columns:

```text
block_time_sec_avg
block_time_sec_std
fork_rate
orphan_rate
reorg_depth_max
hashrate_concentration_top1
hashrate_concentration_top3
miner_entropy
```

Optional column:

```text
label
```

If `label` is present, the current implementation will prefer samples labeled `normal`
when fitting the Isolation Forest training set.

Real-chain collection also adds optional metadata columns such as:

- `chain`
- `start_block`
- `end_block`
- `start_timestamp`
- `end_timestamp`
- `source`
- `metric_notes`

These extra columns are ignored by the current CSV loader, so detector compatibility is preserved.

## Demo workflow

1. Set environment variables:

```bash
set ETH_RPC_URL=https://your-ethereum-rpc-endpoint
set OPENAI_API_KEY=your-openai-api-key
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start research mode:

```bash
python main.py research
```

4. Run a no-LLM real-chain dry run:

```text
analyze eth chain dryrun 100
```

5. Collect a reusable Ethereum baseline CSV:

```text
collect eth metrics 500 data/eth_metrics_latest.csv
```

6. Run rule-based real-chain analysis with the LLM enabled:

```text
analyze eth chain 100
```

7. Run Isolation Forest scoring on real-chain telemetry:

```text
analyze eth detector
```

## Metric interpretation caveats

- Ethereum proposer concentration is only a proxy for hashrate concentration.
- `fork_rate`, `orphan_rate`, and `reorg_depth_max` are placeholders when the data source is canonical Ethereum JSON-RPC alone.
- Stronger security conclusions require richer telemetry such as beacon-chain data, p2p propagation measurements, mempool data, relay / builder telemetry, explicit reorg tracking, or node logs.

## Run tests

From the project root:

```bash
python -m pytest
```
