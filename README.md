# AIagent

Command-line AI agent project skeleton for research on blockchain security, AI agents, and anomaly detection.

## Install dependencies

In the project root directory (`e:\\capstone\\AIagent`), run:

```bash
pip install -r requirements.txt
```

Make sure the following environment variable is set:

- `OPENAI_API_KEY`

## Run the command-line chat

In the project root directory, run:

```bash
python main.py
python main.py research
```

Then you can chat with the model directly in the terminal. Type `exit` or `quit` to end the session.

## Research agent demo commands

When running `python main.py research`, you can use the following demo commands:

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

- Replace mock metrics with real blockchain telemetry
- Add proper dataset storage and versioning
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
