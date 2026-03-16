import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from aiagent.blockchain.mock_chain import get_mock_chain_metrics
from aiagent.detectors.csv_loader import load_metrics_samples_from_csv
from aiagent.detectors.evaluation import evaluate_detection_results
from aiagent.detectors.isolation_forest_detector import (
    fit_isolation_forest,
    score_metrics_with_isolation_forest,
    score_samples_with_isolation_forest,
)
from aiagent.detectors.simple_anomaly import simple_consensus_anomaly_detector
from aiagent.detectors.training import (
    fit_statistical_baseline,
    generate_mock_training_samples,
    get_training_recommendations,
    score_with_statistical_baseline,
)


def run_cli_research_chat() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini")

    base_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI research assistant specialized in blockchain security, "
                "consensus attacks, and anomaly detection. You help the user design "
                "experiments, analyze metrics, and reason about defenses in a clear "
                "and rigorous way.",
            ),
            ("human", "{input}"),
        ]
    )

    print("=== AIagent Research CLI Chat (LangChain) ===")
    print("Type 'exit' or 'quit' to leave the chat.")
    print("Type 'analyze mock chain' to run the mock metrics + anomaly detection pipeline.")
    print("Type 'train mock detector' to fit a statistical baseline on mock samples and score the current mock chain.")
    print("Type 'analyze csv detector' to train an Isolation Forest on data/mock_metrics.csv and explain the results.")
    print("You can also use: 'analyze csv detector <path-to-csv>'.\n")

    history: List[str] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("AI: Bye~")
            break

        history.append(f"User: {user_input}")

        if user_input.lower().startswith("analyze mock chain"):
            try:
                metrics = get_mock_chain_metrics.invoke({"chain": "mockchain"})
                detection = simple_consensus_anomaly_detector.invoke({"metrics": metrics})
            except Exception as e:
                print(f"[Error] Tool execution failed: {e}")
                continue

            metrics_json = json.dumps(metrics, indent=2)
            detection_json = json.dumps(detection, indent=2)

            analysis_input = (
                "We just executed a mock blockchain monitoring pipeline that "
                "(1) fetched consensus-level metrics and (2) applied a simple "
                "rule-based anomaly detector.\n\n"
                f"Raw metrics (JSON):\n{metrics_json}\n\n"
                f"Detector output (JSON):\n{detection_json}\n\n"
                "Please do the following:\n"
                "1. Summarize what the metrics and detector output suggest about the chain's security state.\n"
                "2. Hypothesize what type(s) of consensus attack, if any, these indicators might be consistent with.\n"
                "3. Suggest at least three follow-up metrics or checks you would add in a real system.\n"
            )

            try:
                response = llm.invoke(analysis_input)
            except Exception as e:
                print(f"[Error] LangChain/OpenAI call failed: {e}")
                continue

        elif user_input.lower().startswith("train mock detector"):
            try:
                samples = generate_mock_training_samples(sample_count=120)
                baseline = fit_statistical_baseline(samples)
                metrics = get_mock_chain_metrics.invoke({"chain": "mockchain"})
                baseline_detection = score_with_statistical_baseline(metrics, baseline)
                recommendations = get_training_recommendations()
            except Exception as e:
                print(f"[Error] Training pipeline failed: {e}")
                continue

            baseline_json = json.dumps(baseline, indent=2)
            metrics_json = json.dumps(metrics, indent=2)
            baseline_detection_json = json.dumps(baseline_detection, indent=2)
            recommendations_json = json.dumps(recommendations, indent=2)

            analysis_input = (
                "We just executed a mock detector training and scoring pipeline.\n\n"
                "Pipeline steps:\n"
                "1. Generate labeled mock training samples with both normal and abnormal windows.\n"
                "2. Fit a statistical baseline using only normal samples.\n"
                "3. Score the current mock chain metrics against the fitted baseline.\n"
                "4. Review training recommendations for a real detector.\n\n"
                f"Fitted baseline (JSON):\n{baseline_json}\n\n"
                f"Current metrics (JSON):\n{metrics_json}\n\n"
                f"Baseline scoring result (JSON):\n{baseline_detection_json}\n\n"
                f"Training recommendations (JSON):\n{recommendations_json}\n\n"
                "Please do the following:\n"
                "1. Explain what the fitted baseline and scoring result tell us.\n"
                "2. Explain why a statistical baseline is useful as an initial detector in blockchain security monitoring.\n"
                "3. Summarize the most important next steps for building a stronger ML detector.\n"
            )

            try:
                response = llm.invoke(analysis_input)
            except Exception as e:
                print(f"[Error] LangChain/OpenAI call failed: {e}")
                continue

        elif user_input.lower().startswith("analyze csv detector"):
            try:
                command_parts = user_input.split(maxsplit=3)
                csv_path = command_parts[3] if len(command_parts) == 4 else "data/mock_metrics.csv"
                samples = load_metrics_samples_from_csv(csv_path)
                model = fit_isolation_forest(samples, contamination=0.3)
                batch_results = score_samples_with_isolation_forest(samples, model)
                evaluation = evaluate_detection_results(batch_results)
                current_metrics = get_mock_chain_metrics.invoke({"chain": "mockchain"})
                current_result = score_metrics_with_isolation_forest(current_metrics, model)
            except Exception as e:
                print(f"[Error] CSV detector pipeline failed: {e}")
                continue

            batch_results_json = json.dumps(batch_results[:5], indent=2)
            evaluation_json = json.dumps(evaluation, indent=2)
            current_metrics_json = json.dumps(current_metrics, indent=2)
            current_result_json = json.dumps(current_result, indent=2)

            analysis_input = (
                "We just executed a CSV-based anomaly detection workflow using Isolation Forest.\n\n"
                "Pipeline steps:\n"
                f"1. Load labeled consensus-metric windows from {csv_path}.\n"
                "2. Train an Isolation Forest model, prioritizing normal samples when available.\n"
                "3. Score the CSV samples and also score the current mock chain metrics.\n"
                "4. Compute basic evaluation statistics over the CSV predictions.\n\n"
                f"Sample scoring preview (JSON):\n{batch_results_json}\n\n"
                f"Evaluation summary (JSON):\n{evaluation_json}\n\n"
                f"Current mock chain metrics (JSON):\n{current_metrics_json}\n\n"
                f"Current mock chain Isolation Forest result (JSON):\n{current_result_json}\n\n"
                "Please do the following:\n"
                "1. Explain what the CSV-trained detector appears to learn from the samples.\n"
                "2. Interpret the evaluation summary, especially the per-label detection rates.\n"
                "3. Interpret the current mock chain result in blockchain-security terms.\n"
                "4. Explain the limitations of this small dataset and how to improve it for real research use.\n"
            )

            try:
                response = llm.invoke(analysis_input)
            except Exception as e:
                print(f"[Error] LangChain/OpenAI call failed: {e}")
                continue

        else:
            chain = base_prompt | llm
            try:
                response = chain.invoke({"input": user_input})
            except Exception as e:
                print(f"[Error] LangChain/OpenAI call failed: {e}")
                continue

        content = response.content
        print(f"AI: {content}\n")

        history.append(f"AI: {content}")
