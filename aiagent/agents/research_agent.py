from typing import List
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from aiagent.blockchain.mock_chain import get_mock_chain_metrics
from aiagent.detectors.simple_anomaly import simple_consensus_anomaly_detector


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
    print("Type 'analyze mock chain' to run the mock metrics + anomaly detection pipeline.\n")

    history: List[str] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("AI: Bye~")
            break

        history.append(f"User: {user_input}")

        # Special command: run mock metrics + simple anomaly detector, then ask LLM to explain.
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
