import sys

from aiagent.agents.chat_agent import run_cli_chat
from aiagent.agents.research_agent import run_cli_research_chat


def main() -> None:
    agent = "simple"
    if len(sys.argv) > 1 and sys.argv[1] in {"simple", "research"}:
        agent = sys.argv[1]

    if agent == "research":
        run_cli_research_chat()
    else:
        run_cli_chat()
