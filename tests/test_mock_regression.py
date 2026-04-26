import pytest

from aiagent.agents import research_agent


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChatOpenAI:
    def __init__(self, model: str) -> None:
        self.model = model

    def invoke(self, prompt: str) -> _FakeResponse:
        assert "mock blockchain monitoring pipeline" in prompt
        return _FakeResponse("mock analysis ok")


def test_existing_mock_command_still_works(
    monkeypatch,
    capsys,
) -> None:
    inputs = iter(["analyze mock chain", "quit"])

    monkeypatch.setattr(research_agent, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(research_agent, "get_openai_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    research_agent.run_cli_research_chat()

    output = capsys.readouterr().out
    assert "mock analysis ok" in output
