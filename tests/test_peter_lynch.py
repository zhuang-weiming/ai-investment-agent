import pytest
from src.agents.peter_lynch_agent import PeterLynchAgent

def test_peter_lynch_agent_basic():
    agent = PeterLynchAgent()
    result = agent.analyze('600519')
    assert "signal" in result
    assert "confidence" in result
    assert "reasoning" in result
    assert result["signal"] in ["bullish", "bearish", "neutral"]
    assert 0 <= result["confidence"] <= 100
