import pytest
from src.agents.warren_buffett_agent import warren_buffett_agent

class DummyState(dict):
    pass

def test_warren_buffett_agent_basic():
    state = DummyState({
        "data": {
            "tickers": ["600519"],
            "analyst_signals": {},
        },
        "messages": [],
    })
    result = warren_buffett_agent(state)
    assert "messages" in result
    assert "data" in result
    assert "warren_buffett_agent" in result["data"]["analyst_signals"]
    ticker_result = result["data"]["analyst_signals"]["warren_buffett_agent"]["600519"]
    assert ticker_result["signal"] in ["bullish", "bearish", "neutral"]
    assert 0 <= ticker_result["confidence"] <= 100
    assert isinstance(ticker_result["reasoning"], str)
