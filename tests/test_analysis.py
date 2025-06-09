"""Test analysis strategies"""
import pytest
from src.agents.peter_lynch_agent import PeterLynchAgent
from src.agents.warren_buffett_agent import warren_buffett_agent
from src.agents.technical_agent import technical_analyst_agent

class DummyState(dict):
    pass

def test_peter_lynch_agent_basic():
    agent = PeterLynchAgent()
    result = agent.analyze('600519')
    assert "signal" in result
    assert "confidence" in result
    assert "reasoning" in result
    assert result["signal"] in ["bullish", "bearish", "neutral"]
    assert 0 <= result["confidence"] <= 100

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

def test_technical_analyst_agent_basic():
    state = DummyState({
        "data": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "tickers": ["600519"],
            "analyst_signals": {},
        },
        "messages": [],
    })
    result = technical_analyst_agent(state)
    assert "messages" in result
    assert "data" in result
    assert "technical_analyst_agent" in result["data"]["analyst_signals"]
    ticker_result = result["data"]["analyst_signals"]["technical_analyst_agent"]["600519"]
    assert ticker_result["signal"] in ["bullish", "bearish", "neutral"]
    assert 0 <= ticker_result["confidence"] <= 100
