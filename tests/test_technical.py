import pytest
import pandas as pd
import numpy as np
from src.agents.technical_agent import technical_analyst_agent

class DummyState(dict):
    pass

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

def test_technical_agent_with_akshare_data():
    # Create sample data in AKShare format
    hist_data = pd.DataFrame({
        "日期": pd.date_range(start="2024-01-01", periods=120, freq="D").strftime("%Y-%m-%d"),
        "开盘": np.random.uniform(50, 55, 120),
        "最高": np.random.uniform(54, 58, 120),
        "最低": np.random.uniform(48, 52, 120),
        "收盘": np.random.uniform(51, 56, 120),
        "成交量": np.random.uniform(200000, 500000, 120),
        "成交额": np.random.uniform(1e7, 2e7, 120),
        "振幅": np.random.uniform(1, 3, 120),
        "涨跌幅": np.random.uniform(-2, 2, 120),
        "涨跌额": np.random.uniform(-1, 1, 120),
        "换手率": np.random.uniform(0.5, 1.5, 120)
    })

    market_data = pd.Series({
        "最新价": 53.5,
        "涨跌幅": 1.2,
        "成交量": 300000,
        "换手率": 0.8,
        "市盈率-动态": 15.5,
        "市净率": 2.1
    })

    # Create state with AKShare format data
    state = {
        "data": {
            "tickers": ["000333.SZ"],
            "hist_data": hist_data,
            "market_data": market_data,
            "analyst_signals": {}
        },
        "messages": []
    }

    # Run technical analysis
    result = technical_analyst_agent(state)
    
    # Verify structure and content
    assert "data" in result
    assert "messages" in result
    assert len(result["messages"]) > 0
    
    signals = result["data"]["analyst_signals"]["technical_analyst_agent"]
    assert "000333.SZ" in signals
    
    analysis = signals["000333.SZ"]
    assert "signal" in analysis
    assert analysis["signal"] in ["bullish", "bearish", "neutral"]
    assert "confidence" in analysis
    assert 0 <= analysis["confidence"] <= 100
    assert "reasoning" in analysis
    assert "metrics" in analysis
    
    metrics = analysis["metrics"]
    assert "rsi" in metrics
    assert "trend" in metrics
    assert "momentum" in metrics
    assert "price" in metrics
    assert "volume" in metrics
    assert "turnover" in metrics
