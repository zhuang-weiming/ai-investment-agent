import pytest
import pandas as pd
import numpy as np
from src.data.akshare_collector import AKShareCollector

@pytest.fixture
def collector():
    return AKShareCollector()

@pytest.fixture
def sample_financial_data():
    data = {
        '市盈率': [15.5],
        '市净率': [2.1],
        '营业收入同比增长率': [12.3],
        '基本每股收益同比增长率': [8.7],
        '净资产收益率': [16.8],
        '销售毛利率': [35.2],
        '资产负债率': [45.6],
        '净利润': [1000000],
        '折旧和摊销': [200000],
        '资本支出': [300000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_current_quote():
    data = {
        '最新价': 25.6,
        '成交量': 1000000,
        '最高': 26.1,
        '最低': 25.2,
        '开盘': 25.4,
        '昨收': 25.3,
        '涨跌幅': 1.2,
        '总市值': 10000000000
    }
    return pd.Series(data)

@pytest.fixture
def sample_hist_data():
    dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
    data = {
        '收盘': np.random.uniform(20, 30, 250),
        '成交量': np.random.uniform(500000, 1500000, 250),
        '最高': np.random.uniform(21, 31, 250),
        '最低': np.random.uniform(19, 29, 250)
    }
    return pd.DataFrame(data, index=dates)

def test_calculate_metrics(collector, sample_financial_data, sample_current_quote, sample_hist_data):
    metrics = collector._calculate_metrics("000333.SZ", sample_hist_data, sample_financial_data, sample_current_quote)
    
    # Test basic metrics
    assert 'pe_ratio' in metrics
    assert metrics['pe_ratio'] == pytest.approx(15.5)
    assert metrics['pb_ratio'] == pytest.approx(2.1)
    assert metrics['revenue_growth'] == pytest.approx(12.3)
    assert metrics['eps_growth'] == pytest.approx(8.7)
    assert metrics['roe'] == pytest.approx(16.8)
    assert metrics['profit_margin'] == pytest.approx(35.2)
    assert metrics['debt_ratio'] == pytest.approx(45.6)
    
    # Test technical metrics
    assert 'sma_50' in metrics
    assert 'sma_200' in metrics
    assert 'rsi_14' in metrics
    assert 'volume_ma' in metrics

def test_get_field_value(collector, sample_financial_data):
    # Test with exact match
    value = collector._get_field_value(sample_financial_data, 'pe_ratio')
    assert value == pytest.approx(15.5)
    
    # Test with field mapping
    value = collector._get_field_value(sample_financial_data, '市盈率')
    assert value == pytest.approx(15.5)
    
    # Test with missing field
    value = collector._get_field_value(sample_financial_data, 'non_existent_field', default=0.0)
    assert value == 0.0

def test_calculate_owner_earnings(collector, sample_financial_data):
    owner_earnings = collector._calculate_owner_earnings(sample_financial_data)
    expected = 1000000 + 200000 - 300000  # net_income + depreciation - capex
    assert owner_earnings == pytest.approx(expected)

def test_calculate_rsi(collector):
    # Test RSI calculation with sample price data
    prices = np.array([10, 12, 11, 13, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17, 19])
    rsi = collector._calculate_rsi(prices)
    assert 0 <= rsi <= 100  # RSI should always be between 0 and 100

def test_safe_divide(collector):
    assert collector._safe_divide(10, 2) == pytest.approx(5.0)
    assert collector._safe_divide(10, 0) == 0
    assert collector._safe_divide(np.nan, 2) == 0
    assert collector._safe_divide(10, np.nan) == 0
