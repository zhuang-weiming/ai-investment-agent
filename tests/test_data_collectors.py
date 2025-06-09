import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.stock_data_collector import StockDataCollector
from src.data.akshare_collector import AKShareCollector
from src.data.eastmoney_collector import EastmoneyCollector

@pytest.fixture
def stock_data_collector():
    return StockDataCollector()

@pytest.fixture
def akshare_collector():
    return AKShareCollector()

@pytest.fixture
def eastmoney_collector():
    return EastmoneyCollector()



def test_url_generation_for_different_markets(stock_data_collector):
    """Test URL generation for different market symbols"""
    # Test China A-share (Shenzhen)
    sz_urls = stock_data_collector._get_urls_for_symbol('000333.SZ')
    assert 'quote' in sz_urls
    assert 'stats' in sz_urls
    assert 'financials' in sz_urls
    assert 'sz' in sz_urls['quote'].lower()
    
    # Test China A-share (Shanghai)
    sh_urls = stock_data_collector._get_urls_for_symbol('600519.SH')
    assert 'quote' in sh_urls
    assert 'stats' in sh_urls
    assert 'financials' in sh_urls
    
    # Test Hong Kong
    hk_urls = stock_data_collector._get_urls_for_symbol('0700.HK')
    assert 'quote' in hk_urls
    assert 'stats' in hk_urls
    assert 'financials' in hk_urls
    assert 'hk' in hk_urls['quote'].lower()
    
    # Test US market
    us_urls = stock_data_collector._get_urls_for_symbol('AAPL')
    assert 'quote' in us_urls
    assert 'stats' in us_urls
    assert 'financials' in us_urls
    assert 'finance.example.com' in us_urls['quote'].lower()

def test_akshare_collector_rate_limiting():
    """Test that AKShare collector implements rate limiting"""
    collector = AKShareCollector(rate_limit_delay=0.2)
    
    # Mock the API call function
    with patch('src.data.akshare_collector.ak.stock_zh_a_hist') as mock_api:
        mock_api.return_value = pd.DataFrame({
            '日期': pd.date_range(start='2024-01-01', periods=10),
            '开盘': np.random.rand(10) * 100,
            '收盘': np.random.rand(10) * 100,
            '最高': np.random.rand(10) * 100,
            '最低': np.random.rand(10) * 100,
            '成交量': np.random.rand(10) * 1000000,
        })
        
        # Call the API twice
        start_time = pd.Timestamp.now()
        collector._retry_api_call(mock_api, '000333')
        collector._retry_api_call(mock_api, '000333')
        end_time = pd.Timestamp.now()
        
        # Check that the second call was delayed
        elapsed = (end_time - start_time).total_seconds()
        assert elapsed >= 0.2, f"Rate limiting not enforced, elapsed time: {elapsed}s"

def test_akshare_collector_data_quality_check(akshare_collector):
    """Test data quality check functionality"""
    # Good data
    good_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })
    assert akshare_collector._check_data_quality(good_df, ['col1', 'col2'])
    
    # Missing columns
    bad_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col4': [7, 8, 9]
    })
    assert not akshare_collector._check_data_quality(bad_df, ['col1', 'col2'])
    
    # Empty dataframe
    empty_df = pd.DataFrame()
    assert not akshare_collector._check_data_quality(empty_df, ['col1'])

def test_normalize_chinese_cols(akshare_collector):
    """Test normalization of Chinese column names"""
    # Create a DataFrame with Chinese column names
    df = pd.DataFrame({
        '日期': ['2024-01-01', '2024-01-02'],
        '收盘': [100, 101],
        '开盘': [99, 100],
        '最高': [102, 103],
        '最低': [98, 99],
        '成交量': [1000, 1100]
    })
    
    # Normalize column names
    target_cols = ['日期', '收盘', '开盘', '最高', '最低', '成交量']
    normalized_df = akshare_collector._normalize_chinese_cols(df, target_cols)
    
    # Check that all columns are present
    for col in target_cols:
        assert col in normalized_df.columns

def test_retry_api_call(akshare_collector):
    """Test retry mechanism for API calls"""
    # Mock function that fails twice then succeeds
    mock_func = MagicMock(side_effect=[Exception("API Error"), Exception("API Error"), pd.DataFrame({'data': [1, 2, 3]})])
    
    # Call with retry
    result = akshare_collector._retry_api_call(mock_func, '000333', max_retries=3)
    
    # Check that the function was called 3 times
    assert mock_func.call_count == 3
    # Check that we got a result
    assert result is not None
    
    # Mock function that always fails
    always_fail = MagicMock(side_effect=Exception("API Error"))
    
    # Call with retry
    result = akshare_collector._retry_api_call(always_fail, '000333', max_retries=2)
    
    # Check that the function was called 2 times
    assert always_fail.call_count == 2
    # Check that we got None
    assert result is None

def test_safe_divide(akshare_collector):
    """Test safe division function"""
    assert akshare_collector._safe_divide(10, 2) == 5.0
    assert akshare_collector._safe_divide(10, 0) == 0
    assert akshare_collector._safe_divide(np.nan, 2) == 0
    assert akshare_collector._safe_divide(10, np.nan) == 0