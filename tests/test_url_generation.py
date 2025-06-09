"""Test URL generation for different stock symbols"""
import pytest
from src.data.stock_data_collector import StockDataCollector
from tests.conftest import TEST_STOCK_SYMBOLS

def test_china_stock_url_generation():
    """Test case 1: Test URL generation for Chinese stocks"""
    collector = StockDataCollector()
    symbol = TEST_STOCK_SYMBOLS['china']
    
    # Get URLs for the symbol
    url_dict = collector._get_urls_for_symbol(symbol)
    
    # Verify all expected URL types are generated
    expected_url_types = {"quote", "stats", "financials", "analysis", "holders"}
    assert set(url_dict.keys()) == expected_url_types, "Missing some URL types"
    
    # Extract code and market from symbol
    code = symbol.split('.')[0]
    market = symbol.split('.')[1].lower()
    
    # Check each URL
    for url_type, url in url_dict.items():
        # Verify code and market are correctly formatted in URL
        assert code in url, f"Code {code} not found in URL {url}"
        assert market in url.lower(), f"Market {market} not found in URL {url}"
        
        # Verify URL uses EastMoney domain
        assert "eastmoney.com" in url, f"URL {url} doesn't use EastMoney domain"
    
    # Verify specific URL formats
    assert "quote.eastmoney.com" in url_dict.get("quote"), "Quote URL not using correct domain"
    assert "emweb.securities.eastmoney.com" in url_dict.get("stats"), "Stats URL not using correct domain"

def test_url_generation_symbol_truncation():
    """Test symbol truncation for URLs"""
    collector = StockDataCollector()
    long_symbol = "000333.SZ.EXTRA"  # Symbol with extra characters
    
    url_dict = collector._get_urls_for_symbol(long_symbol)
    
    # Verify symbol is truncated correctly
    for url_type, url in url_dict.items():
        # Check that the code is in the URL
        assert "000333" in url, f"URL missing code: {url_type}={url}"
        # Check that the market is in the URL
        assert "sz" in url.lower(), f"URL missing market: {url_type}={url}"
        # Check that the extra part is not in the URL
        assert ".EXTRA" not in url, f"URL contains extra symbol parts: {url_type}={url}"
        # Check that we're using EastMoney domain
        assert "eastmoney.com" in url, f"URL not using EastMoney domain: {url_type}={url}"
