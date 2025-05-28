"""Test URL generation for different stock symbols"""
import pytest
from src.models.stock_data_collector import StockDataCollector
from tests.conftest import TEST_STOCK_SYMBOLS, EXPECTED_URL_PATTERNS

@pytest.mark.asyncio
async def test_china_stock_url_generation():
    """Test case 1: Test URL generation for Chinese stocks"""
    collector = StockDataCollector()
    symbol = TEST_STOCK_SYMBOLS['china']
    
    # Get URLs for the symbol
    urls = collector._get_urls_for_symbol(symbol)
    
    # Verify all expected URLs are generated
    assert len(urls) == len(EXPECTED_URL_PATTERNS), "Missing some URLs"
    
    # Check each URL pattern
    for url in urls:
        # Verify symbol is correctly formatted in URL
        assert symbol in url, f"Symbol {symbol} not found in URL {url}"
        
        # Verify URL matches one of the expected patterns
        matches_pattern = any(
            url == pattern.format(symbol=symbol)
            for pattern in EXPECTED_URL_PATTERNS.values()
        )
        assert matches_pattern, f"URL {url} doesn't match any expected pattern"
        
    # Verify specific URL formats
    quote_url = f"https://finance.yahoo.com/quote/{symbol}"
    stats_url = f"https://finance.yahoo.com/quote/{symbol}/key-statistics"
    
    assert quote_url in urls, "Quote URL not found"
    assert stats_url in urls, "Statistics URL not found"

@pytest.mark.asyncio
async def test_url_generation_symbol_truncation():
    """Test symbol truncation for URLs"""
    collector = StockDataCollector()
    long_symbol = "000333.SZ.EXTRA"  # Symbol with extra characters
    
    urls = collector._get_urls_for_symbol(long_symbol)
    
    # Verify symbol is truncated correctly
    for url in urls:
        assert "000333.SZ" in url, f"URL contains incorrect symbol format: {url}"
        assert ".EXTRA" not in url, f"URL contains extra symbol parts: {url}"
