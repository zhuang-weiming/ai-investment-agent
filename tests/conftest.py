"""Test configuration and fixtures"""
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from dotenv import load_dotenv

load_dotenv()

# Mock HTML data that simulates a Yahoo Finance page
MOCK_PAGE_HTML = '''
<html>
<head><title>Midea Group (000333.SZ) - Yahoo Finance</title></head>
<body>
    <div id="quote-header">
        Midea Group Co., Ltd. (000333.SZ)
        Current Price: 150.25 (+2.50, +1.69%)
        Shenzhen - Shenzhen Delayed Price. Currency in CNY
    </div>
    <div id="quote-summary">
        Previous Close: 147.75
        Open: 148.20
        Day's Range: 147.50 - 151.30
        52 Week Range: 120.80 - 165.40
        Volume: 12,345,678
        Avg. Volume: 15,234,567
        Market Cap: 1.05T
        Beta (5Y Monthly): 0.92
        PE Ratio (TTM): 18.25
        EPS (TTM): 8.23
        Earnings Date: Jul 24, 2025 - Jul 29, 2025
        Forward Dividend & Yield: 5.20 (3.46%)
        Ex-Dividend Date: Apr 15, 2025
    </div>
    <div id="company-description">
        Midea Group Co., Ltd. manufactures and trades in household appliances, 
        robots, industrial automation systems, and HVAC systems worldwide.
    </div>
</body>
</html>
'''

# Sample data for tests
SAMPLE_FINANCIAL_DATA = {
    'symbol': '000333.SZ',
    'vix': 15.5,
    'timestamp': '2025-05-22T10:30:00Z',
    'current_price': 150.8,  # Added current price
    # Add all required metrics at the top level
    'pe_ratio': 18.25,
    'pb_ratio': 2.5,
    'roe': 18.0,
    'debt_ratio': 25.0,
    'fcf': 120000000.0,  # Added missing fcf metric
    'owner_earnings': 1000000000.0,  # Already here but mentioned in error
    'moat_type': 'brand',
    'mgmt_score': 1.8,
    'industry_pe': 22.0,
    'historical_pb': 3.0,
    'market_share': 12.5,  # Already here but mentioned in error
    'brand_value': 85.0,
    'capital_allocation': 0.8,
    'market_concentration': 65.0,
    'regulatory_risk': 45.0,
    'competitive_position': 2,
    'insider_ownership': 15.0,
    'buyback_efficiency': 0.75,
    'industry_growth': 8.5,
    'industry_cycle_position': 'growth',
    # Keep the original nested structure for other tests
    'fundamental_data': {
        'pe_ratio': 18.25,
        'pb_ratio': 2.5,
        'revenue_growth': 10.5,
        'eps_growth': 8.7,
        'peg_ratio': 1.23,
        'roe': 18.0,
        'debt_ratio': 25.0,
        'profit_margin': 35.2,
        'market_share': 12.5,
        'brand_value': 85.0,
        'owner_earnings': 1000000000.0,
        'moat_type': 'brand',
        'mgmt_score': 1.8,
        'industry_pe': 22.0,
        'historical_pb': 3.0,
        'capital_allocation': 0.8,
        'market_concentration': 65.0,
        'regulatory_risk': 45.0,
        'competitive_position': 2,
        'insider_ownership': 15.0,
        'buyback_efficiency': 0.75,
        'industry_growth': 8.5,
        'industry_cycle_position': 'growth',
        'fcf': 120000000.0  # Also add fcf here for completeness
    },
    'technical_data': {
        'close_prices': [148.2, 149.5, 150.25, 151.0, 150.8],
        'volumes': [1000000, 1200000, 1100000, 1300000, 1250000],
        'high_prices': [149.0, 150.0, 151.3, 151.5, 151.0],
        'low_prices': [147.5, 148.0, 149.5, 150.0, 149.8],
        'sma_50': 150.0,
        'sma_200': 145.5,
        'rsi_14': 55.0,
        'macd': 0.5,
        'macd_signal': 0.3,
        'macd_hist': 0.2,
        'volume_ma': 1170000
    },
    'market_data': {
        'last_price': 150.8,
        'volume': 1250000,
        'high': 151.0,
        'low': 149.8,
        'open': 150.0,
        'prev_close': 150.25,
        'change_pct': 0.37
    },
    'pages': [
        {
            'url': 'https://finance.yahoo.com/quote/000333.SZ',
            'title': 'Midea Group (000333.SZ)',
            'content': '''Midea Group Co., Ltd. (000333.SZ) Stock Price, Quote and News...
PEG Ratio: 1.23
Moat Type: Wide
Technical: SMA(50): 150.0, RSI: 55, Momentum: Bullish, Trend: Up
Financials: Revenue Growth: 10%, EPS Growth: 8%, Debt/Equity: 0.3, FCF Margin: 12%
Insider Buys: 2, Insider Sells: 0
News: "Midea Group expands overseas..."
Close Prices: [148.2, 149.5, 150.25, 151.0, 150.8]
Volumes: [1000000, 1200000, 1100000, 1300000, 1250000]
High Prices: [149.0, 150.0, 151.3, 151.5, 151.0]
Low Prices: [147.5, 148.0, 149.5, 150.0, 149.8]
'''
        },
        {
            'url': 'https://finance.yahoo.com/quote/000333.SZ/financials',
            'title': 'Midea Group (000333.SZ) Financial Statements',
            'content': '''Financial statements, balance sheets, cash flow...
PEG Ratio: 1.23
Moat Type: Wide
Technical: SMA(50): 150.0, RSI: 55, Momentum: Bullish, Trend: Up
Financials: Revenue Growth: 10%, EPS Growth: 8%, Debt/Equity: 0.3, FCF Margin: 12%
Insider Buys: 2, Insider Sells: 0
News: "Midea Group expands overseas..."
Close Prices: [148.2, 149.5, 150.25, 151.0, 150.8]
Volumes: [1000000, 1200000, 1100000, 1300000, 1250000]
High Prices: [149.0, 150.0, 151.3, 151.5, 151.0]
Low Prices: [147.5, 148.0, 149.5, 150.0, 149.8]
'''
        }
    ],
    # Add all required fields for technical strategy
    'close_prices': [148.2, 149.5, 150.25, 151.0, 150.8],
    'volumes': [1000000, 1200000, 1100000, 1300000, 1250000],
    'high_prices': [149.0, 150.0, 151.3, 151.5, 151.0],
    'low_prices': [147.5, 148.0, 149.5, 150.0, 149.8],
    # Add all required metrics for Peter Lynch and Warren Buffett strategies
    'revenue_growth': 10.0,
    'eps_growth': 8.0,
    'peg_ratio': 1.23,
    'debt_to_equity': 0.3,
    'fcf_margin': 0.12,
    'news_sentiment': 5.0,
    'insider_buys': 2,
    'insider_sells': 0,
    # Buffett metrics
    'owner_earnings': 1000000000.0,
    'roe': 0.18,
    'debt_ratio': 0.25,
    'fcf': 800000000.0,
    'moat_type': 'brand',
    'mgmt_score': 1.8,
    'pe_ratio': 18.25,
    'industry_pe': 22.0,
    'pb_ratio': 2.5,
    'historical_pb': 3.0,
}

# Test stock symbols
TEST_STOCK_SYMBOLS = {
    'china': '000333.SZ',  # Midea Group
    'us': 'AAPL',         # Apple Inc.
    'hk': '0700.HK'       # Tencent Holdings
}

# Expected URL patterns for different data types
# Note: We no longer use Yahoo Finance URLs, but keeping this structure for test compatibility
# These are now EastMoney URLs for China stocks
EXPECTED_URL_PATTERNS = {
    'quote': "http://quote.eastmoney.com/{market}{code}.html",
    'stats': "http://emweb.securities.eastmoney.com/PC_HSF10/OperationsRequired/Index?type=web&code={market}{code}",
    'financials': "http://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index?type=web&code={market}{code}",
    'analysis': "http://emweb.securities.eastmoney.com/PC_HSF10/ProfitForecast/Index?code={market}{code}",
    'holders': "http://emweb.securities.eastmoney.com/PC_HSF10/ShareholderResearch/Index?code={market}{code}"
}

@pytest_asyncio.fixture
async def mock_browser():
    """Mock browser fixture for testing"""
    mock = AsyncMock()
    mock.new_page = AsyncMock()
    mock.close = AsyncMock()
    return mock

@pytest_asyncio.fixture
async def mock_page():
    """Mock Playwright page object"""
    mock = AsyncMock()
    mock.goto = AsyncMock()
    mock.content = AsyncMock(return_value=MOCK_PAGE_HTML)
    mock.wait_for_selector = AsyncMock()
    return mock

@pytest_asyncio.fixture
async def mock_browser():
    """Mock Playwright browser object"""
    mock = AsyncMock()
    mock.new_page = AsyncMock()
    return mock

@pytest_asyncio.fixture
async def mock_playwright():
    """Mock Playwright instance"""
    mock = AsyncMock()
    mock.chromium.launch = AsyncMock()
    return mock

@pytest.fixture
def mock_html():
    """Mock HTML content for testing"""
    return MOCK_PAGE_HTML

@pytest.fixture
def sample_financial_data():
    """Returns sample financial data for testing"""
    return SAMPLE_FINANCIAL_DATA.copy()
