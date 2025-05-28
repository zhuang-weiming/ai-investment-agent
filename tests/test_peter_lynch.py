import pytest
from src.strategies.peter_lynch import PeterLynchStrategy
from src.strategies.base_strategy import AnalysisResult

@pytest.fixture
def sample_stock_data():
    """Sample data for a good growth stock at reasonable price"""
    return {
        'revenue_growth': 0.255,  # 25.5%
        'eps_growth': 0.302,  # 30.2%
        'net_income': 500000000,
        'operating_margin': 0.22,  # 22%
        'free_cash_flow': 400000000,
        'total_debt': 1000000000,
        'shareholders_equity': 2000000000,
        'market_cap': 10000000000,
        'pe_ratio': 20,
        'peg_ratio': 0.75,
        'news_sentiment': 7.5,
        'insider_buys': 3,
        'insider_sells': 1
    }

@pytest.fixture
def poor_stock_data():
    """Sample data for an overvalued stock with poor fundamentals"""
    return {
        'revenue_growth': -0.052,  # -5.2%
        'eps_growth': -0.085,  # -8.5%
        'net_income': -50000000,
        'operating_margin': 0.05,  # 5%
        'free_cash_flow': -20000000,
        'total_debt': 800000000,
        'shareholders_equity': 200000000,
        'market_cap': 2000000000,
        'pe_ratio': None,  # Negative earnings
        'peg_ratio': None,  # Can't calculate with negative growth
        'news_sentiment': 3.0,
        'insider_buys': 0,
        'insider_sells': 2
    }

def test_analyze_growth():
    """Test growth analysis calculation"""
    strategy = PeterLynchStrategy()
    
    # Test strong growth
    growth = strategy.analyze_growth({
        'revenue_growth': 0.30,
        'eps_growth': 0.25
    })
    assert growth['score'] >= 8.0  # High score for strong growth
    
    # Test poor growth
    growth = strategy.analyze_growth({
        'revenue_growth': -0.05,
        'eps_growth': -0.08
    })
    assert growth['score'] <= 3.0  # Low score for negative growth

def test_analyze_fundamentals():
    """Test fundamentals analysis"""
    strategy = PeterLynchStrategy()
    
    # Test good fundamentals
    fundamentals = strategy.analyze_fundamentals({
        'total_debt': 1000000000,
        'shareholders_equity': 2000000000,  # D/E = 0.5
        'operating_margin': 0.22,  # 22%
        'free_cash_flow': 400000000
    })
    assert fundamentals['score'] >= 8.0
    
    # Test poor fundamentals
    fundamentals = strategy.analyze_fundamentals({
        'total_debt': 800000000,
        'shareholders_equity': 200000000,  # D/E = 4.0
        'operating_margin': 0.05,  # 5%
        'free_cash_flow': -20000000
    })
    assert fundamentals['score'] <= 3.0

def test_analyze_valuation():
    """Test GARP valuation analysis"""
    strategy = PeterLynchStrategy()
    
    # Test attractive valuation
    valuation = strategy.analyze_valuation({
        'peg_ratio': 0.75,
        'market_cap': 10000000000,
        'net_income': 800000000  # P/E = 12.5
    })
    assert valuation['score'] >= 8.0
    
    # Test expensive valuation
    valuation = strategy.analyze_valuation({
        'peg_ratio': 2.5,
        'market_cap': 10000000000,
        'net_income': 200000000  # P/E = 50
    })
    assert valuation['score'] <= 3.0

def test_analyze_sentiment():
    """Test sentiment analysis"""
    strategy = PeterLynchStrategy()
    
    # Test positive sentiment
    sentiment = strategy.analyze_sentiment({
        'news_sentiment': 7.5,
        'insider_buys': 3,
        'insider_sells': 1
    })
    assert sentiment['score'] >= 7.0
    
    # Test negative sentiment
    sentiment = strategy.analyze_sentiment({
        'news_sentiment': 3.0,
        'insider_buys': 0,
        'insider_sells': 2
    })
    assert sentiment['score'] <= 4.0

@pytest.mark.asyncio
async def test_analyze_bullish_case(sample_stock_data):
    """Test analysis for a good growth stock at reasonable price"""
    strategy = PeterLynchStrategy()
    result = await strategy.analyze(sample_stock_data)
    
    assert isinstance(result, AnalysisResult)
    assert result.signal == "bullish"
    assert result.confidence >= 70  # High confidence for strong metrics
    assert "ten-bagger" in result.reasoning.lower()
    assert "peg ratio" in result.reasoning.lower()
    assert result.raw_data == sample_stock_data

@pytest.mark.asyncio
async def test_analyze_bearish_case(poor_stock_data):
    """Test analysis for an overvalued stock with poor fundamentals"""
    strategy = PeterLynchStrategy()
    result = await strategy.analyze(poor_stock_data)
    
    assert isinstance(result, AnalysisResult)
    assert result.signal == "bearish"
    assert result.confidence >= 60  # Still confident in bearish signal
    assert "steering clear" in result.reasoning.lower()
    assert result.raw_data == poor_stock_data

@pytest.mark.asyncio
async def test_analyze_missing_data():
    """Test analysis with missing metrics"""
    strategy = PeterLynchStrategy()
    result = await strategy.analyze({
        'revenue_growth': None,
        'eps_growth': None,
        'market_cap': 1000000000
    })
    
    assert result.signal == "neutral"
    assert result.confidence == 30  # Low confidence with missing data
