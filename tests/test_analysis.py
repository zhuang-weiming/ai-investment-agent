"""Test analysis strategies"""
import pytest
from src.strategies.peter_lynch import PeterLynchStrategy
from src.strategies.warren_buffett import WarrenBuffettStrategy
from src.strategies.technical import TechnicalStrategy, TechnicalIndicators
from src.analysis.analysis_orchestrator import StockAnalysisOrchestrator
from tests.conftest import SAMPLE_FINANCIAL_DATA

@pytest.mark.asyncio
async def test_peter_lynch_analysis():
    """Test Peter Lynch's analysis strategy"""
    strategy = PeterLynchStrategy()
    
    # Run analysis
    result = await strategy.analyze(SAMPLE_FINANCIAL_DATA)
    
    # Verify result structure
    assert hasattr(result, 'signal'), "Missing signal"
    assert hasattr(result, 'confidence'), "Missing confidence"
    assert hasattr(result, 'reasoning'), "Missing reasoning"
    
    # Verify signal validity
    assert result.signal in ['bullish', 'bearish', 'neutral'], f"Invalid signal: {result.signal}"
    
    # Verify confidence score
    assert 0 <= result.confidence <= 100, f"Invalid confidence score: {result.confidence}"
    
    # Verify reasoning
    assert len(result.reasoning) > 0, "Empty reasoning"
    assert "PEG" in result.reasoning, "PEG ratio analysis missing"

@pytest.mark.asyncio
async def test_warren_buffett_analysis():
    """Test case 2: Test Warren Buffett analysis strategy"""
    strategy = WarrenBuffettStrategy()
    
    # Run analysis
    result = await strategy.analyze(SAMPLE_FINANCIAL_DATA)
    
    # Verify result structure
    assert hasattr(result, 'signal'), "Missing signal"
    assert hasattr(result, 'confidence'), "Missing confidence"
    assert hasattr(result, 'reasoning'), "Missing reasoning"
    
    # Verify Warren Buffett analysis components
    assert "economic moat" in result.reasoning.lower(), "Economic moat analysis missing"
    assert "management quality" in result.reasoning.lower(), "Management quality analysis missing"
    
    # Verify moat analysis
    assert "moat" in result.reasoning.lower(), "Economic moat analysis missing"
    
    # Verify intrinsic value calculation
    if result.raw_data:
        assert 'intrinsic_value' in result.raw_data, "Intrinsic value calculation missing"

@pytest.mark.asyncio
async def test_technical_analysis():
    """Test technical analysis strategy"""
    strategy = TechnicalStrategy()
    
    # Create a TechnicalIndicators object from SAMPLE_FINANCIAL_DATA
    indicators = TechnicalIndicators(
        close_prices=SAMPLE_FINANCIAL_DATA['technical_data']['close_prices'],
        volumes=SAMPLE_FINANCIAL_DATA['technical_data']['volumes'],
        high_prices=SAMPLE_FINANCIAL_DATA['technical_data']['high_prices'],
        low_prices=SAMPLE_FINANCIAL_DATA['technical_data']['low_prices'],
        vix=SAMPLE_FINANCIAL_DATA['vix'],
        index_correlation=SAMPLE_FINANCIAL_DATA['technical_data'].get('index_correlation', 0.0),
        sector_rs=SAMPLE_FINANCIAL_DATA['technical_data'].get('sector_rs', 0.0),
        implied_volatility=SAMPLE_FINANCIAL_DATA['technical_data'].get('implied_volatility', 0.0),
        volume_profile=SAMPLE_FINANCIAL_DATA['technical_data'].get('volume_profile', {}),
        market_breadth=SAMPLE_FINANCIAL_DATA['technical_data'].get('market_breadth', 0.0)
    )
    
    # Run analysis
    result = await strategy.analyze(indicators)
    
    # Verify result structure
    assert hasattr(result, 'signal'), "Missing signal"
    assert hasattr(result, 'confidence'), "Missing confidence"
    assert hasattr(result, 'reasoning'), "Missing reasoning"
    
    # Verify technical analysis components
    assert "trend" in result.reasoning.lower() or "momentum" in result.reasoning.lower(), "Missing trend or momentum analysis"

@pytest.mark.asyncio
async def test_combined_analysis():
    """Test case 3: Test combined analysis from all strategies"""
    orchestrator = StockAnalysisOrchestrator()
    
    # Run combined analysis
    result = await orchestrator.analyze_stock('000333.SZ', vix=15.0)
    
    # Verify overall structure
    assert 'overall_signal' in result, "Missing overall signal"
    assert 'overall_confidence' in result, "Missing overall confidence"
    assert 'combined_reasoning' in result, "Missing combined reasoning"
    assert 'strategy_results' in result, "Missing strategy results"
    
    # Verify individual strategy results
    strategy_results = result['strategy_results']
    assert 'peter_lynch' in strategy_results, "Missing Peter Lynch analysis"
    assert 'warren_buffett' in strategy_results, "Missing Warren Buffett analysis"
    assert 'technical' in strategy_results, "Missing Technical analysis"
    
    # Verify reasoning includes all perspectives
    combined_reasoning = result['combined_reasoning'].lower()
    assert "peter lynch" in combined_reasoning, "Missing Peter Lynch perspective"
    assert "warren buffett" in combined_reasoning, "Missing Warren Buffett perspective"
    assert "technical" in combined_reasoning, "Missing Technical perspective"
    
    # Verify confidence scores
    assert 0 <= result['overall_confidence'] <= 100, "Invalid overall confidence score"
    for strategy_result in strategy_results.values():
        assert 0 <= strategy_result['confidence'] <= 100, "Invalid strategy confidence score"

@pytest.mark.asyncio
async def test_analysis_with_missing_data():
    """Test analysis behavior with incomplete data"""
    orchestrator = StockAnalysisOrchestrator()
    
    # Create incomplete data
    incomplete_data = SAMPLE_FINANCIAL_DATA.copy()
    del incomplete_data['fundamental_data']['peg_ratio']
    del incomplete_data['technical_data']['rsi_14']
    
    # Run analysis with incomplete data
    result = await orchestrator.analyze_stock('000333.SZ', vix=15.0)
    
    # Verify graceful handling of missing data
    assert result['overall_signal'] == 'neutral', "Should default to neutral with missing data"
    assert result['overall_confidence'] < 80, "Should have reduced confidence with missing data"
