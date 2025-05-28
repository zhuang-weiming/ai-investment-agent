import pytest
import pandas as pd
import numpy as np
from src.strategies.technical import TechnicalStrategy, TechnicalIndicators
from src.strategies.base_strategy import AnalysisResult

@pytest.fixture
def sample_technical_data():
    # Generate sample price data with an upward trend
    n_points = 200
    base_prices = np.linspace(100, 150, n_points)  # Upward trend
    # Add some noise to prices
    close_prices = base_prices + np.random.normal(0, 2, n_points)
    high_prices = close_prices + np.random.uniform(0, 2, n_points)
    low_prices = close_prices - np.random.uniform(0, 2, n_points)
    volumes = np.random.randint(1000000, 2000000, n_points)
    
    return TechnicalIndicators(
        close_prices=close_prices.tolist(),
        volumes=volumes.tolist(),
        high_prices=high_prices.tolist(),
        low_prices=low_prices.tolist(),
        vix=20.5,  # Moderate market volatility
        index_correlation=0.75,
        sector_rs=1.2,
        implied_volatility=25.0
    )

def test_technical_indicators_validation():
    # Test with valid data
    valid_data = {
        'close_prices': [1.0, 2.0, 3.0, 4.0, 5.0],
        'volumes': [100, 200, 300, 400, 500],
        'high_prices': [1.1, 2.2, 3.3, 4.4, 5.5],
        'low_prices': [0.9, 1.8, 2.7, 3.6, 4.5],
        'vix': 15.0
    }
    
    indicators = TechnicalIndicators(**valid_data)
    assert len(indicators.close_prices) == 5
    assert indicators.vix == 15.0

    # Test with inconsistent data lengths
    with pytest.raises(ValueError):
        invalid_data = valid_data.copy()
        invalid_data['volumes'] = [100, 200, 300]  # Wrong length
        TechnicalIndicators(**invalid_data)

def test_macd_calculation(sample_technical_data):
    strategy = TechnicalStrategy()
    prices = np.array(sample_technical_data.close_prices)
    macd_data = strategy.calculate_macd(prices)
    
    assert 'macd' in macd_data
    assert 'signal' in macd_data
    assert 'histogram' in macd_data
    assert len(macd_data['macd']) == len(prices)

def test_rsi_calculation(sample_technical_data):
    strategy = TechnicalStrategy()
    prices = np.array(sample_technical_data.close_prices)
    rsi = strategy.calculate_rsi(prices)
    
    assert len(rsi) == len(prices)
    assert all(0 <= val <= 100 for val in rsi)  # RSI should be between 0 and 100

def test_trend_signals(sample_technical_data):
    strategy = TechnicalStrategy()
    trend_data = strategy.calculate_trend_signals(sample_technical_data)
    
    assert isinstance(trend_data, dict)
    assert 'signal' in trend_data
    assert trend_data['signal'] in ['bullish', 'bearish', 'neutral']
    assert 0 <= trend_data['confidence'] <= 100
    assert trend_data['strength'] in ['strong', 'medium', 'weak']
    assert 'metrics' in trend_data

def test_momentum_signals(sample_technical_data):
    strategy = TechnicalStrategy()
    momentum_data = strategy.calculate_momentum_signals(sample_technical_data)
    
    assert isinstance(momentum_data, dict)
    assert 'signal' in momentum_data
    assert momentum_data['signal'] in ['bullish', 'bearish', 'neutral']
    assert 0 <= momentum_data['confidence'] <= 100
    assert 'metrics' in momentum_data

@pytest.mark.asyncio
async def test_analyze_technical_strategy(sample_technical_data):
    strategy = TechnicalStrategy()
    analysis_result = await strategy.analyze(sample_technical_data)
    assert isinstance(analysis_result, AnalysisResult)
    assert hasattr(analysis_result, 'signal')
    assert analysis_result.signal in ['bullish', 'bearish', 'neutral']
    assert hasattr(analysis_result, 'confidence')
    assert 0 <= analysis_result.confidence <= 100
    assert hasattr(analysis_result, 'reasoning')
    assert isinstance(analysis_result.reasoning, str)
    assert len(analysis_result.reasoning) > 0 # Check that reasoning is not empty
    
    assert 'details' in analysis_result
    assert isinstance(analysis_result['details'], dict)
    assert 'trend' in analysis_result['details']
    assert 'momentum' in analysis_result['details']
    assert 'volume' in analysis_result['details']
    
    # Example: Check for specific keywords in reasoning if certain conditions are met
    # This would depend on the sample_technical_data provided
    # For instance, if sample_technical_data is expected to be bullish:
    # if analysis_result['signal'] == 'bullish':
    #     assert "upward momentum" in analysis_result['reasoning'].lower()
    #     assert "strengthening bullish momentum" in analysis_result['reasoning'].lower() 
    #     assert "supporting price movements" in analysis_result['reasoning'].lower()
