import pytest
from src.analysis.analysis_orchestrator import StockAnalysisOrchestrator
from src.data.stock_data_collector import StockDataCollector
import typing
from datetime import datetime

# Define typing aliases
Dict = typing.Dict
Any = typing.Any

class MockStockDataCollector(StockDataCollector):
    """Mock data collector that returns different data based on market conditions"""
    def __init__(self, mock_data: Dict[str, Any]):
        self.mock_data = mock_data
        
    async def collect_stock_data(self, symbol: str, vix: float = None) -> Dict[str, Any]:
        data = self.mock_data.copy()
        
        # Add VIX to the data
        data['vix'] = vix
        
        # Restructure price data correctly
        data['price_data'] = {
            'close_prices': data.pop('close_prices'),
            'volumes': data.pop('volumes'),
            'high_prices': data.pop('high_prices'),
            'low_prices': data.pop('low_prices'),
            'vix': vix
        }
        data['symbol'] = symbol
        return data
        
    async def close(self):
        pass

@pytest.fixture
def bull_market_data() -> Dict[str, Any]:
    """Fixture providing sample stock data for a bull market"""
    return {
        'symbol': 'AAPL',
        'pages': [{
            'text': '''
            Financial Metrics:
            PEG Ratio: 0.85
            Revenue Growth: 22.5%
            EPS Growth: 28.2%
            Debt/Equity: 0.8
            Free Cash Flow Margin: 12.5%
            P/E Ratio: 22.5
            ROE: 42.5%
            Current Ratio: 2.2
            Operating Margin: 30.2%
            Owner Earnings: 35000000000
            Free Cash Flow: 90000000000
            Market Share: 18.5%
            Brand Value Score: 95.5
            
            Insider Trading Activity:
            Latest Purchase: 25,000 shares by CEO
            Previous Sale: 5,000 shares by CFO
            Insider Ownership: 10.5%
            '''
        }],
        'close_prices': [150.0, 155.0, 160.0, 165.0, 170.0],  # Uptrend
        'volumes': [1500000, 1600000, 1700000, 1800000, 1900000],  # Increasing volume
        'high_prices': [152.0, 157.0, 162.0, 167.0, 172.0],
        'low_prices': [149.0, 154.0, 159.0, 164.0, 169.0],
        'revenue_growth': 22.5,
        'eps_growth': 28.2,
        'peg_ratio': 0.85,
        'debt_to_equity': 0.8,
        'fcf_margin': 0.125,
        'pe_ratio': 22.5,
        'roe': 0.425,
        'current_ratio': 2.2,
        'operating_margin': 0.302,
        'owner_earnings': 35000000000,
        'debt_ratio': 0.25,
        'fcf': 90000000000,
        'moat_type': 'wide',
        'mgmt_score': 90,
        'industry_pe': 25.0,
        'pb_ratio': 6.2,
        'historical_pb': 5.8,
        'market_share': 18.5,
        'brand_value': 95.5,
        'insider_ownership': 10.5,
        'buyback_efficiency': 0.9,
        'capital_allocation': 95,
        'industry_growth': 15.5,
        'market_concentration': 0.85,
        'regulatory_risk': 'low',
        'industry_cycle_position': 'growth',
        'competitive_position': 'leader',
        'news_sentiment': 8.5,
        'insider_buys': 3,
        'insider_sells': 1,
        'timestamp': datetime.now().isoformat()
    }

@pytest.fixture
def bear_market_data() -> Dict[str, Any]:
    """Fixture providing sample stock data for a bear market"""
    return {
        'symbol': 'AAPL',
        'pages': [{
            'text': '''
            Financial Metrics:
            PEG Ratio: 1.25
            Revenue Growth: 5.5%
            EPS Growth: 2.2%
            Debt/Equity: 1.5
            Free Cash Flow Margin: 5.5%
            P/E Ratio: 15.5
            ROE: 22.5%
            Current Ratio: 1.2
            Operating Margin: 18.2%
            Owner Earnings: 15000000000
            Free Cash Flow: 40000000000
            Market Share: 15.5%
            Brand Value Score: 85.5
            
            Insider Trading Activity:
            Latest Purchase: 5,000 shares by CEO
            Previous Sale: 15,000 shares by CFO
            Insider Ownership: 6.5%
            '''
        }],
        'close_prices': [170.0, 165.0, 160.0, 155.0, 150.0],  # Downtrend
        'volumes': [2500000, 2600000, 2700000, 2800000, 2900000],  # High volume selling
        'high_prices': [172.0, 167.0, 162.0, 157.0, 152.0],
        'low_prices': [169.0, 164.0, 159.0, 154.0, 149.0],
        'revenue_growth': 5.5,
        'eps_growth': 2.2,
        'peg_ratio': 1.25,
        'debt_to_equity': 1.5,
        'fcf_margin': 0.055,
        'pe_ratio': 15.5,
        'roe': 0.225,
        'current_ratio': 1.2,
        'operating_margin': 0.182,
        'owner_earnings': 15000000000,
        'debt_ratio': 0.45,
        'fcf': 40000000000,
        'moat_type': 'narrow',
        'mgmt_score': 70,
        'industry_pe': 14.0,
        'pb_ratio': 3.2,
        'historical_pb': 4.8,
        'market_share': 15.5,
        'brand_value': 85.5,
        'insider_ownership': 6.5,
        'buyback_efficiency': 0.6,
        'capital_allocation': 75,
        'industry_growth': 5.5,
        'market_concentration': 0.65,
        'regulatory_risk': 'medium',
        'industry_cycle_position': 'contraction',
        'competitive_position': 'challenger',
        'news_sentiment': 4.5,
        'insider_buys': 1,
        'insider_sells': 3,
        'timestamp': datetime.now().isoformat()
    }

async def test_bull_market_analysis(bull_market_data):
    """Test analysis in bull market conditions with low VIX"""
    # Setup
    orchestrator = StockAnalysisOrchestrator()
    orchestrator.data_collector = MockStockDataCollector(bull_market_data)
    
    # Run analysis with low VIX (bull market)
    results = await orchestrator.analyze_stock('AAPL', vix=12.5)
    
    # Validate results
    assert results is not None
    assert results['symbol'] == 'AAPL'
    assert results['vix'] == 12.5
    
    # In bull market conditions, we expect a bullish signal
    assert results['overall_signal'] == 'bullish'
    assert results['overall_confidence'] > 70  # High confidence in bull market
    
    # Check individual strategies
    assert results['strategy_results']['technical']['signal'] == 'bullish'
    assert results['strategy_results']['technical']['confidence'] > 75
    
    # Warren Buffett might be more cautious even in bull markets
    assert results['strategy_results']['warren_buffett']['confidence'] > 60

async def test_bear_market_analysis(bear_market_data):
    """Test analysis in bear market conditions with high VIX"""
    # Setup
    orchestrator = StockAnalysisOrchestrator()
    orchestrator.data_collector = MockStockDataCollector(bear_market_data)
    
    # Run analysis with high VIX (bear market)
    results = await orchestrator.analyze_stock('AAPL', vix=35.0)
    
    # Validate results
    assert results is not None
    assert results['symbol'] == 'AAPL'
    assert results['vix'] == 35.0
    
    # In bear market conditions, we expect a bearish or neutral signal
    assert results['overall_signal'] in ['bearish', 'neutral']
    
    # Check individual strategies
    assert results['strategy_results']['technical']['signal'] == 'bearish'
    
    # Combined reasoning should mention high volatility
    assert 'volatility' in results['combined_reasoning'].lower() or 'vix' in results['combined_reasoning'].lower()

async def test_mixed_signals_analysis(bull_market_data, bear_market_data):
    """Test analysis with mixed signals from different strategies"""
    # Create mixed data: bullish technicals but bearish fundamentals
    mixed_data = bull_market_data.copy()
    
    # Replace fundamental metrics with bear market data
    for key in ['revenue_growth', 'eps_growth', 'peg_ratio', 'debt_to_equity',
                'fcf_margin', 'pe_ratio', 'roe', 'current_ratio']:
        mixed_data[key] = bear_market_data[key]
    
    # Setup
    orchestrator = StockAnalysisOrchestrator()
    orchestrator.data_collector = MockStockDataCollector(mixed_data)
    
    # Run analysis with moderate VIX
    results = await orchestrator.analyze_stock('AAPL', vix=20.0)
    
    # Validate results
    assert results is not None
    
    # With mixed signals, we expect the confidence to be lower
    assert results['overall_confidence'] < 75
    
    # The combined reasoning should mention conflicting signals
    assert 'conflict' in results['combined_reasoning'].lower() or \
           'mixed' in results['combined_reasoning'].lower() or \
           'contrast' in results['combined_reasoning'].lower()