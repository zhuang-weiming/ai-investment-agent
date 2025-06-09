import pytest
from src.analysis.analysis_orchestrator import StockAnalysisOrchestrator
from src.data.stock_data_collector import StockDataCollector
import asyncio
import typing
from datetime import datetime

# Define typing aliases
Dict = typing.Dict
Any = typing.Any
List = typing.List

class MockStockDataCollector(StockDataCollector):
    """Mock data collector for testing"""
    def __init__(self, mock_data: Dict[str, Any]):
        self.mock_data = mock_data
        
    async def collect_stock_data(self, symbol: str, vix: float = None) -> Dict[str, Any]:
        data = self.mock_data.copy()
        
        # Restructure price data correctly
        data['price_data'] = {
            'close_prices': data.pop('close_prices'),
            'volumes': data.pop('volumes'),
            'high_prices': data.pop('high_prices'),
            'low_prices': data.pop('low_prices'),
            'vix': vix or 15.5
        }
        data['symbol'] = symbol
        return data
        
    async def close(self):
        pass

@pytest.fixture
def sample_stock_data() -> Dict[str, Any]:
    """Fixture providing sample stock data with good metrics"""
    return {
        'symbol': 'AAPL',
        'pages': [{
            'text': '''
            Financial Metrics:
            PEG Ratio: 0.85
            Revenue Growth: 22.5%
            EPS Growth: 28.2%
            Debt/Equity: 1.1
            Free Cash Flow Margin: 7.5%
            P/E Ratio: 15.5
            ROE: 42.5%
            Current Ratio: 1.8
            Operating Margin: 25.2%
            Owner Earnings: 25000000000
            Free Cash Flow: 80000000000
            Market Share: 15.5%
            Brand Value Score: 95.5
            
            Insider Trading Activity:
            Latest Purchase: 15,000 shares by CEO
            Previous Sale: 5,000 shares by CFO
            Insider Ownership: 8.5%
            '''
        }],
        'close_prices': [150.0, 152.0, 155.0, 153.0, 156.0],
        'volumes': [1000000, 1200000, 950000, 1100000, 1300000],
        'high_prices': [152.0, 154.0, 156.0, 155.0, 158.0],
        'low_prices': [149.0, 151.0, 153.0, 152.0, 155.0],
        'vix': 15.5,
        'revenue_growth': 22.5,
        'eps_growth': 28.2,
        'peg_ratio': 0.85,
        'debt_to_equity': 1.1,
        'fcf_margin': 0.075,
        'pe_ratio': 15.5,
        'roe': 0.425,
        'current_ratio': 1.8,
        'operating_margin': 0.252,
        'owner_earnings': 25000000000,
        'debt_ratio': 0.35,
        'fcf': 80000000000,
        'moat_type': 'wide',
        'mgmt_score': 85,
        'industry_pe': 18.5,
        'pb_ratio': 5.2,
        'historical_pb': 4.8,
        'market_share': 15.5,
        'brand_value': 95.5,
        'insider_ownership': 8.5,
        'buyback_efficiency': 0.8,
        'capital_allocation': 90,
        'industry_growth': 12.5,
        'market_concentration': 0.75,
        'regulatory_risk': 'low',
        'industry_cycle_position': 'growth',
        'competitive_position': 'leader',
        'news_sentiment': 7.2,
        'insider_buys': 2,
        'insider_sells': 1,
        'timestamp': datetime.now().isoformat()
    }

@pytest.mark.asyncio
async def test_full_analysis_pipeline(sample_stock_data):
    """Test the complete analysis pipeline from data collection to strategy signals"""
    # Setup
    orchestrator = StockAnalysisOrchestrator()
    orchestrator.data_collector = MockStockDataCollector(sample_stock_data)
    
    # Run analysis
    results = await orchestrator.analyze_stock('AAPL', vix=15.5)
    
    # Validate overall structure
    assert results is not None
    assert results['symbol'] == 'AAPL'
    assert 'data_timestamp' in results
    assert results['vix'] == 15.5
    
    # Validate presence of strategy results
    assert 'strategy_results' in results
    strategy_names = ['peter_lynch', 'warren_buffett', 'technical']
    for strategy in strategy_names:
        assert strategy in results['strategy_results']
        strategy_result = results['strategy_results'][strategy]
        assert 'signal' in strategy_result
        assert 'confidence' in strategy_result
        assert 'reasoning' in strategy_result
        assert strategy_result['signal'] in ['bullish', 'neutral', 'bearish']
        assert 0 <= strategy_result['confidence'] <= 100
    
    # Validate final analysis results
    assert 'overall_signal' in results
    assert results['overall_signal'] in ['bullish', 'neutral', 'bearish']
    assert 'overall_confidence' in results
    assert 0 <= results['overall_confidence'] <= 100
    assert 'combined_reasoning' in results
    for strategy in strategy_names:
        assert strategy.replace('_', ' ').title() in results['combined_reasoning']

@pytest.mark.asyncio
async def test_error_handling():
    """Test how the system handles errors in data collection"""
    class ErrorCollector(StockDataCollector):
        async def collect_stock_data(self, symbol: str, vix: float = None) -> Dict[str, Any]:
            raise Exception("API Error: Unable to fetch data")
            
        async def close(self):
            pass
    
    orchestrator = StockAnalysisOrchestrator()
    orchestrator.data_collector = ErrorCollector()
    
    # System should handle errors gracefully
    with pytest.raises(Exception) as exc_info:
        await orchestrator.analyze_stock('AAPL', vix=15.5)
    assert "Unable to fetch data" in str(exc_info.value)

@pytest.mark.asyncio
async def test_concurrent_analysis(sample_stock_data):
    """Test the system's ability to handle concurrent stock analysis"""
    orchestrator = StockAnalysisOrchestrator()
    orchestrator.data_collector = MockStockDataCollector(sample_stock_data)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    vix = 15.5
    
    tasks = [orchestrator.analyze_stock(symbol, vix) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == len(symbols)
    for result in results:
        assert result is not None
        assert result['symbol'] in symbols
        assert 'strategy_results' in result
        # Each strategy should provide results
        assert all(strategy in result['strategy_results'] 
                  for strategy in ['peter_lynch', 'warren_buffett', 'technical'])
        # Each result should have confidence and signal
        for strategy_result in result['strategy_results'].values():
            assert 0 <= strategy_result['confidence'] <= 100
            assert strategy_result['signal'] in ['bullish', 'neutral', 'bearish']
        assert result['overall_signal'] in ['bullish', 'neutral', 'bearish']
        assert 0 <= result['overall_confidence'] <= 100
        assert len(result['combined_reasoning']) > 0

@pytest.mark.asyncio
async def test_missing_data_handling(sample_stock_data):
    """Test how the system handles missing or incomplete data"""
    # Create incomplete data by removing some key metrics
    incomplete_data = sample_stock_data.copy()
    del incomplete_data['peg_ratio']
    del incomplete_data['revenue_growth']
    del incomplete_data['owner_earnings']
    del incomplete_data['market_share']
    
    orchestrator = StockAnalysisOrchestrator()
    orchestrator.data_collector = MockStockDataCollector(incomplete_data)
    
    # System should still provide analysis with available data
    result = await orchestrator.analyze_stock('AAPL', vix=15.5)
    
    assert result is not None
    assert result['symbol'] == 'AAPL'
    assert result['overall_signal'] in ['bullish', 'neutral', 'bearish']
    assert 'strategy_results' in result
    assert 'combined_reasoning' in result
    
    # Each strategy should still provide results
    for strategy_name, strategy_result in result['strategy_results'].items():
        assert strategy_result['signal'] in ['bullish', 'neutral', 'bearish']
        assert 0 <= strategy_result['confidence'] <= 100
        
        # For strategies with missing data, verify they mention it in reasoning
        if strategy_name in ['warren_buffett', 'technical']:
            assert ('missing' in strategy_result['reasoning'].lower() or 
                   'insufficient' in strategy_result['reasoning'].lower())
        
    # Overall confidence should reflect data quality
    assert result['overall_confidence'] < 90
