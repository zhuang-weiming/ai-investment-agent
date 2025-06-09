import pytest
import json
import sys
import typing
from unittest.mock import patch, MagicMock

# Mock the strategies modules before importing app
sys.modules['src.agents.technical_agent'] = MagicMock()
sys.modules['src.agents.peter_lynch_agent'] = MagicMock()
sys.modules['src.agents.warren_buffett_agent'] = MagicMock()
sys.modules['src.strategies.technical'] = MagicMock()
sys.modules['src.strategies.peter_lynch'] = MagicMock()
sys.modules['src.strategies.warren_buffett'] = MagicMock()

# Now import the app
from src.web.app import app

@pytest.fixture
def client():
    """Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_analysis_result():
    """Mock analysis result for testing"""
    return {
        'symbol': 'AAPL',
        'overall_signal': 'bullish',
        'overall_confidence': 75.5,
        'combined_reasoning': 'Combined reasoning text',
        'strategy_results': {
            'peter_lynch': {
                'signal': 'bullish',
                'confidence': 80.0,
                'reasoning': 'Peter Lynch reasoning'
            },
            'warren_buffett': {
                'signal': 'neutral',
                'confidence': 65.0,
                'reasoning': 'Warren Buffett reasoning'
            },
            'technical': {
                'signal': 'bullish',
                'confidence': 85.0,
                'reasoning': 'Technical reasoning'
            }
        }
    }

def test_index_route(client):
    """Test the index route returns the correct template"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'AI Investment Advisor' in response.data
    assert b'Stock Code' in response.data

def test_analyze_route_success(client, mock_analysis_result):
    """Test the analyze route with successful analysis"""
    # Create a mock that returns a coroutine function
    async def mock_coro(*args, **kwargs):
        return mock_analysis_result
    
    # Patch the analyze_stock method to return our awaitable mock
    with patch('src.web.app.orchestrator.analyze_stock', mock_coro):
        response = client.post('/analyze', data={'stock_code': 'AAPL'})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['symbol'] == 'AAPL'
        assert data['overall_signal'] == 'bullish'
        assert data['overall_confidence'] == '75.5%'
        assert 'analysis' in data
        assert 'peter_lynch' in data['analysis']
        assert 'warren_buffett' in data['analysis']
        assert 'technical' in data['analysis']

def test_analyze_route_missing_stock_code(client):
    """Test the analyze route with missing stock code"""
    response = client.post('/analyze', data={})
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Stock code is required' in data['error']

def test_analyze_route_error(client):
    """Test the analyze route with an error during analysis"""
    # Create a mock that raises an exception when awaited
    async def mock_error(*args, **kwargs):
        raise Exception('Test error')
    
    with patch('src.web.app.orchestrator.analyze_stock', mock_error):
        response = client.post('/analyze', data={'stock_code': 'INVALID'})
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Test error' in data['error']