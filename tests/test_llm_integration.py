import pytest
from unittest.mock import patch, MagicMock
from src.llm.ollama_client import OllamaQwenAnalyzer
from src.llm.stock_analyzer import StockAnalyzer
from langchain.llms import Ollama

@pytest.fixture
def mock_ollama_response():
    """Mock response from Ollama API"""
    return {
        "model": "qwen3-nothink:14b",
        "response": "This is a mock response from the Ollama API. Based on my analysis, the stock appears to be bullish with strong fundamentals."
    }

@pytest.fixture
def mock_ollama_qwen_analyzer(mock_ollama_response):
    """Mock OllamaQwenAnalyzer for testing"""
    with patch('src.llm.ollama_client.requests.post') as mock_post:
        # Setup the mock to return our mock response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=mock_ollama_response)
        )
        yield OllamaQwenAnalyzer(base_url="http://localhost:11434", model="qwen3-nothink:14b")

@pytest.fixture
def mock_langchain_ollama():
    """Mock Langchain Ollama for testing"""
    with patch('src.llm.stock_analyzer.Ollama') as mock_ollama:
        mock_instance = mock_ollama.return_value
        mock_instance.invoke = MagicMock(return_value="This is a mock analysis from Langchain Ollama.")
        yield mock_instance

@pytest.fixture
def stock_analyzer(mock_langchain_ollama):
    """Stock analyzer with mocked Ollama"""
    with patch('src.llm.stock_analyzer.Ollama', return_value=mock_langchain_ollama):
        analyzer = StockAnalyzer(model_name="qwen3-nothink:14b")
        yield analyzer

def test_ollama_qwen_analyzer_build_prompt(mock_ollama_qwen_analyzer):
    """Test that the OllamaQwenAnalyzer builds a prompt correctly"""
    prompt = mock_ollama_qwen_analyzer.build_peter_lynch_prompt(
        financials=[{"pe_ratio": 15.5}],
        news=[{"title": "Test News"}],
        insider=[{"trades": "Buy"}],
        market_cap=1000000000
    )
    
    assert prompt is not None
    assert isinstance(prompt, str)
    assert "Peter Lynch strategy" in prompt
    assert "financials" in prompt.lower()
    assert "news" in prompt.lower()
    assert "insider" in prompt.lower()
    assert "market cap" in prompt.lower()

def test_ollama_qwen_analyzer_analyze(mock_ollama_qwen_analyzer, mock_ollama_response):
    """Test that the OllamaQwenAnalyzer can analyze a prompt"""
    result = mock_ollama_qwen_analyzer.analyze("Test prompt")
    
    assert result is not None
    assert result == mock_ollama_response

def test_stock_analyzer_initialization():
    """Test that the StockAnalyzer initializes correctly"""
    with patch('src.llm.stock_analyzer.Ollama') as mock_ollama:
        analyzer = StockAnalyzer(model_name="qwen3-nothink:14b")
        assert analyzer.llm is not None
        assert mock_ollama.called

def test_stock_analyzer_create_analysis_chain(stock_analyzer):
    """Test that the StockAnalyzer creates an analysis chain correctly"""
    chain = stock_analyzer.create_analysis_chain("Test system prompt")
    
    assert chain is not None
    assert hasattr(chain, 'llm')
    assert hasattr(chain, 'prompt')

def test_stock_analyzer_clean_scraped_data(stock_analyzer):
    """Test that the StockAnalyzer cleans scraped data correctly"""
    scraped_data = [
        {
            "url": "https://example.com",
            "price": 150.0,
            "change": "+1.5%",
            "news": [
                {"title": "Test News", "summary": "Test Summary"}
            ]
        },
        {
            "error": "Failed to scrape"
        }
    ]
    
    cleaned_data = stock_analyzer._clean_scraped_data(scraped_data)
    
    assert cleaned_data is not None
    assert isinstance(cleaned_data, str)
    assert "Source: https://example.com" in cleaned_data
    assert "Price: 150.0" in cleaned_data
    assert "Change: +1.5%" in cleaned_data
    assert "Test News" in cleaned_data
    assert "Test Summary" in cleaned_data
    assert "Failed to scrape" not in cleaned_data  # Error data should be skipped