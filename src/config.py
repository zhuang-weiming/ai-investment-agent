"""Configuration settings for the AI Investment Agent"""
from typing import Dict, List

# Stock URLs configuration
STOCK_URLS: Dict[str, List[str]] = {
    "AAPL": [  # Apple Inc.
        "https://finance.yahoo.com/quote/AAPL",
        "https://www.marketwatch.com/investing/stock/aapl",
        "https://www.reuters.com/markets/companies/AAPL.O",
        "https://seekingalpha.com/symbol/AAPL",
    ],
    "TSLA": [  # Tesla
        "https://finance.yahoo.com/quote/TSLA",
        "https://www.marketwatch.com/investing/stock/tsla",
        "https://www.reuters.com/markets/companies/TSLA.O",
        "https://seekingalpha.com/symbol/TSLA",
    ],
    "MSFT": [  # Microsoft
        "https://finance.yahoo.com/quote/MSFT",
        "https://www.marketwatch.com/investing/stock/msft",
        "https://www.reuters.com/markets/companies/MSFT.O",
        "https://seekingalpha.com/symbol/MSFT",
    ],
    "NVDA": [  # NVIDIA
        "https://finance.yahoo.com/quote/NVDA",
        "https://www.marketwatch.com/investing/stock/nvda",
        "https://www.reuters.com/markets/companies/NVDA.O",
        "https://seekingalpha.com/symbol/NVDA",
    ]
}

# LLM Model Configuration
MODEL_CONFIG = {
    "name": "qwen3:14b",  # Updated to match your local model name
    "temperature": 0.7,
    "max_tokens": 32000,
}

# System prompt for stock analysis
SYSTEM_PROMPT = """You are an expert financial analyst AI assistant with deep knowledge in stock market analysis.
For the given stock, analyze the following aspects based on the provided data:

1. Current Market Position:
   - Stock price trends and key levels
   - Trading volume analysis
   - Market capitalization context

2. Financial Health:
   - Key financial metrics
   - Revenue and profit trends
   - Debt and cash position

3. Technical Analysis:
   - Moving averages (50-day, 200-day)
   - RSI and momentum indicators
   - Support and resistance levels

4. Market Sentiment:
   - Recent news impact
   - Analyst recommendations
   - Institutional investor positions

5. Risk Assessment:
   - Market risks
   - Company-specific risks
   - Industry/sector risks

6. Forward-Looking Analysis:
   - Short-term outlook (1-3 months)
   - Long-term prospects (1-2 years)
   - Growth catalysts and potential headwinds

Provide a balanced analysis with specific data points and clear reasoning for your conclusions.
Express confidence levels in your predictions and highlight key factors that could change the outlook.
"""

# Default user question template
DEFAULT_USER_QUESTION = """Based on the current market data and news, what is your comprehensive analysis 
and prediction for {stock_symbol}? Please provide specific insights about recent developments and their potential impact."""
