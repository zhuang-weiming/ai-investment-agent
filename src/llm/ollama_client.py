# Ollama Qwen3 14B LLM client
import requests
from src.config import MODEL_CONFIG

class OllamaQwenAnalyzer:
    def __init__(self, base_url='http://localhost:11434', model=None):
        self.base_url = base_url
        self.model = model or MODEL_CONFIG["name"]

    def build_peter_lynch_prompt(self, financials, news, insider, market_cap):
        # Compose a prompt for the LLM based on the Peter Lynch strategy
        return f"""
You are a financial analyst using the Peter Lynch strategy. Analyze the following data:
Financials: {financials}
News: {news}
Insider Trades: {insider}
Market Cap: {market_cap}
Give a bullish, neutral, or bearish signal with confidence (0-100) and reasoning.
"""

    def analyze(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()
        return {"signal": "error", "confidence": 0, "reasoning": "LLM call failed"}
