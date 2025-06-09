# Agent for Peter Lynch strategy using LLM and data collectors

import typing
from src.data.eastmoney_collector import EastmoneyCollector
from src.data.akshare_collector import AKShareCollector
from src.llm.stock_analyzer import StockAnalyzer
from src.config import MODEL_CONFIG

class PeterLynchAgent:
    def __init__(self):
        self.eastmoney_collector = EastmoneyCollector()
        self.ak_collector = AKShareCollector()
        # Force no_think mode by explicitly setting the model name
        self.llm = StockAnalyzer(model_name="qwen3-nothink:14b")

    def analyze(self, symbol: str) -> typing.Dict[str, typing.Any]:
        # 1. Collect data from Eastmoney (China market focus)
        financials = self.eastmoney_collector.get_financials(symbol)
        news = self.eastmoney_collector.get_news(symbol)
        insider = self.eastmoney_collector.get_insider_trades(symbol)
        market_cap = self.eastmoney_collector.get_market_cap(symbol)

        # Fallback to AKShare if Eastmoney fails
        if not financials:
            financials = self.ak_collector.get_financials(symbol)
        if not news:
            news = self.ak_collector.get_news(symbol)
        if not insider:
            insider = self.ak_collector.get_insider_trades(symbol)
        if not market_cap:
            market_cap = self.ak_collector.get_market_cap(symbol)

        # 2. Use LLM to analyze and generate reasoning
        prompt = self.llm.build_peter_lynch_prompt(financials, news, insider, market_cap)
        result = self.llm.analyze(prompt)
        return result
