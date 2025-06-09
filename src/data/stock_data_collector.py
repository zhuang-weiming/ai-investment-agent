# Stock data collector (moved from models)
from typing import Any, Dict, List
import re


class StockDataCollector:
    def _get_urls_for_symbol(self, symbol: str) -> Dict[str, str]:
        """Generate URLs for data collection for a given symbol

        Args:
            symbol: Stock symbol (e.g., '000333.SZ' or 'AAPL')

        Returns:
            Dictionary of URLs for different data sources
        """
        # Clean up symbol by removing extra suffixes
        clean_symbol = re.sub(r'\.(SZ|SH|HK|US).*$', '', symbol)

        # Base URLs for different markets
        if '.HK' in symbol:
            base_url = f"http://quote.eastmoney.com/hk/{clean_symbol}.html"
            data_url = f"http://emweb.securities.eastmoney.com/PC_HKF10/CompanyBigNews/PageAjax?code={clean_symbol}"
            finance_url = f"http://emweb.securities.eastmoney.com/PC_HKF10/FinancialAnalysis/PageAjax?code={clean_symbol}"
        elif '.SZ' in symbol or '.SH' in symbol:
            base_url = f"http://quote.eastmoney.com/sz/{clean_symbol}.html"
            data_url = f"http://emweb.securities.eastmoney.com/PC_HSF10/CompanyBigNews/PageAjax?code={clean_symbol}"
            finance_url = f"http://emweb.securities.eastmoney.com/PC_HSF10/FinancialAnalysis/PageAjax?code={clean_symbol}"
        else:  # US market
            base_url = f"https://finance.yahoo.com/quote/{clean_symbol}"
            data_url = f"https://finance.yahoo.com/quote/{clean_symbol}/key-statistics"
            finance_url = f"https://finance.yahoo.com/quote/{clean_symbol}/financials"

        return {
            "quote": base_url,
            "stats": data_url,
            "financials": finance_url,
        }

    def get_financials(self, symbol: str) -> Dict[str, Any]:
        """Get financial data from appropriate source based on symbol"""
        urls = self._get_urls_for_symbol(symbol)
        # Placeholder for actual implementation
        return {}

    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news data from appropriate source based on symbol"""
        urls = self._get_urls_for_symbol(symbol)
        # Placeholder for actual implementation
        return []

    def get_insider_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Get insider trading data from appropriate source based on symbol"""
        urls = self._get_urls_for_symbol(symbol)
        # Placeholder for actual implementation
        return []

    def get_market_cap(self, symbol: str) -> float:
        """Get market cap from appropriate source based on symbol"""
        urls = self._get_urls_for_symbol(symbol)
        # Placeholder for actual implementation
        return 0.0
