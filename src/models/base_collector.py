"""Base data collector interface for web scraping"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseDataCollector(ABC):
    """Abstract base class for web data collectors"""
    
    @abstractmethod
    async def collect_stock_data(self, symbol: str, vix: float) -> Dict[str, Any]:
        """Collect all required data for a stock from web sources"""
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up web scraping resources"""
        pass
    
    @abstractmethod
    def _process_scraped_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Process and combine data from different web sources"""
        pass
