from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

class AnalysisResult:
    """Strategy analysis result container"""
    def __init__(self, signal: str, confidence: float, reasoning: str, raw_data: Dict[str, Any] = None):
        self.signal = signal
        self.confidence = confidence
        self.reasoning = reasoning
        self.raw_data = raw_data or {}
        
        # Initialize dictionary storage
        self._dict = {}
        
        # Add all attributes to dictionary storage
        self._dict['signal'] = signal
        self._dict['confidence'] = confidence
        self._dict['reasoning'] = reasoning
        
        # Add raw data items to dictionary storage
        if raw_data and isinstance(raw_data, dict):
            for key, value in raw_data.items():
                self._dict[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Implement contains check for dictionary-like access"""
        return key in self._dict
    
    def __getitem__(self, key: str) -> Any:
        """Implement dictionary access"""
        return self._dict.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Implement dictionary assignment"""
        self._dict[key] = value
        
    @property
    def details(self):
        """Access to detailed analysis results"""
        return self._dict.get('details', {})

class BaseStrategy(ABC):
    """Base class for all analysis strategies"""
    
    @abstractmethod
    async def analyze(self, stock_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze the stock data and return a signal"""
        pass
    
    @abstractmethod
    def validate_data(self, stock_data: Dict[str, Any]) -> bool:
        """Validate that all required data is present"""
        pass
    
    def calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score (0-100) based on metrics"""
        # Implement common confidence calculation logic
        pass
