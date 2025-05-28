from typing import Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy, AnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators container"""
    close_prices: List[float]
    volumes: List[float]
    high_prices: List[float]
    low_prices: List[float]
    vix: float
    index_correlation: float = 0.0  # Correlation with market index
    sector_rs: float = 0.0  # Relative strength vs sector
    implied_volatility: float = 0.0  # Option implied volatility
    volume_profile: Dict[float, float] = field(default_factory=dict)  # Price levels vs volume
    market_breadth: float = 0.0  # Market breadth indicator
    
    def __post_init__(self):
        """Initialize and validate technical indicators data"""
        # Validate required fields
        if not isinstance(self.close_prices, list) or len(self.close_prices) < 2:
            raise ValueError("close_prices must be a non-empty list with at least 2 values")
            
        if not isinstance(self.volumes, list):
            raise ValueError("volumes must be a list")
            
        if not isinstance(self.high_prices, list) or len(self.high_prices) < 2:
            raise ValueError("high_prices must be a non-empty list with at least 2 values")
            
        if not isinstance(self.low_prices, list) or len(self.low_prices) < 2:
            raise ValueError("low_prices must be a non-empty list with at least 2 values")
            
        # Convert numpy arrays to lists if needed
        self.close_prices = self._to_list(self.close_prices)
        self.volumes = self._to_list(self.volumes)
        self.high_prices = self._to_list(self.high_prices)
        self.low_prices = self._to_list(self.low_prices)
        
        # Ensure all lists have the same length
        lengths = {
            'close_prices': len(self.close_prices),
            'volumes': len(self.volumes),
            'high_prices': len(self.high_prices),
            'low_prices': len(self.low_prices)
        }
        
        # If lengths differ, truncate longer lists to match the shortest one
        if len(set(lengths.values())) != 1:
            min_length = min(lengths.values())
            if min_length > 0:
                self.close_prices = self.close_prices[:min_length]
                self.volumes = self.volumes[:min_length]
                self.high_prices = self.high_prices[:min_length]
                self.low_prices = self.low_prices[:min_length]
    
    def _to_list(self, data: Union[list, np.ndarray]) -> list:
        """Helper method to convert numpy arrays to lists"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data
        
    def get(self, key: str, default: Any = None) -> Any:
        """Implement dictionary-like get method"""
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
    
    def __getitem__(self, key: str):
        """Implement dictionary access for TechnicalIndicators"""
        try:
            # Try to get attribute directly
            value = object.__getattribute__(self, key)
            return value
        except AttributeError:
            # If attribute not found, try calculated properties
            if key == 'sma_50':
                return self.sma_50
            elif key == 'sma_200':
                return self.sma_200
            elif key == 'rsi_14':
                return self.rsi_14
            elif key == 'macd':
                return self.macd
            else:
                raise KeyError(f"No attribute or calculated indicator named '{key}'")
    
    @property
    def sma_50(self):
        """Calculate 50-day simple moving average"""
        if len(self.close_prices) >= 50:
            return np.mean(self.close_prices[-50:])
        elif len(self.close_prices) >= 1:
            return float(self.close_prices[-1])
        return 0.0
    
    @property
    def sma_200(self):
        """Calculate 200-day simple moving average"""
        if len(self.close_prices) >= 200:
            return np.mean(self.close_prices[-200:])
        elif len(self.close_prices) >= 1:
            return float(self.close_prices[-1])
        return 0.0
    
    @property
    def rsi_14(self):
        """Calculate 14-day Relative Strength Index"""
        if len(self.close_prices) >= 14:
            return self._calculate_rsi(np.array(self.close_prices))
        return 50.0  # Neutral RSI
    
    @property
    def macd(self):
        """Calculate MACD indicator"""
        if len(self.close_prices) >= 26:
            return self._calculate_macd(np.array(self.close_prices))
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Internal method to calculate RSI"""
        try:
            if len(prices) < 14:
                return 50.0
                
            deltas = np.diff(prices)
            seed = deltas[:14]
            up = seed[seed >= 0].sum() / 14
            down = -seed[seed < 0].sum() / 14
            
            rs = up / down if down != 0 else float('inf')
            rsi = 100. - 100. / (1. + rs)
            
            # Smooth with Wilder's formula for remaining values
            for i in range(14, len(deltas)):
                delta = deltas[i]
                if delta > 0:
                    up = (up * 13 + delta) / 14
                    down = (down * 13) / 14
                else:
                    up = (up * 13) / 14
                    down = (down * 13 - delta) / 14
                
                rs = up / down if down != 0 else float('inf')
                rsi = 100. - 100. / (1. + rs)
                
            return float(rsi)
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0  # Return neutral value on error
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Internal method to calculate MACD"""
        try:
            if len(prices) < 26:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
                
            # Calculate EMAs
            ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().values
            
            # Calculate MACD line
            macd_line = ema12 - ema26
            
            # Calculate signal line (9-day EMA of MACD line)
            signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line[-1]),
                'signal': float(signal_line[-1]),
                'histogram': float(histogram[-1])
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}


class TechnicalStrategy(BaseStrategy):
    """Technical analysis strategy implementation"""
    
    def validate_data(self, stock_data: Dict[str, Any]) -> bool:
        """Validate that all required data is present"""
        # Check if stock_data is a TechnicalIndicators instance
        if not isinstance(stock_data, TechnicalIndicators):
            return False
            
        # Check if we have enough price data
        if not stock_data.close_prices or len(stock_data.close_prices) < 14:  # Need at least 14 days for RSI
            return False
            
        # Check if we have volume data
        if not stock_data.volumes or len(stock_data.volumes) < 1:
            return False
            
        return True
    
    def calculate_macd(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # Ensure we have enough data
        if len(prices) < 26:
            return {'macd': np.array([]), 'signal': np.array([]), 'histogram': np.array([])}
        
        # Calculate EMAs
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        # Calculate MACD line
        macd_line = ema12 - ema26
        
        # Calculate signal line (9-day EMA of MACD line)
        signal_line = self._calculate_ema(macd_line, 9)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_ema(self, data: np.ndarray, span: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        # Use pandas EMA implementation for simplicity
        return pd.Series(data).ewm(span=span, adjust=False).mean().values
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        # Ensure we have enough data
        if len(prices) <= period:
            return np.array([50.0] * len(prices))  # Default neutral value
        
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:  # Avoid division by zero
            rs = float('inf')
        else:
            rs = up / down
        
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        # Calculate RSI for the rest of the data
        for i in range(period, len(prices)):
            delta = deltas[i-1]  # Current price change
            
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            # Update moving averages
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            # Calculate RS and RSI
            if down == 0:  # Avoid division by zero
                rs = float('inf')
            else:
                rs = up / down
            
            rsi[i] = 100. - 100. / (1. + rs)
        
        # Ensure RSI is within bounds
        rsi = np.clip(rsi, 0, 100)
        return rsi
    
    def calculate_trend_signals(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Calculate trend signals based on moving averages and price action"""
        # Default values
        signal = 'neutral'
        confidence = 50.0
        strength = 'medium'
        metrics = {}
        
        # Check if we have enough data for calculations
        if len(indicators.close_prices) < 50:
            return {
                'signal': signal,
                'confidence': confidence,
                'strength': strength,
                'metrics': {'error': 'Insufficient data for trend analysis'}
            }
        
        # Get current price and moving averages
        current_price = indicators.close_prices[-1]
        sma_50 = indicators.sma_50
        sma_200 = indicators.sma_200 if indicators.sma_200 is not None else None
        
        # Calculate trend metrics
        metrics['price_vs_sma50'] = current_price / sma_50 - 1 if sma_50 else 0
        
        if sma_200:
            metrics['price_vs_sma200'] = current_price / sma_200 - 1
            metrics['sma50_vs_sma200'] = sma_50 / sma_200 - 1
            
            # Golden cross / death cross detection
            metrics['golden_cross'] = metrics['sma50_vs_sma200'] > 0
            
            # Determine trend signal based on moving averages
            if metrics['price_vs_sma50'] > 0 and metrics['price_vs_sma200'] > 0 and metrics['golden_cross']:
                signal = 'bullish'
                confidence = min(70 + 30 * metrics['price_vs_sma200'], 100)
                strength = 'strong' if metrics['price_vs_sma200'] > 0.1 else 'medium'
            elif metrics['price_vs_sma50'] < 0 and metrics['price_vs_sma200'] < 0 and not metrics['golden_cross']:
                signal = 'bearish'
                confidence = min(70 + 30 * abs(metrics['price_vs_sma200']), 100)
                strength = 'strong' if metrics['price_vs_sma200'] < -0.1 else 'medium'
            elif metrics['price_vs_sma50'] > 0:
                signal = 'bullish'
                confidence = 60
                strength = 'medium'
            elif metrics['price_vs_sma50'] < 0:
                signal = 'bearish'
                confidence = 60
                strength = 'medium'
        else:
            # Simplified trend analysis with just 50-day SMA
            if metrics['price_vs_sma50'] > 0.05:
                signal = 'bullish'
                confidence = 65
                strength = 'medium'
            elif metrics['price_vs_sma50'] < -0.05:
                signal = 'bearish'
                confidence = 65
                strength = 'medium'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'strength': strength,
            'metrics': metrics
        }
    
    def calculate_momentum_signals(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Calculate momentum signals based on RSI, MACD, and volume"""
        # Default values
        signal = 'neutral'
        confidence = 50.0
        metrics = {}
        
        # Check if we have enough data
        if len(indicators.close_prices) < 26:  # Need at least 26 days for MACD
            return {
                'signal': signal,
                'confidence': confidence,
                'metrics': {'error': 'Insufficient data for momentum analysis'}
            }
        
        # Get RSI value
        rsi = indicators.rsi_14
        if rsi is not None:
            metrics['rsi'] = rsi
            
            # RSI signal
            if rsi > 70:
                metrics['rsi_signal'] = 'overbought'
            elif rsi < 30:
                metrics['rsi_signal'] = 'oversold'
            else:
                metrics['rsi_signal'] = 'neutral'
        
        # Get MACD values
        macd_data = indicators.macd
        if macd_data and isinstance(macd_data, dict):
            # Use the last values if they're arrays
            macd_value = macd_data['macd'] if not isinstance(macd_data['macd'], np.ndarray) else macd_data['macd'][-1]
            signal_value = macd_data['signal'] if not isinstance(macd_data['signal'], np.ndarray) else macd_data['signal'][-1]
            histogram = macd_data['histogram'] if not isinstance(macd_data['histogram'], np.ndarray) else macd_data['histogram'][-1]
            
            metrics['macd'] = macd_value
            metrics['macd_signal'] = signal_value
            metrics['macd_histogram'] = histogram
            
            # MACD signal
            if macd_value > signal_value:
                metrics['macd_trend'] = 'bullish'
            else:
                metrics['macd_trend'] = 'bearish'
        
        # Combine signals
        if 'rsi_signal' in metrics and 'macd_trend' in metrics:
            if metrics['rsi_signal'] == 'oversold' and metrics['macd_trend'] == 'bullish':
                signal = 'bullish'
                confidence = 75.0
            elif metrics['rsi_signal'] == 'overbought' and metrics['macd_trend'] == 'bearish':
                signal = 'bearish'
                confidence = 75.0
            elif metrics['macd_trend'] == 'bullish':
                signal = 'bullish'
                confidence = 60.0
            elif metrics['macd_trend'] == 'bearish':
                signal = 'bearish'
                confidence = 60.0
        elif 'macd_trend' in metrics:
            signal = metrics['macd_trend']
            confidence = 55.0
        elif 'rsi_signal' in metrics:
            if metrics['rsi_signal'] == 'oversold':
                signal = 'bullish'
                confidence = 60.0
            elif metrics['rsi_signal'] == 'overbought':
                signal = 'bearish'
                confidence = 60.0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'metrics': metrics
        }
    
    async def analyze(self, indicators: TechnicalIndicators) -> AnalysisResult:
        """Analyze stock using technical indicators"""
        try:
            # Validate input
            if not isinstance(indicators, TechnicalIndicators):
                raise ValueError("Expected TechnicalIndicators object")
            
            # Calculate trend signals
            trend_data = self.calculate_trend_signals(indicators)
            
            # Calculate momentum signals
            momentum_data = self.calculate_momentum_signals(indicators)
            
            # Combine signals
            if trend_data['signal'] == momentum_data['signal']:
                # Signals agree, use higher confidence
                signal = trend_data['signal']
                confidence = max(trend_data['confidence'], momentum_data['confidence'])
            else:
                # Signals disagree, weight by confidence
                trend_weight = trend_data['confidence'] / (trend_data['confidence'] + momentum_data['confidence'])
                momentum_weight = momentum_data['confidence'] / (trend_data['confidence'] + momentum_data['confidence'])
                
                if trend_weight > momentum_weight:
                    signal = trend_data['signal']
                    confidence = trend_data['confidence'] * 0.8  # Reduce confidence due to disagreement
                else:
                    signal = momentum_data['signal']
                    confidence = momentum_data['confidence'] * 0.8  # Reduce confidence due to disagreement
            
            # Generate reasoning
            reasoning = self._generate_reasoning(trend_data, momentum_data, indicators)
            
            # Prepare raw data for result
            raw_data = {
                'trend_analysis': trend_data,
                'momentum_analysis': momentum_data,
                'indicators': {
                    'sma_50': indicators.sma_50,
                    'sma_200': indicators.sma_200,
                    'rsi_14': indicators.rsi_14,
                    'macd': indicators.macd,
                    'vix': indicators.vix
                }
            }
            
            return AnalysisResult(
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                raw_data=raw_data
            )
            
        except Exception as e:
            logger.error(f"Error during technical analysis: {str(e)}")
            return AnalysisResult(
                signal='neutral',
                confidence=30.0,
                reasoning=f"Technical analysis encountered an error: {str(e)}",
                raw_data={'error': str(e)}
            )
    
    def _generate_reasoning(self, trend_data: Dict[str, Any], momentum_data: Dict[str, Any], 
                           indicators: TechnicalIndicators) -> str:
        """Generate reasoning text based on technical analysis"""
        parts = []
        
        # Add trend analysis
        if 'error' in trend_data['metrics']:
            parts.append(f"Trend Analysis: {trend_data['metrics']['error']}")
        else:
            trend_str = f"Trend Analysis: The stock is showing {trend_data['strength']} {trend_data['signal']} trend signals"
            
            # Add details about moving averages if available
            metrics = trend_data['metrics']
            if 'price_vs_sma50' in metrics:
                pct = metrics['price_vs_sma50'] * 100
                trend_str += f", with price {abs(pct):.1f}% {'above' if pct > 0 else 'below'} the 50-day moving average"
            
            if 'price_vs_sma200' in metrics:
                pct = metrics['price_vs_sma200'] * 100
                trend_str += f" and {abs(pct):.1f}% {'above' if pct > 0 else 'below'} the 200-day moving average"
            
            if 'golden_cross' in metrics:
                if metrics['golden_cross']:
                    trend_str += ". The 50-day moving average is above the 200-day moving average, indicating a potential long-term uptrend"
                else:
                    trend_str += ". The 50-day moving average is below the 200-day moving average, indicating a potential long-term downtrend"
            
            parts.append(trend_str + ".")
        
        # Add momentum analysis
        if 'error' in momentum_data['metrics']:
            parts.append(f"Momentum Analysis: {momentum_data['metrics']['error']}")
        else:
            momentum_str = f"Momentum Analysis: The stock is showing {momentum_data['signal']} momentum signals"
            
            # Add RSI details if available
            metrics = momentum_data['metrics']
            if 'rsi' in metrics:
                momentum_str += f", with RSI at {metrics['rsi']:.1f}"
                if metrics.get('rsi_signal') == 'overbought':
                    momentum_str += " (overbought territory)"
                elif metrics.get('rsi_signal') == 'oversold':
                    momentum_str += " (oversold territory)"
            
            # Add MACD details if available
            if 'macd_trend' in metrics:
                momentum_str += f". MACD indicates a {metrics['macd_trend']} momentum"
            
            parts.append(momentum_str + ".")
        
        # Add volatility context if available
        if indicators.vix:
            vix_str = f"Market Volatility: VIX is at {indicators.vix:.1f}, indicating "
            if indicators.vix > 30:
                vix_str += "high market volatility and potential for larger price swings"
            elif indicators.vix > 20:
                vix_str += "moderate market volatility"
            else:
                vix_str += "relatively low market volatility"
            
            parts.append(vix_str + ".")
        
        # Combine all parts
        return "\n\n".join(parts)

    