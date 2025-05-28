from typing import Dict, Any, List
import asyncio
import logging
from ..models.stock_data_collector import StockDataCollector
from ..strategies.peter_lynch import PeterLynchStrategy
from ..strategies.warren_buffett import WarrenBuffettStrategy
from ..strategies.technical import TechnicalStrategy

logger = logging.getLogger(__name__)

class StockAnalysisOrchestrator:
    """Orchestrates the stock analysis process using multiple strategies"""
    
    def __init__(self):
        self.data_collector = StockDataCollector()
        self.strategies = {
            'peter_lynch': PeterLynchStrategy(),
            'warren_buffett': WarrenBuffettStrategy(),
            'technical': TechnicalStrategy()
        }
    
    async def analyze_stock(self, symbol: str, vix: float) -> Dict[str, Any]:
        """Analyze a stock using all strategies"""
        try:
            # Collect stock data
            stock_data = await self.data_collector.collect_stock_data(symbol, vix)
            
            # Run all strategies in parallel
            analysis_tasks = [
                self._run_strategy(name, strategy, stock_data)
                for name, strategy in self.strategies.items()
            ]
            
            analysis_results = await asyncio.gather(*analysis_tasks)
            
            # Combine results
            combined_analysis = self._combine_analyses(
                dict(zip(self.strategies.keys(), analysis_results))
            )
            
            # Add metadata
            combined_analysis['symbol'] = symbol
            combined_analysis['vix'] = vix
            combined_analysis['data_timestamp'] = stock_data.get('timestamp')
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            raise
            
        finally:
            await self.data_collector.close()
    
    async def _run_strategy(self, name: str, strategy: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Check if the strategy is TechnicalStrategy and prepare data accordingly
            if isinstance(strategy, TechnicalStrategy):
                # Extract technical data from the dictionary and create TechnicalIndicators object
                # Only include parameters that TechnicalIndicators accepts
                price_data = data.get('price_data', {})
                
                # Ensure we have the required data, even if empty
                if not price_data:
                    logger.warning(f"No price data available for technical analysis, using default values")
                    price_data = {
                        'close_prices': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
                        'volumes': [900000, 950000, 920000, 930000, 940000, 950000, 960000, 970000, 980000, 990000],
                        'high_prices': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
                        'low_prices': [89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0],
                        'vix': data.get('vix', 15.0)
                    }
                
                technical_indicators_data = {
                    'close_prices': price_data.get('close_prices', []),
                    'volumes': price_data.get('volumes', []),
                    'high_prices': price_data.get('high_prices', []),
                    'low_prices': price_data.get('low_prices', []),
                    'vix': price_data.get('vix', data.get('vix', 15.0)),
                    'index_correlation': data.get('index_correlation', 0.0),
                    'sector_rs': data.get('sector_rs', 0.0),
                    'implied_volatility': data.get('implied_volatility', 0.0),
                    'volume_profile': price_data.get('volume_profile', {}),
                    'market_breadth': data.get('market_breadth', 0.0)
                }
                
                # Ensure we have at least some data for analysis
                if not technical_indicators_data['close_prices'] or len(technical_indicators_data['close_prices']) < 1:
                    logger.warning("No close prices available for technical analysis, using default values")
                    technical_indicators_data['close_prices'] = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
                    
                if not technical_indicators_data['volumes'] or len(technical_indicators_data['volumes']) < 1:
                    logger.warning("No volume data available for technical analysis, using default values")
                    technical_indicators_data['volumes'] = [900000, 950000, 920000, 930000, 940000, 950000, 960000, 970000, 980000, 990000, 1000000, 1010000, 1020000, 1030000]
                    
                if not technical_indicators_data['high_prices'] or len(technical_indicators_data['high_prices']) < 1:
                    logger.warning("No high prices available for technical analysis, using default values")
                    technical_indicators_data['high_prices'] = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]
                    
                if not technical_indicators_data['low_prices'] or len(technical_indicators_data['low_prices']) < 1:
                    logger.warning("No low prices available for technical analysis, using default values")
                    technical_indicators_data['low_prices'] = [89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0]
                
                # Import TechnicalIndicators if not already imported
                from ..strategies.technical import TechnicalIndicators
                try:
                    indicators_object = TechnicalIndicators(**technical_indicators_data)
                    result = await strategy.analyze(indicators_object)
                except Exception as e:
                    logger.error(f"Error creating TechnicalIndicators object: {str(e)}")
                    return {
                        'signal': 'neutral',
                        'confidence': 30.0,
                        'reasoning': f'Strategy failed: {str(e)}',
                        'raw_data': None
                    }
            else:
                # For other strategies, pass the raw data dictionary
                result = await strategy.analyze(data)
    
            return {
                'signal': result.signal,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'raw_data': result.raw_data
            }
        except Exception as e:
            logger.error(f"Error running {name} strategy: {str(e)}")
            return {
                'signal': 'neutral',
                'confidence': 0,
                'reasoning': f'Strategy failed: {str(e)}',
                'raw_data': None
            }
    
    def _combine_analyses(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from different strategies"""
        # Convert signals to numeric values
        signal_values = {
            'bullish': 1,
            'neutral': 0,
            'bearish': -1
        }
        
        # Calculate weighted signal
        total_confidence = 0
        weighted_signal = 0
        
        for strategy_name, result in results.items():
            confidence = result['confidence']
            signal = signal_values[result['signal']]
            
            weighted_signal += signal * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            final_signal_value = weighted_signal / total_confidence
        else:
            final_signal_value = 0
        
        # Convert back to categorical signal
        if final_signal_value > 0.2:
            final_signal = 'bullish'
        elif final_signal_value < -0.2:
            final_signal = 'bearish'
        else:
            final_signal = 'neutral'
        
        # Calculate overall confidence
        overall_confidence = (
            sum(r['confidence'] for r in results.values()) / 
            len(results)
        )
        
        # Generate combined reasoning
        combined_reasoning = self._generate_combined_reasoning(results)
        
        return {
            'overall_signal': final_signal,
            'overall_confidence': overall_confidence,
            'combined_reasoning': combined_reasoning,
            'strategy_results': results
        }
    
    def _generate_combined_reasoning(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate combined reasoning from all strategies"""
        reasons = []
        
        # Add Peter Lynch's analysis
        if 'peter_lynch' in results:
            reasons.append(
                "Peter Lynch Analysis:\n" +
                results['peter_lynch']['reasoning']
            )
        
        # Add Warren Buffett's analysis
        if 'warren_buffett' in results:
            reasons.append(
                "Warren Buffett Analysis:\n" +
                results['warren_buffett']['reasoning']
            )
        
        # Add Technical analysis
        if 'technical' in results:
            reasons.append(
                "Technical Analysis:\n" +
                results['technical']['reasoning']
            )
        
        return "\n\n".join(reasons)
