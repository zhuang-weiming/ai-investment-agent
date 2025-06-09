from typing import Dict, Any, List
import asyncio
import logging
from src.data.stock_data_collector import StockDataCollector
from src.agents.peter_lynch_agent import PeterLynchAgent
from src.agents.warren_buffett_agent import warren_buffett_agent
from src.agents.technical_agent import technical_analyst_agent

logger = logging.getLogger(__name__)

class StockAnalysisOrchestrator:
    """Orchestrates the stock analysis process using multiple strategies"""
    
    def __init__(self):
        self.data_collector = StockDataCollector()
        self.strategies = {
            'peter_lynch': PeterLynchAgent(),
            'warren_buffett': lambda data: warren_buffett_agent({'data': {**data, 'tickers': [data['symbol']]}, 'messages': []}),
            'technical': lambda data: technical_analyst_agent({'data': {**data, 'tickers': [data['symbol']]}, 'messages': []})
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
            if name == 'peter_lynch':
                result = await asyncio.to_thread(strategy.analyze, data['symbol'])
                return {
                    'signal': result.get('signal', 'neutral'),
                    'confidence': result.get('confidence', 50),
                    'reasoning': result.get('reasoning', ''),
                    'raw_data': result
                }
            else:
                result = await asyncio.to_thread(strategy, data)
                # Extract the first ticker's result for compatibility
                signals = result['data']['analyst_signals'][f'{name}_agent' if name == 'technical' else 'warren_buffett_agent']
                ticker = data['symbol'] if 'symbol' in data else next(iter(signals))
                ticker_result = signals[ticker]
                return {
                    'signal': ticker_result.get('signal', 'neutral'),
                    'confidence': ticker_result.get('confidence', 50),
                    'reasoning': ticker_result.get('reasoning', ''),
                    'raw_data': ticker_result
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
