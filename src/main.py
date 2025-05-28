import asyncio
import logging
import json
from datetime import datetime
from typing import List
import argparse
from .analysis.analysis_orchestrator import StockAnalysisOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def analyze_stocks(symbols: List[str], vix: float) -> List[dict]:
    """Analyze multiple stocks"""
    orchestrator = StockAnalysisOrchestrator()
    results = []
    
    for symbol in symbols:
        try:
            logger.info(f"Analyzing {symbol}...")
            analysis = await orchestrator.analyze_stock(symbol, vix)
            results.append(analysis)
            logger.info(f"Completed analysis for {symbol}")
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {str(e)}")
            continue
    
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI Stock Analysis Tool')
    parser.add_argument(
        'symbols',
        type=str,
        nargs='+',
        help='Stock symbols to analyze (e.g., AAPL MSFT GOOGL)'
    )
    parser.add_argument(
        '--vix',
        type=float,
        default=15.0,
        help='Current VIX value (default: 15.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for JSON results'
    )
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Run analysis
        results = await analyze_stocks(args.symbols, args.vix)
        
        # Format results
        output = {
            'timestamp': datetime.now().isoformat(),
            'vix': args.vix,
            'analyses': results
        }
        
        # Print to console
        print("\nAnalysis Results:")
        print("-" * 80)
        for analysis in results:
            symbol = analysis['symbol']
            signal = analysis['overall_signal']
            confidence = analysis['overall_confidence']
            
            print(f"\n{symbol} Analysis:")
            print(f"Signal: {signal.upper()}")
            print(f"Confidence: {confidence:.1f}%")
            print("\nReasoning:")
            print(analysis['combined_reasoning'])
            print("-" * 80)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
