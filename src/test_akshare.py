"""Test script for AKShare data collection and analysis"""
import asyncio
import logging
from src.models.akshare_collector import AKShareCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Initialize collector
    collector = AKShareCollector()
    
    try:
        # Test with a few different stocks
        symbols = ["000333", "600519", "000858"]  # Midea Group, Kweichow Moutai, Wuliangye
        
        for symbol in symbols:
            logger.info(f"\nCollecting data for {symbol}")
            stock_data = await collector.collect_stock_data(symbol)
            
            # Validate market data
            market_data = stock_data['market_data']
            logger.info("\nMarket Data:")
            logger.info(f"Last Price: {market_data['last_price']}")
            logger.info(f"Volume: {market_data['volume']}")
            logger.info(f"Change %: {market_data['change_pct']}%")
            
            # Validate fundamental data
            fundamental_data = stock_data['fundamental_data']
            logger.info("\nFundamental Data:")
            logger.info(f"Market Cap: {fundamental_data['market_cap']}")
            logger.info(f"P/E Ratio: {fundamental_data['pe_ratio']}")
            logger.info(f"ROE: {fundamental_data['roe']}%")
            logger.info(f"Revenue Growth: {fundamental_data['revenue_growth']}%")
            
            # Validate technical data
            technical_data = stock_data['technical_data']
            logger.info("\nTechnical Data:")
            logger.info(f"RSI (14): {technical_data['rsi_14']}")
            logger.info(f"SMA 50: {technical_data['sma_50']}")
            logger.info(f"SMA 200: {technical_data['sma_200']}")
            logger.info(f"MACD: {technical_data['macd']}")
            
            # Add a separator between stocks
            logger.info("-" * 80)
            
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
