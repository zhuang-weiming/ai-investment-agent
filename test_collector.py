import sys
import asyncio
sys.path.append('/Users/weimingzhuang/Documents/source_code/ai-investment-agent')
from src.data.stock_data_collector import StockDataCollector
from src.data.eastmoney_collector import EastmoneyCollector

async def test_stock_data_collector():
    collector = StockDataCollector()
    print('Testing StockDataCollector with EastmoneyCollector:')
    
    # Test direct methods first
    print('\nTesting direct methods:')
    news = collector.get_news('000333.SZ')
    print(f'News items: {len(news)}')
    insider_trades = collector.get_insider_trades('000333.SZ')
    print(f'Insider trades: {len(insider_trades)}')
    market_cap = collector.get_market_cap('000333.SZ')
    print(f'Market cap: {market_cap}')
    
    # Test async collect_stock_data method
    print('\nTesting async collect_stock_data method:')
    data = await collector.collect_stock_data('000333.SZ')
    print('Collected data keys:', list(data.keys()))
    print('Financials available:', 'fundamental_data' in data)
    print('Market data available:', 'market_data' in data)
    print('Price data available:', 'price_data' in data)

async def test_eastmoney_collector():
    collector = EastmoneyCollector()
    print('\nTesting EastmoneyCollector directly:')
    
    financials = collector.get_financials('000333.SZ')
    print(f'Got {len(financials)} financial records')
    
    news = collector.get_news('000333.SZ')
    print(f'Got {len(news)} news items')
    
    trades = collector.get_insider_trades('000333.SZ')
    print(f'Got {len(trades)} insider trades')
    
    cap = collector.get_market_cap('000333.SZ')
    print(f'Market cap: {cap}')

async def main():
    await test_eastmoney_collector()
    await test_stock_data_collector()

if __name__ == "__main__":
    asyncio.run(main())