from typing import List, Dict
import asyncio
import logging
from src.scrapers.web_scraper import WebScraper
from src.llm.stock_analyzer import StockAnalyzer
from src.config import MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalysisService:
    def __init__(self, urls: List[str], system_prompt: str, model_name: str = None):
        self.urls = urls
        self.system_prompt = system_prompt
        self.scraper = WebScraper()
        self.analyzer = StockAnalyzer(model_name=model_name or MODEL_CONFIG["name"])
        
    async def run_analysis(self, question: str) -> str:
        """Run the complete analysis pipeline"""
        try:
            logger.info("Starting web scraping...")
            # Scrape data from all URLs
            scraped_data = await self.scraper.scrape_multiple_urls(self.urls)
            
            # Check if we have any valid data
            valid_data = [data for data in scraped_data if "error" not in data]
            if not valid_data:
                raise ValueError("No valid data could be scraped from any of the sources")
                
            logger.info("Creating analysis chain...")
            # Create analysis chain
            chain = self.analyzer.create_analysis_chain(self.system_prompt)
            
            logger.info("Running analysis...")
            # Analyze the scraped data
            analysis = self.analyzer.analyze_stock_data(chain, scraped_data, question)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
            
        finally:
            # Ensure browser is closed
            await self.scraper.close()

async def analyze_stocks(urls: List[str], system_prompt: str, user_question: str) -> str:
    """Convenience function to run stock analysis"""
    service = StockAnalysisService(urls, system_prompt)
    return await service.run_analysis(user_question)
