from typing import Dict, List
import json
import logging
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.config import MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self, model_name: str = MODEL_CONFIG["name"]):
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = Ollama(
            model=model_name,
            temperature=MODEL_CONFIG["temperature"],
            callback_manager=self.callback_manager
        )
        
    def create_analysis_chain(self, system_prompt: str) -> LLMChain:
        """Create an LLM chain with the given system prompt"""
        template = f"""{{system_prompt}}

You are analyzing financial data for stock market analysis. Given the following information:

Context:
{{context}}

Question:
{{question}}

Please provide a detailed analysis following these guidelines:
1. Start with a summary of key metrics (price, change, market cap)
2. Analyze the technical indicators and market sentiment
3. Review recent news and their impact
4. Assess risks and opportunities
5. Provide short-term and long-term predictions

Format your response in a clear, structured manner.

Analysis:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["system_prompt", "context", "question"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _clean_scraped_data(self, scraped_data: List[Dict]) -> str:
        """Clean and format scraped data for the LLM"""
        cleaned_data = []
        
        for data in scraped_data:
            if "error" in data:
                logger.warning(f"Skipping data with error: {data['error']}")
                continue
                
            source = data.get('url', 'Unknown source')
            cleaned_entry = f"Source: {source}\n"
            
            # Add price information if available
            if 'price' in data:
                cleaned_entry += f"Price: {data['price']}\n"
            if 'change' in data:
                cleaned_entry += f"Change: {data['change']}\n"
                
            # Add news if available
            if 'news' in data and isinstance(data['news'], list):
                cleaned_entry += "\nRecent News:\n"
                for news_item in data['news']:
                    if isinstance(news_item, dict):
                        title = news_item.get('title', '')
                        summary = news_item.get('summary', '')
                        if title or summary:
                            cleaned_entry += f"- {title}\n  {summary}\n"
                            
            # Add any additional data
            for key, value in data.items():
                if key not in ['url', 'price', 'change', 'news', 'error'] and value:
                    if isinstance(value, (str, int, float)):
                        cleaned_entry += f"{key}: {value}\n"
                        
            cleaned_data.append(cleaned_entry)
            
        return "\n\n".join(cleaned_data)
    
    def analyze_stock_data(self, chain: LLMChain, scraped_data: List[Dict], question: str) -> str:
        """Analyze stock data using the LLM chain"""
        try:
            # Clean and prepare the context
            context = self._clean_scraped_data(scraped_data)
            
            # Get analysis from LLM
            response = chain.run(
                system_prompt=chain.prompt.template,
                context=context,
                question=question
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return f"Error analyzing stock data: {str(e)}"
