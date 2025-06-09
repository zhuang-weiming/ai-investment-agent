from typing import Dict, List, Any
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
1. Key Metrics Analysis:
   - Current market metrics (price, market cap)
   - Basic technical indicators (RSI, MACD, SMA)

2. Technical Analysis:
   - Price trends (SMA 50/200)
   - Momentum indicators (RSI, MACD)
   - Volume analysis

3. Investment Recommendation:
   - Risk assessment
   - Growth potential
   - Valuation analysis
   - Time horizon considerations

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
                system_prompt="You are an expert financial analyst AI assistant.",
                context=context,
                question=question
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return f"Error analyzing stock data: {str(e)}"
    
    def build_peter_lynch_prompt(self, financials: List[Dict], news: List[Dict], 
                              insider: List[Dict], market_cap: float) -> str:
        """Build prompt for Peter Lynch style analysis
        
        Focuses on:
        - Company growth rate
        - PEG ratio
        - Future earnings potential
        - Insider activity
        - Market sentiment
        """
        system_prompt = """You are Peter Lynch, the legendary fund manager known for investing in companies 
        with good stories and strong fundamentals. Analyze this stock using your investment philosophy:

        1. Growth at a Reasonable Price (GARP)
        2. Focus on companies you understand
        3. Look for fast-growing companies in slow-growing industries
        4. Strong balance sheets
        5. Consider insider buying/selling
        """

        context = {
            "financials": financials,
            "news": news,
            "insider_trades": insider,
            "market_cap": market_cap
        }

        question = """Based on the available data, provide an investment analysis considering:
        1. Growth rate vs. industry average
        2. PEG ratio analysis
        3. Management quality and insider activity
        4. Company narrative and market position
        5. Overall investment recommendation
        """

        chain = self.create_analysis_chain(system_prompt)
        return chain.run(
            system_prompt=system_prompt,
            context=json.dumps(context, indent=2),
            question=question
        )

    def build_technical_prompt(self, technical_data: Dict) -> str:
        """Build prompt for technical analysis"""
        system_prompt = """You are a technical analyst focusing on price action, trends, 
        and momentum indicators. Analyze the technical signals in this data."""

        context = {"technical_data": technical_data}

        question = """Based on the technical indicators, provide:
        1. Trend analysis (using moving averages)
        2. Momentum analysis (RSI, MACD)
        3. Volume analysis
        4. Support/resistance levels
        5. Trading recommendation
        """

        chain = self.create_analysis_chain(system_prompt)
        return chain.run(
            system_prompt=system_prompt,
            context=json.dumps(context, indent=2),
            question=question
        )

    def build_warren_buffett_prompt(self, financials: List[Dict], moat_analysis: Dict,
                                  competitive_analysis: Dict) -> str:
        """Build prompt for Warren Buffett style analysis"""
        system_prompt = """You are Warren Buffett, focusing on companies with strong competitive 
        advantages and long-term value. Analyze this company using your investment principles:

        1. Strong economic moat
        2. Consistent earnings power
        3. High return on equity
        4. Low debt
        5. Simple, understandable business
        """

        context = {
            "financials": financials,
            "moat_analysis": moat_analysis,
            "competitive_analysis": competitive_analysis
        }

        question = """Based on the available data, provide an investment analysis considering:
        1. Economic moat strength
        2. Financial strength and stability
        3. Management quality and capital allocation
        4. Intrinsic value estimation
        5. Long-term investment potential
        """

        chain = self.create_analysis_chain(system_prompt)
        return chain.run(
            system_prompt=system_prompt,
            context=json.dumps(context, indent=2),
            question=question
        )
        
    def analyze(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response to extract signal, confidence, and reasoning"""
        try:
            # Default values
            signal = "neutral"
            confidence = 50
            reasoning = llm_response
            
            # Look for signal keywords
            lower_response = llm_response.lower()
            if "bullish" in lower_response:
                signal = "bullish"
            elif "bearish" in lower_response:
                signal = "bearish"
            
            # Try to extract confidence
            import re
            confidence_patterns = [
                r"confidence[:\s]*(\d+)", 
                r"confidence[:\s]*([\d\.]+)%",
                r"(\d+)%\s*confidence"
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, lower_response)
                if match:
                    try:
                        confidence = float(match.group(1))
                        # If confidence was expressed as a percentage out of 100
                        if confidence > 0 and confidence <= 100:
                            break
                    except ValueError:
                        pass
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Error parsing response: {str(e)}"
            }
