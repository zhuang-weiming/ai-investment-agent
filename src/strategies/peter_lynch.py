from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, AnalysisResult
import logging
import pandas as pd
import re
import numpy as np

logger = logging.getLogger(__name__)

class PeterLynchStrategy(BaseStrategy):
    """Implements Peter Lynch's investment strategy:
      - Invest in what you know (clear, understandable businesses)
      - Growth at a Reasonable Price (GARP), emphasizing the PEG ratio
      - Look for consistent revenue & EPS increases and manageable debt
      - Be alert for potential "ten-baggers" (high-growth opportunities)
      - Avoid overly complex or highly leveraged businesses
      - Use news sentiment and insider trades for secondary inputs
      - If fundamentals strongly align with GARP, be more aggressive
    """
    
    def __init__(self):
        self.required_metrics = [
            'revenue_growth',
            'eps_growth',
            'net_income',
            'operating_margin', 
            'free_cash_flow',
            'total_debt',
            'shareholders_equity',
            'market_cap',
            'pe_ratio',
            'peg_ratio',
            'news_sentiment',
            'insider_buys',
            'insider_sells'
        ]

    def analyze_growth(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate growth based on revenue and EPS trends"""
        if not metrics:
            return {"score": 0, "details": "Insufficient financial data for growth analysis"}

        details = []
        raw_score = 0  # We'll sum up points, then scale to 0–10 

        # 1) Revenue Growth
        rev_growth = metrics.get('revenue_growth')
        if rev_growth is not None:
            if rev_growth > 0.25:
                raw_score += 3
                details.append(f"Strong revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.10:
                raw_score += 2
                details.append(f"Moderate revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.02:
                raw_score += 1
                details.append(f"Slight revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Flat or negative revenue growth: {rev_growth:.1%}")
        else:
            details.append("No revenue growth data available.")

        # 2) EPS Growth
        eps_growth = metrics.get('eps_growth')
        if eps_growth is not None:
            if eps_growth > 0.25:
                raw_score += 3
                details.append(f"Strong EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                raw_score += 2
                details.append(f"Moderate EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.02:
                raw_score += 1
                details.append(f"Slight EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative EPS growth: {eps_growth:.1%}")
        else:
            details.append("No EPS growth data available.")

        # raw_score can be up to 6 => scale to 0-10
        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    def analyze_fundamentals(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate basic fundamentals: D/E ratio, margins, FCF"""
        if not metrics:
            return {"score": 0, "details": "Insufficient fundamentals data"}

        details = []
        raw_score = 0  # We'll accumulate up to 6 points

        # 1) Debt-to-Equity
        if metrics.get('shareholders_equity') and metrics.get('total_debt'):
            equity = metrics['shareholders_equity']
            debt = metrics['total_debt']
            if equity > 0:
                de_ratio = debt / equity
                if de_ratio < 0.5:
                    raw_score += 2
                    details.append(f"Low debt-to-equity: {de_ratio:.2f}")
                elif de_ratio < 1.0:
                    raw_score += 1
                    details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
                else:
                    details.append(f"High debt-to-equity: {de_ratio:.2f}")
            else:
                details.append("Negative equity")
        else:
            details.append("No debt/equity data available")

        # 2) Operating Margin
        if metrics.get('operating_margin') is not None:
            margin = metrics['operating_margin']
            if margin > 0.20:
                raw_score += 2
                details.append(f"Strong operating margin: {margin:.1%}")
            elif margin > 0.10:
                raw_score += 1
                details.append(f"Moderate operating margin: {margin:.1%}")
            else:
                details.append(f"Low operating margin: {margin:.1%}")
        else:
            details.append("No operating margin data available")

        # 3) Free Cash Flow
        if metrics.get('free_cash_flow') is not None:
            fcf = metrics['free_cash_flow']
            if fcf > 0:
                raw_score += 2
                details.append(f"Positive free cash flow: {fcf:,.0f}")
            else:
                details.append(f"Negative free cash flow: {fcf:,.0f}")
        else:
            details.append("No free cash flow data available")

        # Scale to 0-10
        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    def analyze_valuation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Growth at a Reasonable Price (GARP)"""
        if not metrics:
            return {"score": 0, "details": "Insufficient valuation data"}

        details = []
        raw_score = 0

        # Check PEG ratio first
        peg_ratio = metrics.get('peg_ratio')
        if peg_ratio is not None:
            details.append(f"PEG ratio: {peg_ratio:.2f}")
            if peg_ratio < 1:
                raw_score += 3
            elif peg_ratio < 2:
                raw_score += 2
            elif peg_ratio < 3:
                raw_score += 1

        # Check P/E approximation if we have market cap and net income
        if metrics.get('market_cap') and metrics.get('net_income'):
            if metrics['net_income'] > 0:
                pe_ratio = metrics['market_cap'] / metrics['net_income']
                details.append(f"Estimated P/E: {pe_ratio:.2f}")
                if pe_ratio < 15:
                    raw_score += 2
                elif pe_ratio < 25:
                    raw_score += 1

        if not details:
            return {"score": 0, "details": "No valuation metrics available"}

        final_score = min(10, (raw_score / 5) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    def analyze_sentiment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment based on news and insider activity"""
        score = 5  # Default neutral score
        details = []

        # News sentiment (scale of 0-10 => normalize to 0-10)
        sentiment = metrics.get('news_sentiment', 5.0)
        if sentiment is not None:
            score = sentiment

        # Insider activity
        buys = metrics.get('insider_buys', 0)
        sells = metrics.get('insider_sells', 0)
        total = buys + sells

        if total > 0:
            buy_ratio = buys / total
            if buy_ratio > 0.7:
                score += 3  # Heavy buying is very positive
                details.append(f"Heavy insider buying: {buys} buys vs {sells} sells")
            elif buy_ratio > 0.4:
                score += 1  # More buys than sells is mildly positive
                details.append(f"Moderate insider buying: {buys} buys vs {sells} sells")
            else:
                score -= 1  # More sells than buys is negative
                details.append(f"Mostly insider selling: {buys} buys vs {sells} sells")
        else:
            details.append("No significant insider transactions")

        final_score = min(10, max(0, score))  # Ensure score is between 0-10
        if not details:
            details = ["Limited sentiment data available"]

        return {"score": final_score, "details": "; ".join(details)}

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data has minimum required metrics"""
        try:
            # Check for presence of key metrics
            required_growth_metrics = ['revenue_growth', 'eps_growth']
            
            # Growth metrics - need at least one
            has_growth = any(data.get(m) is not None for m in required_growth_metrics)
            if not has_growth:
                logger.warning("Missing all growth metrics")
                return False

            # If we have growth metrics, consider data valid
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """Analyze stock using Peter Lynch's principles"""
        try:
            # Extract metrics from data
            metrics = {k: data.get(k) for k in self.required_metrics}
            
            # Ensure metrics dictionary is not None and has default values for missing metrics
            if metrics is None:
                metrics = {}
            
            # Set default values for critical metrics if they're missing
            for key in self.required_metrics:
                if key not in metrics or metrics[key] is None:
                    if key in ['revenue_growth', 'eps_growth', 'operating_margin', 'free_cash_flow']:
                        metrics[key] = 0.0
                    elif key in ['pe_ratio']:
                        metrics[key] = 15.0
                    elif key in ['peg_ratio']:
                        metrics[key] = 1.5
                    elif key in ['market_cap', 'net_income', 'total_debt', 'shareholders_equity']:
                        metrics[key] = 1000000.0  # Default to 1M
                    elif key in ['news_sentiment']:
                        metrics[key] = 5.0
                    elif key in ['insider_buys', 'insider_sells']:
                        metrics[key] = 0
            
            # Perform sub-analyses with Lynch's typical weightings:
            # 30% Growth, 25% Valuation, 20% Fundamentals, 25% Sentiment
            try:
                growth = self.analyze_growth(metrics)
            except Exception as e:
                logger.error(f"Error in growth analysis: {str(e)}")
                growth = {"score": 0.0, "details": f"Growth analysis error: {str(e)}"}
                
            try:
                valuation = self.analyze_valuation(metrics)
            except Exception as e:
                logger.error(f"Error in valuation analysis: {str(e)}")
                valuation = {"score": 0.0, "details": f"Valuation analysis error: {str(e)}"}
                
            try:
                fundamentals = self.analyze_fundamentals(metrics)
            except Exception as e:
                logger.error(f"Error in fundamentals analysis: {str(e)}")
                fundamentals = {"score": 0.0, "details": f"Fundamentals analysis error: {str(e)}"}
                
            try:
                sentiment = self.analyze_sentiment(metrics)
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                sentiment = {"score": 0.0, "details": f"Sentiment analysis error: {str(e)}"}
            
            # Ensure all scores are numeric and not None
            growth_score = growth.get("score", 0.0) if isinstance(growth, dict) else 0.0
            valuation_score = valuation.get("score", 0.0) if isinstance(valuation, dict) else 0.0
            fundamentals_score = fundamentals.get("score", 0.0) if isinstance(fundamentals, dict) else 0.0
            sentiment_score = sentiment.get("score", 0.0) if isinstance(sentiment, dict) else 0.0
            
            # Calculate weighted total score (0-10 scale)
            total_score = (
                growth_score * 0.30 +
                valuation_score * 0.25 +
                fundamentals_score * 0.20 +
                sentiment_score * 0.25
            )

            # Map score to signal and calculate confidence
            if total_score >= 7.5:
                signal = "bullish"
                confidence = total_score * 10  # Scale 0-10 to 0-100
            elif total_score <= 4.5:
                signal = "bearish"
                confidence = max(60, (10 - total_score) * 10)  # Higher confidence for lower scores
            else:
                signal = "neutral"
                confidence = total_score * 10
            
            # Generate detailed reasoning
            try:
                reasoning = self._generate_lynch_reasoning(
                    metrics,
                    growth,
                    valuation,
                    fundamentals,
                    sentiment,
                    total_score
                )
            except Exception as e:
                logger.error(f"Error generating reasoning: {str(e)}")
                reasoning = f"Analysis completed with score {total_score:.1f}, but error generating detailed reasoning: {str(e)}"
            
            return AnalysisResult(
                signal=signal,
                confidence=min(100, confidence),  # Cap at 100
                reasoning=reasoning,
                raw_data=data
            )
            
        except Exception as e:
            logger.error(f"Error in Peter Lynch analysis: {str(e)}")
            return AnalysisResult(
                signal="neutral",
                confidence=30,
                reasoning=f"Analysis error: {str(e)}",
                raw_data=data
            )

    def _generate_lynch_reasoning(self, metrics: Dict[str, Any],
                              growth: Dict[str, Any],
                              valuation: Dict[str, Any],
                              fundamentals: Dict[str, Any],
                              sentiment: Dict[str, Any],
                              total_score: float) -> str:
        """Generate Peter Lynch-style analysis reasoning"""
        parts = []
        
        # Growth potential (including ten-bagger check)
        revenue_growth = metrics.get('revenue_growth', 0)
        eps_growth = metrics.get('eps_growth', 0)
        
        # Ensure values are not None
        if revenue_growth is None:
            revenue_growth = 0
        if eps_growth is None:
            eps_growth = 0
            
        if revenue_growth > 0.25 and eps_growth > 0.25:
            parts.append(
                "I see potential ten-bagger characteristics here with outstanding revenue "
                f"growth of {revenue_growth:.1%} and EPS growth of {eps_growth:.1%}."
            )
        
        # GARP analysis with PEG focus
        peg_ratio = metrics.get('peg_ratio')
        if peg_ratio is not None:
            peg = peg_ratio
            parts.append(
                f"PEG ratio at {peg:.2f} is {'fantastic' if peg < 1 else 'reasonable' if peg < 2 else 'concerning'} - "
                f"{'this is the kind of growth-value combination I look for!' if peg < 1 else 'a bit pricey for my taste.' if peg > 2 else 'fairly valued.'}"
            )

        # Business quality
        operating_margin = metrics.get('operating_margin')
        if operating_margin is not None:
            margin = operating_margin
            parts.append(
                f"Operating margin of {margin:.1%} shows {'excellent' if margin > 0.2 else 'decent' if margin > 0.1 else 'concerning'} business economics."
            )
            
        # Balance sheet strength
        total_debt = metrics.get('total_debt')
        shareholders_equity = metrics.get('shareholders_equity')
        if total_debt is not None and shareholders_equity is not None and shareholders_equity > 0:
            de_ratio = total_debt / shareholders_equity
            parts.append(
                f"Debt-to-equity of {de_ratio:.1f}x is {'very manageable' if de_ratio < 0.5 else 'somewhat concerning' if de_ratio > 1 else 'acceptable'}."
            )

        # Sentiment indicators
        if sentiment.get('details'):
            parts.append(sentiment['details'])

        # Conclusion with Lynch-style observation
        if total_score >= 7.5:
            parts.append(
                "This is exactly the kind of company I'd invest in - "
                "strong growth at a reasonable price with solid fundamentals."
            )
        elif total_score <= 4.5:
            parts.append(
                "I'm steering clear of this one - "
                "the numbers don't support the story, and that's a red flag for me."
            )
        else:
            parts.append(
                "I'm on the fence here - "
                "some things to like, but not quite compelling enough for my style."
            )

        parts.append("/no_think")
        return " ".join(parts)

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = 0.0) -> Any:
        """Safely get a value from data dictionary with default"""
        if not isinstance(data, dict):
            return default
        
        value = data.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        
        # For metrics that should be percentages, ensure they're in reasonable range
        if key in ['pe_ratio', 'industry_pe', 'earnings_growth', 'revenue_growth', 'eps_growth']:
            try:
                value = float(value)
                if value < -100:  # Cap negative growth at -100%
                    return -100.0
                elif value > 1000:  # Cap extremely high values
                    return 1000.0
                return value
            except:
                return default
        
        return value
    
    def calculate_growth_score(self, data: Dict[str, Any]) -> float:
        """Calculate growth score with improved error handling"""
        try:
            # Extract relevant data
            pe_ratio = self._safe_get(data, 'pe_ratio')
            industry_pe = self._safe_get(data, 'industry_pe')
            earnings_growth = self._safe_get(data, 'earnings_growth')
            revenue_growth = self._safe_get(data, 'revenue_growth')
            eps_growth = self._safe_get(data, 'eps_growth')
            
            # Calculate growth score
            score = 0.0
            
            # P/E ratio relative to industry
            if industry_pe != 0 and pe_ratio != 0:
                if pe_ratio < industry_pe:
                    score += 40 * (1 - (pe_ratio / industry_pe))  # Reward lower P/E
                else:
                    score += 40 * (industry_pe / pe_ratio)  # Penalize higher P/E
            else:
                score += 20  # Neutral score if we can't compare P/E ratios
            
            # Earnings growth
            if earnings_growth > 0:
                score += min(earnings_growth * 2, 30)  # Cap at 30 points
            elif earnings_growth < 0:
                score += max(0, 20 + earnings_growth)  # Reduce score for negative growth
            else:
                score += 10  # Neutral score for 0 growth
            
            # Revenue growth
            if revenue_growth > 0:
                score += min(revenue_growth, 20)  # Cap at 20 points
            elif revenue_growth < 0:
                score += max(0, 10 + revenue_growth)  # Reduce score for negative growth
            else:
                score += 10  # Neutral score for 0 growth
            
            # EPS growth
            if eps_growth > 0:
                score += min(eps_growth, 10)  # Cap at 10 points
            elif eps_growth < 0:
                score += max(0, 5 + eps_growth)  # Reduce score for negative growth
            else:
                score += 5  # Neutral score for 0 growth
            
            # Cap the final score between 0 and 100
            return max(0.0, min(score, 100.0))
        except Exception as e:
            logger.error(f"Error calculating growth score: {str(e)}")
            return 50.0  # Return neutral score on error
