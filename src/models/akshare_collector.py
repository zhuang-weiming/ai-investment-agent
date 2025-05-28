"""AKShare based stock data collector"""
from typing import Dict, List, Any, Optional, Union
import akshare as ak
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import time
from functools import wraps
from src.utils.validation import (
    validate_stock_symbol,
    validate_dataframe_columns,
    validate_quote_data,
    validate_metrics_data,
    validate_historical_data
)

logger = logging.getLogger(__name__)

# Required fields for data validation
REQUIRED_FINANCIAL_METRICS = [
    '市盈率', '市净率', '营业收入同比增长率', '基本每股收益同比增长率',
    '净资产收益率', '销售毛利率', '资产负债率'
]

REQUIRED_QUOTE_FIELDS = [
    '最新价', '成交量', '最高', '最低', '开盘', '昨收', '涨跌幅'
]

REQUIRED_TECHNICAL_METRICS = [
    'sma_50', 'sma_200', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volume_ma'
]

def retry_on_exception(retries=3, delay=1):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        sleep_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {str(e)}")
                        time.sleep(sleep_time)
            logger.error(f"All {retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator

class AKShareCollector:
    """Collects stock data from AKShare API"""
    
    # Constants with both Chinese and English field names
    FIELD_MAPPINGS = {
        'market_cap': ['总市值', 'market_cap', 'total_market_value', '市值'],
        'pe_ratio': ['市盈率', 'PE', 'pe_ratio', 'PE_TTM', '市盈率(动态)', '市盈率(静态)'],
        'pb_ratio': ['市净率', 'PB', 'pb_ratio', '市净率(动态)', '市净率(静态)'],
        'revenue_growth': ['营业收入同比增长率', 'revenue_yoy_growth', '营收增长率'],
        'eps_growth': ['基本每股收益同比增长率', 'eps_yoy_growth', '每股收益增长率'],
        'roe': ['净资产收益率', 'ROE', 'roe', '净资产收益率(加权)'],
        'profit_margin': ['销售毛利率', 'gross_margin', '毛利率'],
        'debt_ratio': ['资产负债率', 'debt_ratio', '负债率'],
        'net_income': ['净利润', 'net_income', '净利润(元)'],
        'depreciation': ['折旧和摊销', 'depreciation', '折旧'],
        'capex': ['资本支出', 'capital_expenditure', '资本开支'],
        'volume': ['成交量', 'volume', '成交量(手)'],
        'close': ['收盘', 'close', '收盘价'],
        'high': ['最高', 'high', '最高价'],
        'low': ['最低', 'low', '最低价']
    }

    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)

    def _get_field_value(self, data: Union[pd.DataFrame, pd.Series], field: str, default: Optional[float] = None) -> Optional[float]:
        """Get value from data using multiple possible field names with encoding handling and normalization"""
        possible_names = self.FIELD_MAPPINGS.get(field, [field])
        # Normalize all possible names
        normalized_names = [n.strip().lower() for n in possible_names]
        # Get all available columns/fields, normalized
        if isinstance(data, pd.DataFrame):
            available = [str(c).strip().lower() for c in data.columns]
        else:
            available = [str(i).strip().lower() for i in data.index]
        for name in normalized_names:
            if name in available:
                idx = available.index(name)
                col = data.columns[idx] if isinstance(data, pd.DataFrame) else data.index[idx]
                val = data[col].iloc[0] if isinstance(data, pd.DataFrame) else data[col]
                if not pd.isna(val):
                    try:
                        float_val = float(val)
                        if not np.isinf(float_val):
                            return float_val
                    except Exception:
                        continue
        # If not found, log available columns for debugging
        logger.warning(f"Field '{field}' not found. Available: {available}")
        return default

    def _safe_get_metric(self, data: pd.DataFrame, metric_key: str, default: Optional[float] = None) -> Optional[float]:
        """Safely extract a metric value with multiple fallback options"""
        try:
            possible_names = self.METRIC_MAPPINGS.get(metric_key, [metric_key])
            for name in possible_names:
                if name in data.columns:
                    value = data[name].iloc[0]
                    if not pd.isna(value):
                        try:
                            return float(value)
                        except:
                            continue
            return default
        except Exception as e:
            logger.warning(f"Error getting metric {metric_key}: {str(e)}")
            return default

    def _get_quote_value(self, quote: pd.Series, field: str, alternative_fields: List[str] = None) -> Optional[float]:
        """Safely extract value from quote data with fallback to alternative field names"""
        try:
            if field in quote:
                val = quote[field]
                return float(val) if not pd.isna(val) else None
            if alternative_fields:
                for alt_field in alternative_fields:
                    if alt_field in quote:
                        val = quote[alt_field]
                        return float(val) if not pd.isna(val) else None
            return None
        except:
            return None

    async def collect_stock_data(self, symbol: str, vix: Optional[float] = None) -> Dict[str, Any]:
        """Collect fundamental and technical data using AKShare"""
        try:
            # Validate stock symbol format first
            if not validate_stock_symbol(symbol):
                raise ValueError(f"Invalid stock symbol format: {symbol}")

            # Check cache first
            if symbol in self.cache:
                data, timestamp = self.cache[symbol]
                if datetime.now() - timestamp < self.cache_duration:
                    logger.info(f"Using cached data for {symbol}")
                    return data

            logger.info(f"Collecting fresh data for {symbol}")
            
            # Get basic stock info and industry data
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            industry = self._get_field_value(stock_info, '所属行业', 'Unknown')
            
            # Get industry metrics
            industry_metrics = self._get_industry_metrics(symbol)
            
            # Collect financial data from multiple sources
            financial_data = self._get_fundamental_indicators(symbol)
            
            # Get additional market data
            additional_data = self._get_additional_market_data(symbol)
            
            # Get realtime market data
            try:
                realtime_quotes = ak.stock_zh_a_spot_em()
                current_quote = realtime_quotes[realtime_quotes['代码'] == symbol].iloc[0]
            except Exception as e:
                logger.error(f"Error fetching quote data: {str(e)}")
                current_quote = pd.Series()

            # Get historical price data with fallback
            try:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                end_date = datetime.now().strftime('%Y%m%d')
                try:
                    hist_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                                 start_date=start_date,
                                                 end_date=end_date)
                except Exception as e:
                    logger.warning(f"Failed to get 1 year history, trying 100 days: {str(e)}")
                    start_date = (datetime.now() - timedelta(days=100)).strftime('%Y%m%d')
                    hist_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                                 start_date=start_date,
                                                 end_date=end_date)
            except Exception as e:
                logger.error(f"Error fetching historical data: {str(e)}")
                hist_data = pd.DataFrame()

            # Calculate all metrics
            metrics = self._calculate_metrics(symbol, hist_data, financial_data, current_quote)
            
            # Update metrics with industry data
            metrics.update(industry_metrics)
            
            # Calculate additional derived metrics
            try:
                # Market share calculation
                if metrics.get('revenue') and industry_metrics.get('industry_revenue'):
                    metrics['market_share'] = self._calculate_market_share(
                        metrics['revenue'], 
                        industry_metrics['industry_revenue']
                    )
                
                # Brand value calculation
                metrics['brand_value'] = self._calculate_brand_value(metrics, metrics.get('market_share', 0.0))
                
                # Regulatory risk calculation
                metrics['regulatory_risk'] = self._calculate_regulatory_risk(industry)
                
                # Capital efficiency metrics
                capital_metrics = self._calculate_capital_efficiency(financial_data)
                metrics.update(capital_metrics)
                
                # Additional market data
                metrics.update(additional_data)
                
            except Exception as e:
                logger.error(f"Error calculating derived metrics: {str(e)}")

            # Construct final response
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'industry': industry,
                'market_data': {
                    'last_price': self._get_field_value(current_quote, '最新价', None),
                    'volume': self._get_field_value(current_quote, '成交量', None),
                    'high': self._get_field_value(current_quote, '最高', None),
                    'low': self._get_field_value(current_quote, '最低', None),
                    'open': self._get_field_value(current_quote, '开盘', None),
                    'prev_close': self._get_field_value(current_quote, '昨收', None),
                    'change_pct': self._get_field_value(current_quote, '涨跌幅', None),
                    'vix': vix
                },
                'fundamental_data': metrics,
                'technical_data': {
                    'close_prices': hist_data[self._get_column_name(hist_data, ['收盘', 'close'])].tolist() if not hist_data.empty else [],
                    'volumes': hist_data[self._get_column_name(hist_data, ['成交量', 'volume'])].tolist() if not hist_data.empty else [],
                    'sma_50': metrics.get('sma_50'),
                    'sma_200': metrics.get('sma_200'),
                    'rsi_14': metrics.get('rsi_14'),
                    'macd': metrics.get('macd'),
                    'macd_signal': metrics.get('macd_signal'),
                    'macd_hist': metrics.get('macd_hist'),
                    'volume_ma': metrics.get('volume_ma')
                }
            }

            # Log missing metrics for debugging
            expected_metrics = {
                'pe_ratio', 'revenue_growth', 'eps_growth', 'roe', 'market_share', 
                'close_prices', 'volumes', 'owner_earnings', 'brand_value',
                'regulatory_risk', 'roic', 'market_concentration'
            }
            missing = expected_metrics - {k for k, v in metrics.items() if v not in [None, 0.0, [], {}]}
            if missing:
                logger.warning(f"Missing metrics: {list(missing)}")

            # Cache the results
            self.cache[symbol] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices: np.ndarray, periods: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            deltas = np.diff(prices)
            seed = deltas[:periods+1]
            up = seed[seed >= 0].sum()/periods
            down = -seed[seed < 0].sum()/periods
            rs = up/down if down != 0 else 0
            rsi = np.zeros_like(prices)
            rsi[:periods] = 100. - 100./(1. + rs)

            for i in range(periods, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta

                up = (up*(periods-1) + upval)/periods
                down = (down*(periods-1) + downval)/periods
                rs = up/down if down != 0 else 0
                rsi[i] = 100. - 100./(1. + rs)

            return float(rsi[-1])
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0  # Return neutral RSI on error

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD line, signal line and histogram"""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signal, adjust=False).mean()
            hist = macd - signal
            return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return 0.0, 0.0, 0.0

    def _calculate_owner_earnings(self, financial_data: pd.DataFrame) -> float:
        """Calculate Owner Earnings (Net Income + Depreciation - Capital Expenditures)"""
        try:
            net_income = self._get_field_value(financial_data, '净利润', 0.0)
            depreciation = self._get_field_value(financial_data, '折旧和摊销', 0.0)
            capex = self._get_field_value(financial_data, '资本支出', 0.0)
            return net_income + depreciation - capex
        except Exception as e:
            logger.error(f"Error calculating owner earnings: {str(e)}")
            return 0.0

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safely perform division handling zeros and NaN"""
        try:
            if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
                return 0
            return float(numerator) / float(denominator)
        except:
            return 0

    def _calculate_metrics(self, symbol: str, hist_data: pd.DataFrame, financial_data: pd.DataFrame, 
                         current_quote: pd.Series) -> Dict[str, float]:
        """Calculate all required metrics from the collected data"""
        metrics = {}
        
        try:
            # Basic financial metrics
            metrics['pe_ratio'] = self._get_field_value(financial_data, 'pe_ratio', 0.0)
            metrics['pb_ratio'] = self._get_field_value(financial_data, 'pb_ratio', 0.0)
            metrics['revenue_growth'] = self._get_field_value(financial_data, 'revenue_growth', 0.0)
            metrics['eps_growth'] = self._get_field_value(financial_data, 'eps_growth', 0.0)
            metrics['roe'] = self._get_field_value(financial_data, 'roe', 0.0)
            metrics['profit_margin'] = self._get_field_value(financial_data, 'profit_margin', 0.0)
            metrics['debt_ratio'] = self._get_field_value(financial_data, 'debt_ratio', 0.0)
            metrics['owner_earnings'] = self._calculate_owner_earnings(financial_data)

            # Technical metrics
            if not hist_data.empty:
                close_prices = hist_data[self._get_column_name(hist_data, ['收盘', 'close'])].values
                volumes = hist_data[self._get_column_name(hist_data, ['成交量', 'volume'])].values

                # Moving averages
                metrics['sma_50'] = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else close_prices[-1]
                metrics['sma_200'] = np.mean(close_prices[-200:]) if len(close_prices) >= 200 else close_prices[-1]
                metrics['volume_ma'] = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]

                # RSI and MACD
                metrics['rsi_14'] = self._calculate_rsi(close_prices)
                macd, signal, hist = self._calculate_macd(pd.Series(close_prices))
                metrics['macd'] = macd
                metrics['macd_signal'] = signal
                metrics['macd_hist'] = hist

            # Current market metrics
            metrics['price'] = self._get_field_value(current_quote, '最新价', 0.0)
            metrics['volume'] = self._get_field_value(current_quote, '成交量', 0.0)
            metrics['market_cap'] = self._get_field_value(current_quote, 'market_cap', 0.0)
            metrics['price_change_pct'] = self._get_field_value(current_quote, '涨跌幅', 0.0)
            
            # Try to get additional fundamental metrics
            try:
                industry_metrics = self._get_industry_metrics(symbol)
                metrics.update(industry_metrics)
                
                # Calculate market share if we have industry data
                if metrics.get('market_cap') and metrics.get('industry_market_cap'):
                    metrics['market_share'] = self._safe_divide(
                        metrics['market_cap'],
                        metrics.get('industry_market_cap', 0.0)
                    ) * 100
                
                # Use industry averages as fallbacks
                if metrics.get('pe_ratio', 0.0) == 0.0 and metrics.get('industry_pe'):
                    metrics['pe_ratio'] = metrics['industry_pe']
                if metrics.get('pb_ratio', 0.0) == 0.0 and metrics.get('industry_pb'):
                    metrics['pb_ratio'] = metrics['industry_pb']
            except Exception as e:
                logger.error(f"Error calculating additional metrics: {str(e)}")

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            # Ensure we return at least empty metrics
            for required_metric in REQUIRED_FINANCIAL_METRICS + REQUIRED_TECHNICAL_METRICS:
                if required_metric not in metrics:
                    metrics[required_metric] = 0.0

        return metrics

    def _get_column_name(self, df: pd.DataFrame, possible_names: List[str]) -> str:
        """Find the first matching column name from a list of possibilities"""
        for name in possible_names:
            if name in df.columns:
                return name
        # Try GB2312 encoded names
        for name in possible_names:
            encoded_names = [col for col in df.columns 
                           if col.encode('utf-8').decode('gb2312', errors='ignore') == name]
            if encoded_names:
                return encoded_names[0]
        raise ValueError(f"Could not find any of these columns: {possible_names}")

    def _get_fundamental_indicators(self, symbol: str) -> pd.DataFrame:
        """Get fundamental indicators from multiple sources"""
        try:
            # Try main financial indicators first
            data = ak.stock_financial_analysis_indicator(symbol=symbol)
            
            if data.empty:
                # Try alternative source: stock_a_lg_indicator
                data = ak.stock_a_lg_indicator(symbol=symbol)
            
            if data.empty:
                # Try another alternative: stock_financial_abstract
                data = ak.stock_financial_abstract(symbol=symbol)
                
            if data.empty:
                # As last resort, try getting basic info
                data = ak.stock_individual_info_em(symbol=symbol)
                
            return data
        except Exception as e:
            logger.error(f"Error getting fundamental indicators: {str(e)}")
            return pd.DataFrame()

    def _get_industry_metrics(self, symbol: str) -> Dict[str, float]:
        """Get industry related metrics"""
        try:
            # Get industry data
            ind_data = ak.stock_industry_pe_analysis(symbol=symbol[:6])
            metrics = {}
            
            if not ind_data.empty:
                metrics['industry_pe'] = self._get_field_value(ind_data, '行业市盈率', 0.0)
                metrics['industry_pb'] = self._get_field_value(ind_data, '行业市净率', 0.0)
                metrics['market_concentration'] = self._get_field_value(ind_data, '行业集中度', 50.0)
            
            # Get company position in industry
            try:
                rank_data = ak.stock_rank_cxg(symbol=symbol[:6])
                if not rank_data.empty:
                    metrics['competitive_position'] = self._get_field_value(rank_data, '排名', 0.0)
            except:
                pass
                
            return metrics
        except Exception as e:
            logger.error(f"Error getting industry metrics: {str(e)}")
            return {}

    def _calculate_market_share(self, revenue: float, industry_revenue: float) -> float:
        """Calculate market share percentage"""
        try:
            return self._safe_divide(revenue, industry_revenue) * 100 if industry_revenue > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating market share: {str(e)}")
            return 0.0

    def _calculate_brand_value(self, metrics: Dict[str, float], market_share: float) -> float:
        """Calculate brand value score based on various metrics"""
        try:
            profit_margin = metrics.get('profit_margin', 0.0)
            revenue_growth = metrics.get('revenue_growth', 0.0)
            market_position = market_share / 100  # Convert to decimal
            
            # Simple weighted average of key factors
            brand_value = (
                0.4 * market_position +  # Market position weight
                0.3 * (profit_margin / 100) +  # Profitability weight
                0.3 * (revenue_growth / 100)  # Growth weight
            ) * 100  # Convert back to percentage
            
            return max(0.0, min(100.0, brand_value))
        except Exception as e:
            logger.error(f"Error calculating brand value: {str(e)}")
            return 0.0

    def _calculate_capital_efficiency(self, financial_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate capital efficiency metrics"""
        try:
            roic = self._get_field_value(financial_data, '投资回报率', 0.0)
            cash_conversion = self._get_field_value(financial_data, '现金转换周期', 0.0)
            capex_ratio = self._get_field_value(financial_data, '资本支出比率', 0.0)
            
            return {
                'roic': roic,
                'cash_conversion_cycle': cash_conversion,
                'capex_to_sales': capex_ratio
            }
        except Exception as e:
            logger.error(f"Error calculating capital efficiency: {str(e)}")
            return {'roic': 0.0, 'cash_conversion_cycle': 0.0, 'capex_to_sales': 0.0}

    def _calculate_regulatory_risk(self, industry: str) -> float:
        """Calculate regulatory risk score (0-100) based on industry"""
        try:
            # Define high-risk industries and their base risk scores
            industry_risk = {
                '金融': 80,
                '银行': 80,
                '证券': 75,
                '保险': 75,
                '医药': 70,
                '医疗器械': 70,
                '房地产': 65,
                '互联网': 60,
                '电信': 60,
                '能源': 55,
                '采矿': 50,
            }
            
            # Get base risk score or default to moderate risk
            base_risk = 40.0  # Default moderate risk
            for key, risk in industry_risk.items():
                if key in industry:
                    base_risk = risk
                    break
                    
            return base_risk
        except Exception as e:
            logger.error(f"Error calculating regulatory risk: {str(e)}")
            return 50.0  # Return moderate risk on error

    def _get_additional_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get additional market data from various sources"""
        try:
            # Try to get institutional holdings
            inst_holdings = ak.stock_institute_hold(symbol=symbol)
            
            # Try to get margin trading data
            margin_data = ak.stock_margin_detail_em(symbol=symbol)
            
            # Try to get short interest
            short_data = ak.stock_a_below_cost_em(symbol=symbol)
            
            result = {}
            
            if not inst_holdings.empty:
                result['institutional_ownership'] = self._get_field_value(inst_holdings, '机构持股比例', 0.0)
                
            if not margin_data.empty:
                result['margin_ratio'] = self._get_field_value(margin_data, '融资余额占比', 0.0)
                
            if not short_data.empty:
                result['short_ratio'] = self._get_field_value(short_data, '空头持仓比例', 0.0)
                
            return result
        except Exception as e:
            logger.error(f"Error getting additional market data: {str(e)}")
            return {}
