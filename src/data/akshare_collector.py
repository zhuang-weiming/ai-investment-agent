# AKShare data collector
import akshare as ak
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import numpy as np
from src.utils.validation import (
    validate_stock_symbol,
    validate_historical_data,
    validate_dataframe_columns,
    validate_metrics_data
)

logger = logging.getLogger(__name__)

class AKShareCollector:
    def __init__(self, rate_limit_delay: float = 0.5):
        """Initialize the collector with rate limiting
        
        Args:
            rate_limit_delay: Delay in seconds between API calls
        """
        self._last_call_time = 0
        self._rate_limit_delay = rate_limit_delay
        
    def _rate_limit(self):
        """Implement rate limiting between API calls"""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_call_time = time.time()

    def _validate_symbol(self, symbol: str) -> str:
        """Validate and format stock symbol for AKShare
        
        Returns:
            Formatted symbol without exchange suffix
        Raises:
            ValueError if invalid symbol
        """
        if not validate_stock_symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        return symbol.split('.')[0]  # Remove exchange suffix
        
    def _normalize_chinese_cols(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """Handle different Chinese encodings in column names
        
        Args:
            df: DataFrame to normalize
            target_cols: List of expected column names
            
        Returns:
            DataFrame with normalized column names
        """
        rename_map = {}
        for col in df.columns:
            for target in target_cols:
                try:
                    if col == target or col.encode('utf-8').decode('gb2312', errors='ignore') == target:
                        rename_map[col] = target
                        break
                except:
                    continue
        return df.rename(columns=rename_map)

    def _retry_api_call(self, func, *args, max_retries: int = 3, **kwargs) -> Optional[pd.DataFrame]:
        """Retry failed API calls with exponential backoff
        
        Args:
            func: AKShare API function to call
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            DataFrame from API call or None if all retries fail
        """
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {str(e)}")
                    return None
                wait_time = (2 ** attempt) * self._rate_limit_delay
                logger.warning(f"API call failed, retrying in {wait_time:.1f}s: {str(e)}")
                time.sleep(wait_time)
        return None

    def _check_data_quality(self, df: pd.DataFrame, required_cols: List[str], 
                           numeric_cols: Optional[List[str]] = None) -> bool:
        """Check data quality of DataFrame
        
        Args:
            df: DataFrame to check
            required_cols: List of required column names
            numeric_cols: List of columns that should contain numeric data
            
        Returns:
            True if data quality checks pass, False otherwise
        """
        if df is None or df.empty:
            return False
            
        # Check required columns with encoding-safe comparison
        missing = validate_dataframe_columns(df, required_cols)
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            return False
            
        if numeric_cols:
            for col in numeric_cols:
                # Find matching column considering encoding
                col_found = False
                for df_col in df.columns:
                    try:
                        if col == df_col or col.encode('utf-8').decode('gb2312', errors='ignore') == df_col:
                            # Check numeric data quality
                            if not pd.to_numeric(df[df_col], errors='coerce').notna().any():
                                logger.warning(f"No valid numeric data in column: {col}")
                                return False
                            col_found = True
                            break
                    except:
                        continue
                        
                if not col_found:
                    logger.warning(f"Required numeric column not found: {col}")
                    return False
                    
        return True

    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """Get financial metrics for a stock
        
        Returns normalized dictionary with financial metrics:
            - pe_ratio: Price to earnings ratio
            - pb_ratio: Price to book ratio
            - roe: Return on equity
            - revenue_growth: Year-over-year revenue growth
            - eps_growth: Year-over-year EPS growth
            - profit_margin: Profit margin
            - debt_ratio: Debt to assets ratio
            - dividend_yield: Dividend yield
        """
        try:
            code = self._validate_symbol(symbol)
            self._rate_limit()
            
            # Get main financial indicators
            df = self._retry_api_call(ak.stock_financial_analysis_indicator, symbol=code)
            if df is None or df.empty:
                logger.warning(f"No financial data found for {symbol}")
                return {}
                
            # Normalize column names
            df = self._normalize_chinese_cols(df, ['净资产收益率(%)', '营业收入同比增长率(%)', 
                                                '每股收益同比增长率(%)', '销售毛利率(%)', 
                                                '资产负债率(%)', '市盈率', '市净率', '股息率(%)'])
            
            # Get latest row (most recent quarter)
            latest = df.iloc[0].to_dict() if not df.empty else {}
            
            # Normalize to standard field names
            return {
                'pe_ratio': float(latest.get('市盈率', 0)),
                'pb_ratio': float(latest.get('市净率', 0)),
                'roe': float(latest.get('净资产收益率(%)', 0)),
                'revenue_growth': float(latest.get('营业收入同比增长率(%)', 0)),
                'eps_growth': float(latest.get('每股收益同比增长率(%)', 0)),
                'profit_margin': float(latest.get('销售毛利率(%)', 0)),
                'debt_ratio': float(latest.get('资产负债率(%)', 0)),
                'dividend_yield': float(latest.get('股息率(%)', 0))
            }
            
        except ValueError as e:
            logger.error(str(e))
            return {}
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {str(e)}")
            return {}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data for a stock
        
        Returns normalized dictionary with fields:
            - last_price: Latest price
            - volume: Trading volume 
            - change_pct: Price change percentage
            - high: Daily high
            - low: Daily low
            - open: Opening price
            - prev_close: Previous close
            - market_cap: Market capitalization
        """
        try:
            code = self._validate_symbol(symbol)
            self._rate_limit()
            
            df = ak.stock_zh_a_spot_em()
            df = self._normalize_chinese_cols(df, ['代码', '最新价', '成交量', '涨跌幅', '最高', '最低', '开盘', '昨收', '总市值'])
            
            row = df[df['代码'] == code]
            if row.empty:
                logger.warning(f"No market data found for {symbol}")
                return {}
                
            data = row.iloc[0].to_dict()
            
            # Normalize to standard field names
            return {
                'last_price': float(data.get('最新价', 0)),
                'volume': float(data.get('成交量', 0)),
                'change_pct': float(data.get('涨跌幅', 0)),
                'high': float(data.get('最高', 0)),
                'low': float(data.get('最低', 0)), 
                'open': float(data.get('开盘', 0)),
                'prev_close': float(data.get('昨收', 0)),
                'market_cap': float(data.get('总市值', 0))
            }
            
        except ValueError as e:
            logger.error(str(e))
            return {}
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return {}

    def get_historical_data(self, symbol: str, days: int = 120) -> Dict[str, Any]:
        """Get historical price data for technical analysis
        
        Args:
            symbol: Stock symbol
            days: Number of days of history to retrieve
            
        Returns dictionary with date-indexed:
            - close: Adjusted closing prices
            - volume: Trading volumes
            - high: Daily highs
            - low: Daily lows
            - open: Opening prices
        """
        try:
            code = self._validate_symbol(symbol)
            self._rate_limit()
            
            end_date = datetime.now()
            start_date = (end_date - timedelta(days=days)).strftime("%Y%m%d")
            end_date = end_date.strftime("%Y%m%d")
            
            df = ak.stock_zh_a_hist(
                symbol=code,  
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'  # Use forward adjustment
            )
            
            df = self._normalize_chinese_cols(df, ['日期', '收盘', '成交量', '开盘', '最高', '最低'])
            
            if not validate_historical_data(df):
                logger.error(f"Invalid historical data format for {symbol}")
                return {}
                
            # Convert date column to index
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
                
            # Return normalized dictionary
            return {
                'close': df['收盘'].to_dict(),
                'volume': df['成交量'].to_dict(),
                'high': df['最高'].to_dict(),
                'low': df['最低'].to_dict(),
                'open': df['开盘'].to_dict()
            }
            
        except ValueError as e:
            logger.error(str(e))
            return {}
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return {}

    def get_financials(self, symbol: str) -> List[Dict[str, Any]]:
        """Get comprehensive financial indicators for fundamental analysis
        
        Returns list of dictionaries with metrics:
            - Revenue and profitability metrics
            - Cash flow metrics 
            - Operating metrics
            - Balance sheet metrics
            - Industry comparison
        """
        try:
            code = self._validate_symbol(symbol)
            self._rate_limit()
            
            # Get core financial indicators
            df = ak.stock_financial_analysis_indicator(symbol=code)
            df = self._normalize_chinese_cols(df, [
                '营业收入', '营业收入同比增长率', '净利润', '净利润同比增长率',
                '销售毛利率', '销售净利率', '资产负债率', '流动比率',
                '速动比率', '存货周转率', '应收账款周转率', '总资产周转率',
                '市盈率', '市净率', '市销率', '每股收益',
                '净资产收益率', '经营活动现金流量净额', '投资活动现金流量净额',
                '筹资活动现金流量净额', '现金及现金等价物净增加额'
            ])
            
            if df is None or df.empty:
                logger.warning(f"No financial data found for {symbol}")
                return []
                
            # Get cash flow details
            self._rate_limit()
            cash_df = ak.stock_cash_flow_sheet_by_yearly(symbol=code)
            if cash_df is not None and not cash_df.empty:
                cash_df = self._normalize_chinese_cols(cash_df, [
                    '经营活动产生的现金流量净额', '投资活动产生的现金流量净额',
                    '筹资活动产生的现金流量净额', '现金及现金等价物净增加额',
                    '折旧和摊销', '资本支出'
                ])
                
                # Merge relevant cash flow metrics into main df
                cash_data = cash_df.iloc[0].to_dict() if len(cash_df) > 0 else {}
                for k, v in cash_data.items():
                    df[k] = v
            
            # Calculate additional metrics
            records = []
            for _, row in df.iterrows():
                record = row.to_dict()
                
                # Calculate owner earnings (net income + depreciation - capex)
                try:
                    net_income = float(record.get('净利润', 0))
                    depreciation = float(record.get('折旧和摊销', 0))
                    capex = float(record.get('资本支出', 0))
                    record['owner_earnings'] = net_income + depreciation - capex
                except (ValueError, TypeError):
                    record['owner_earnings'] = None
                    
                # Add industry comparison if available
                try:
                    self._rate_limit()
                    ind_df = ak.stock_industry_pe_ratio(symbol=code)
                    if ind_df is not None and not ind_df.empty:
                        ind_df = self._normalize_chinese_cols(ind_df, ['行业市盈率'])
                        record['industry_pe'] = float(ind_df.iloc[0]['行业市盈率'])
                except:
                    record['industry_pe'] = None
                    
                records.append(record)
                
            # Validate required metrics are present
            required_metrics = [
                '营业收入', '净利润', '销售毛利率', '资产负债率',
                '市盈率', '市净率', '净资产收益率'
            ]
            missing = validate_metrics_data(records[0], required_metrics) if records else required_metrics
            if missing:
                logger.warning(f"Missing required metrics for {symbol}: {missing}")
            
            return records
            
        except ValueError as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {str(e)}")
            return []

    def get_news(self, symbol: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get recent news articles for the stock
        
        Args:
            symbol: Stock symbol
            days_back: Number of days of news to retrieve
            
        Returns list of dictionaries with:
            - title: News title
            - date: Publication date
            - source: News source
            - url: Article URL 
            - summary: Article summary if available
        """
        try:
            code = self._validate_symbol(symbol)
            self._rate_limit()
            
            df = ak.stock_news_em(symbol=code)
            df = self._normalize_chinese_cols(df, ['日期', '标题', '来源', '网址', '内容'])
            
            if df is None or df.empty:
                logger.warning(f"No news found for {symbol}")
                return []
                
            # Filter by date if possible
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                df = df[df['日期'] >= cutoff]
            
            # Convert to normalized records
            news = []
            for _, row in df.iterrows():
                news.append({
                    'title': row.get('标题', ''),
                    'date': row.get('日期', ''),
                    'source': row.get('来源', ''),
                    'url': row.get('网址', ''),
                    'summary': row.get('内容', '')
                })
            
            return news
            
        except ValueError as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []

    def get_insider_trades(self, symbol: str, days_back: int = 90) -> List[Dict[str, Any]]:
        """Get recent insider trading data
        
        Args:
            symbol: Stock symbol
            days_back: Number of days of history to retrieve
            
        Returns list of dictionaries with:
            - date: Transaction date
            - name: Insider name
            - position: Insider's position
            - action: Buy/Sell
            - shares: Number of shares
            - price: Transaction price
            - value: Total value
            - shares_held: Remaining shares held
        """
        try:
            code = self._validate_symbol(symbol)
            self._rate_limit()
            
            df = ak.stock_executive_shares_change_em(symbol=code)
            df = self._normalize_chinese_cols(df, [
                '变动日期', '股东名称', '职务', '变动原因',
                '变动数量', '成交均价', '变动后持股数', '变动后持股比例'
            ])
            
            if df is None or df.empty:
                logger.warning(f"No insider trades found for {symbol}")
                return []
            
            # Filter by date if possible
            if '变动日期' in df.columns:
                df['变动日期'] = pd.to_datetime(df['变动日期'])
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                df = df[df['变动日期'] >= cutoff]
            
            # Convert to normalized records
            trades = []
            for _, row in df.iterrows():
                try:
                    shares = float(row.get('变动数量', 0))
                    price = float(row.get('成交均价', 0))
                    value = shares * price
                except (ValueError, TypeError):
                    shares = price = value = 0
                
                trades.append({
                    'date': row.get('变动日期', ''),
                    'name': row.get('股东名称', ''),
                    'position': row.get('职务', ''),
                    'action': row.get('变动原因', ''),
                    'shares': shares,
                    'price': price,
                    'value': value,
                    'shares_held': float(row.get('变动后持股数', 0)),
                    'ownership_pct': float(row.get('变动后持股比例', 0))
                })
            
            return trades
            
        except ValueError as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(f"Error fetching insider trades for {symbol}: {str(e)}")
            return []

    def _get_field_value(self, df: pd.DataFrame, field: str, default: float = 0.0) -> float:
        """Get field value with encoding-safe comparison and field name mapping
        
        Handles both English and Chinese field names with mapping
        """
        field_mapping = {
            'pe_ratio': '市盈率',
            'pb_ratio': '市净率',
            'revenue_growth': '营业收入同比增长率',
            'eps_growth': '基本每股收益同比增长率',
            'roe': '净资产收益率',
            'profit_margin': '销售毛利率',
            'debt_ratio': '资产负债率',
            'net_income': '净利润',
            'depreciation': '折旧和摊销',
            'capex': '资本支出'
        }
        
        try:
            # Try direct match first
            if field in df.columns:
                return float(df[field].iloc[0])
                
            # Try mapped Chinese name
            chinese_field = field_mapping.get(field, field)
            if chinese_field in df.columns:
                return float(df[chinese_field].iloc[0])
                
            # Try encoding-safe comparison
            for col in df.columns:
                try:
                    if (col == chinese_field or 
                        col.encode('utf-8').decode('gb2312', errors='ignore') == chinese_field or
                        col == field):
                        return float(df[col].iloc[0])
                except:
                    continue
            return default
            
        except (ValueError, TypeError, IndexError):
            return default

    def _calculate_owner_earnings(self, df: pd.DataFrame) -> float:
        """Calculate owner earnings from financial data"""
        try:
            net_income = self._get_field_value(df, '净利润')
            depreciation = self._get_field_value(df, '折旧和摊销')
            capex = self._get_field_value(df, '资本支出')
            return net_income + depreciation - capex
        except:
            return 0.0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI for technical analysis"""
        try:
            delta = np.diff(prices)
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            
            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except:
            return 50.0  # Neutral RSI value
            
    def _safe_divide(self, a: float, b: float) -> float:
        """Safe division handling zeros and NaN"""
        try:
            if pd.isna(a) or pd.isna(b) or b == 0:
                return 0.0
            return float(a) / float(b)
        except:
            return 0.0

    def _calculate_metrics(self, symbol: str, hist_data: pd.DataFrame, financial_data: pd.DataFrame, 
                         market_data: pd.Series) -> Dict[str, Any]:
        """Calculate combined metrics for analysis"""
        try:
            metrics = {}
            
            # Market metrics
            metrics['pe_ratio'] = self._get_field_value(financial_data, '市盈率')
            metrics['pb_ratio'] = self._get_field_value(financial_data, '市净率')
            metrics['revenue_growth'] = self._get_field_value(financial_data, '营业收入同比增长率')
            metrics['eps_growth'] = self._get_field_value(financial_data, '基本每股收益同比增长率')
            metrics['roe'] = self._get_field_value(financial_data, '净资产收益率')
            metrics['profit_margin'] = self._get_field_value(financial_data, '销售毛利率')
            metrics['debt_ratio'] = self._get_field_value(financial_data, '资产负债率')
            
            # Technical metrics from historical data
            if hist_data is not None and not hist_data.empty:
                close_prices = hist_data['收盘'].astype(float).values
                volume = hist_data['成交量'].astype(float).values
                
                metrics['rsi_14'] = self._calculate_rsi(close_prices)
                metrics['sma_50'] = float(pd.Series(close_prices).rolling(50).mean().iloc[-1])
                metrics['sma_200'] = float(pd.Series(close_prices).rolling(200).mean().iloc[-1])
                metrics['volume_ma'] = float(pd.Series(volume).rolling(20).mean().iloc[-1])
                
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            return {}

    def get_market_cap(self, symbol: str) -> float:
        """Get market capitalization from real-time market data"""
        try:
            code = self._validate_symbol(symbol)
            self._rate_limit()
            
            df = ak.stock_zh_a_spot_em()
            df = self._normalize_chinese_cols(df, ['代码', '总市值'])
            
            row = df[df['代码'] == code]
            if row.empty:
                return 0.0
                
            market_cap = row.iloc[0].get('总市值', 0)
            return float(market_cap)
            
        except Exception as e:
            logger.error(f"Error fetching market cap for {symbol}: {str(e)}")
            return 0.0
