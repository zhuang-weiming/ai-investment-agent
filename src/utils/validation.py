"""Validation utilities for stock data collection and analysis"""
from typing import Dict, Any, List, Optional
import re
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    Chinese A-share stock symbols are 6 digits, optionally followed by .SZ or .SH
    """
    if not symbol:
        return False
    # Allow digits only or digits with .SZ/.SH suffix
    return bool(symbol.replace('.SH', '').replace('.SZ', '').isdigit())

def validate_dataframe_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """
    Validate that DataFrame contains required columns.
    Returns list of missing columns.
    """
    try:
        # Try both UTF-8 and GB2312 encoded column names
        df_cols = set()
        for col in df.columns:
            try:
                # Try different encodings
                if isinstance(col, bytes):
                    df_cols.add(col.decode('utf-8'))
                    df_cols.add(col.decode('gb2312'))
                else:
                    df_cols.add(col)
            except:
                df_cols.add(col)
        
        missing = []
        for col in required_cols:
            # Try both UTF-8 and GB2312 encoded required column names
            col_found = False
            try:
                col_utf8 = col.encode('utf-8').decode('utf-8')
                col_gb = col.encode('utf-8').decode('gb2312', errors='ignore')
                if col in df_cols or col_utf8 in df_cols or col_gb in df_cols:
                    col_found = True
            except:
                if col in df_cols:
                    col_found = True
            
            if not col_found:
                missing.append(col)
        
        return missing
    except Exception as e:
        logger.error(f"Error validating DataFrame columns: {str(e)}")
        return required_cols

def validate_quote_data(quote: pd.Series, required_fields: List[str]) -> List[str]:
    """
    Validate that quote data contains required fields.
    Returns list of missing fields.
    """
    if quote is None:
        return required_fields
    
    try:
        quote_fields = set(quote.index)
        missing = []
        for field in required_fields:
            # Try different encodings for field names
            field_found = False
            try:
                field_utf8 = field.encode('utf-8').decode('utf-8')
                field_gb = field.encode('utf-8').decode('gb2312', errors='ignore')
                if field in quote_fields or field_utf8 in quote_fields or field_gb in quote_fields:
                    field_found = True
            except:
                if field in quote_fields:
                    field_found = True
            
            if not field_found:
                missing.append(field)
        return missing
    except Exception as e:
        logger.error(f"Error validating quote data: {str(e)}")
        return required_fields

def validate_metrics_data(metrics: Dict[str, Any], required_metrics: List[str]) -> List[str]:
    """
    Validate that metrics data contains required fields.
    Returns list of missing metrics.
    """
    if not metrics:
        return required_metrics
    
    try:
        missing = []
        for metric in required_metrics:
            if metric not in metrics:
                missing.append(metric)
            else:
                value = metrics[metric]
                if value is None or (isinstance(value, float) and (pd.isna(value) or pd.isinf(value))):
                    missing.append(metric)
        return missing
    except Exception as e:
        logger.error(f"Error validating metrics data: {str(e)}")
        return required_metrics

def validate_historical_data(hist_data: pd.DataFrame) -> bool:
    """
    Validate historical data requirements:
    - Not empty
    - Has required columns
    - Data is within expected date range
    - No large gaps in data
    """
    try:
        if hist_data is None or hist_data.empty:
            return False
            
        # Check for minimum data points (e.g., need at least 200 days for SMA200)
        if len(hist_data) < 200:
            return False
            
        # Check for required columns with encoding-safe comparison
        required_cols = ['收盘', '成交量', '开盘', '最高', '最低']
        missing_cols = validate_dataframe_columns(hist_data, required_cols)
        if missing_cols:
            return False
            
        # Validate no missing values in critical columns
        for col in required_cols:
            col_found = False
            for df_col in hist_data.columns:
                try:
                    if col == df_col or col.encode('utf-8').decode('gb2312', errors='ignore') == df_col:
                        if hist_data[df_col].isnull().any():
                            return False
                        col_found = True
                        break
                except:
                    continue
            if not col_found:
                return False
                
        # Check data range
        last_year = datetime.now().replace(year=datetime.now().year - 1)
        if '日期' in hist_data.columns:
            dates = pd.to_datetime(hist_data['日期'])
            if dates.max() - dates.min() < pd.Timedelta(days=180):
                return False
                
            # Check for large gaps (more than 5 trading days)
            gaps = dates.sort_values().diff().max()
            if gaps > pd.Timedelta(days=5):
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating historical data: {str(e)}")
        return False
