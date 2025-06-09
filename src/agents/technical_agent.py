import math
import json
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage

def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default

def make_json_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return make_json_serializable(obj.to_dict())
    return obj

def technical_analyst_agent(state):
    data = state["data"]
    tickers = data["tickers"]
    technical_analysis = {}

    for ticker in tickers:
        prices_df = None
        market_data = None

        # Convert AKShare data format to our format
        if "hist_data" in data:
            prices_df = pd.DataFrame({
                "date": pd.to_datetime(data["hist_data"]["日期"]),
                "open": data["hist_data"]["开盘"].astype(float),
                "high": data["hist_data"]["最高"].astype(float),
                "low": data["hist_data"]["最低"].astype(float),
                "close": data["hist_data"]["收盘"].astype(float),
                "volume": data["hist_data"]["成交量"].astype(float)
            }).set_index("date")

        if "market_data" in data:
            market_data = {
                "last_price": safe_float(data["market_data"].get("最新价")),
                "change_pct": safe_float(data["market_data"].get("涨跌幅")),
                "volume": safe_float(data["market_data"].get("成交量")),
                "turnover_rate": safe_float(data["market_data"].get("换手率")),
                "pe_ratio": safe_float(data["market_data"].get("市盈率-动态")),
                "pb_ratio": safe_float(data["market_data"].get("市净率"))
            }

        # Perform technical analysis if we have data
        if prices_df is not None and market_data is not None:
            # Calculate technical indicators
            rsi = calculate_rsi(prices_df)
            bb = calculate_bollinger_bands(prices_df)
            ema_20 = calculate_ema(prices_df, span=20)
            ema_60 = calculate_ema(prices_df, span=60)
            
            # Get latest values
            current_rsi = float(rsi.iloc[-1])
            current_bb = {k: float(v) for k, v in bb.iloc[-1].items()}
            current_ema20 = float(ema_20.iloc[-1])
            current_ema60 = float(ema_60.iloc[-1])
            current_price = market_data["last_price"]
            
            # Generate signals
            trend_signal = 1 if current_price > current_ema20 > current_ema60 else (-1 if current_price < current_ema20 < current_ema60 else 0)
            momentum_signal = 1 if current_rsi > 60 else (-1 if current_rsi < 40 else 0)
            
            # Combined signal logic
            signal = "bullish" if trend_signal + momentum_signal > 0 else ("bearish" if trend_signal + momentum_signal < 0 else "neutral")
            confidence = min(abs(trend_signal + momentum_signal) * 25 + 50, 100)
            
            technical_analysis[ticker] = {
                "signal": signal,
                "confidence": float(confidence),
                "reasoning": f"Analysis based on: RSI({current_rsi:.1f}), EMA-20/60 trend({'up' if trend_signal > 0 else 'down' if trend_signal < 0 else 'neutral'}), "
                           f"current price relative to Bollinger Bands({current_bb['upper_band']:.1f}/{current_bb['lower_band']:.1f})",
                "metrics": {
                    "rsi": current_rsi,
                    "trend": float(trend_signal),
                    "momentum": float(momentum_signal),
                    "price": current_price,
                    "volume": float(market_data["volume"]),
                    "turnover": float(market_data["turnover_rate"])
                }
            }
        else:
            technical_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 50,
                "reasoning": "Limited data available for analysis",
                "metrics": {}
            }

    technical_analysis = make_json_serializable(technical_analysis)
    message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }

def calculate_trend_signals(prices_df):
    # Placeholder for trend signals calculation logic
    return {"signal": 0, "confidence": 0, "metrics": {}}

def calculate_mean_reversion_signals(prices_df):
    # Placeholder for mean reversion signals calculation logic
    return {"signal": 0, "confidence": 0, "metrics": {}}

def calculate_momentum_signals(prices_df):
    # Placeholder for momentum signals calculation logic
    return {"signal": 0, "confidence": 0, "metrics": {}}

def calculate_volatility_signals(prices_df):
    # Placeholder for volatility signals calculation logic
    return {"signal": 0, "confidence": 0, "metrics": {}}

def calculate_stat_arb_signals(prices_df):
    # Placeholder for statistical arbitrage signals calculation logic
    return {"signal": 0, "confidence": 0, "metrics": {}}

def weighted_signal_combination(signals_dict, weights):
    # Placeholder for weighted signal combination logic
    combined_signal = 0
    total_confidence = 0
    for strategy, signals in signals_dict.items():
        weight = weights.get(strategy, 0)
        combined_signal += signals["signal"] * weight
        total_confidence += signals["confidence"] * weight
    if total_confidence == 0:
        return {"signal": 0, "confidence": 0}
    return {"signal": combined_signal / total_confidence, "confidence": total_confidence}

def normalize_pandas(metrics):
    # Placeholder for metrics normalization logic
    return metrics

def calculate_rsi(prices_df, period=14):
    # Placeholder for RSI calculation logic
    return pd.Series([50] * len(prices_df))

def calculate_bollinger_bands(prices_df, window=20, num_std_dev=2):
    # Placeholder for Bollinger Bands calculation logic
    return pd.DataFrame({
        "upper_band": prices_df["close"].rolling(window).mean() + num_std_dev * prices_df["close"].rolling(window).std(),
        "lower_band": prices_df["close"].rolling(window).mean() - num_std_dev * prices_df["close"].rolling(window).std(),
    })

def calculate_ema(prices_df, span=20):
    # Placeholder for EMA calculation logic
    return prices_df["close"].ewm(span=span, adjust=False).mean()

def calculate_adx(prices_df, period=14):
    # Placeholder for ADX calculation logic
    return pd.Series([25] * len(prices_df))

def calculate_atr(prices_df, period=14):
    # Placeholder for ATR calculation logic
    return pd.Series([1] * len(prices_df))

def calculate_hurst_exponent(prices_df, max_lag=20):
    # Placeholder for Hurst Exponent calculation logic
    return pd.Series([0.5] * len(prices_df))
