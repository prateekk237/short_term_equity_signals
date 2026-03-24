"""
Technical Analysis Engine - All FOSS indicators and signal computation.
Uses the `ta` library for standard indicators and custom logic for support/resistance.
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on OHLCV DataFrame."""
    if df is None or len(df) < 20:
        return df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    open_ = df["Open"]

    # ── Moving Averages ───────────────────────────────────────────────────
    for period in [5, 10, 20, 50]:
        if len(df) >= period:
            df[f"SMA_{period}"] = SMAIndicator(close, window=period).sma_indicator()
            df[f"EMA_{period}"] = EMAIndicator(close, window=period).ema_indicator()

    # ── MACD ──────────────────────────────────────────────────────────────
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # ── RSI ───────────────────────────────────────────────────────────────
    df["RSI_14"] = RSIIndicator(close, window=14).rsi()
    df["RSI_7"] = RSIIndicator(close, window=7).rsi()

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_Pct"] = bb.bollinger_pband()

    # ── Stochastic Oscillator ─────────────────────────────────────────────
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ── ADX (Trend Strength) ──────────────────────────────────────────────
    if len(df) >= 20:
        adx = ADXIndicator(high, low, close, window=14)
        df["ADX"] = adx.adx()
        df["ADX_Pos"] = adx.adx_pos()
        df["ADX_Neg"] = adx.adx_neg()

    # ── ATR (Volatility) ─────────────────────────────────────────────────
    atr = AverageTrueRange(high, low, close, window=14)
    df["ATR"] = atr.average_true_range()

    # ── Williams %R ──────────────────────────────────────────────────────
    df["Williams_R"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    # ── OBV (On-Balance Volume) ──────────────────────────────────────────
    df["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # ── Volume Analysis ──────────────────────────────────────────────────
    df["Vol_SMA_10"] = volume.rolling(window=10).mean()
    df["Vol_SMA_20"] = volume.rolling(window=20).mean()
    df["Vol_Ratio"] = volume / df["Vol_SMA_20"]  # Volume relative to 20-day avg
    df["Vol_Pct_Change"] = volume.pct_change() * 100

    # ── Price Action ─────────────────────────────────────────────────────
    df["Daily_Return"] = close.pct_change() * 100
    df["Weekly_Return"] = close.pct_change(periods=5) * 100
    df["Monthly_Return"] = close.pct_change(periods=20) * 100
    df["Body_Size"] = abs(close - open_) / (high - low + 0.001)  # Candle body ratio
    df["Upper_Shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (high - low + 0.001)
    df["Lower_Shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (high - low + 0.001)

    return df


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Find key support and resistance levels using pivot points and price clustering."""
    if df is None or len(df) < window:
        return {"supports": [], "resistances": [], "pivot": None}

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    # Classic Pivot Points
    last_high = high[-1]
    last_low = low[-1]
    last_close = close[-1]
    pivot = (last_high + last_low + last_close) / 3
    r1 = 2 * pivot - last_low
    r2 = pivot + (last_high - last_low)
    r3 = last_high + 2 * (pivot - last_low)
    s1 = 2 * pivot - last_high
    s2 = pivot - (last_high - last_low)
    s3 = last_low - 2 * (last_high - pivot)

    # Find swing highs and lows (local extrema)
    swing_highs = []
    swing_lows = []
    lookback = 3

    for i in range(lookback, len(high) - lookback):
        if high[i] == max(high[i - lookback : i + lookback + 1]):
            swing_highs.append(high[i])
        if low[i] == min(low[i - lookback : i + lookback + 1]):
            swing_lows.append(low[i])

    # Cluster nearby levels
    def cluster_levels(levels, threshold_pct=0.5):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < threshold_pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [round(np.mean(c), 2) for c in clusters]

    supports = cluster_levels(swing_lows[-10:]) + [round(s1, 2), round(s2, 2), round(s3, 2)]
    resistances = cluster_levels(swing_highs[-10:]) + [round(r1, 2), round(r2, 2), round(r3, 2)]

    # Filter: supports below current price, resistances above
    current_price = close[-1]
    supports = sorted(set(s for s in supports if s < current_price), reverse=True)[:5]
    resistances = sorted(set(r for r in resistances if r > current_price))[:5]

    return {
        "supports": supports,
        "resistances": resistances,
        "pivot": round(pivot, 2),
        "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
        "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
    }


def compute_signal_scores(df: pd.DataFrame, sr_levels: dict) -> dict:
    """
    Compute individual indicator scores and aggregate into a composite signal.
    Returns a dict with each indicator's contribution and the final score.
    Score range: -100 (strong sell) to +100 (strong buy)
    """
    if df is None or len(df) < 30:
        return {"total_score": 0, "signal": "NO DATA", "details": {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    scores = {}

    # ── 1. RSI Score (weight: 15) ────────────────────────────────────────
    rsi = latest.get("RSI_14", 50)
    if rsi < 30:
        scores["RSI"] = {"score": 12, "max": 15, "reason": f"Oversold ({rsi:.1f})"}
    elif rsi < 40:
        scores["RSI"] = {"score": 8, "max": 15, "reason": f"Approaching oversold ({rsi:.1f})"}
    elif rsi > 70:
        scores["RSI"] = {"score": -12, "max": 15, "reason": f"Overbought ({rsi:.1f})"}
    elif rsi > 60:
        scores["RSI"] = {"score": -5, "max": 15, "reason": f"Approaching overbought ({rsi:.1f})"}
    else:
        scores["RSI"] = {"score": 3, "max": 15, "reason": f"Neutral zone ({rsi:.1f})"}

    # ── 2. MACD Score (weight: 15) ───────────────────────────────────────
    macd_val = latest.get("MACD", 0)
    macd_sig = latest.get("MACD_Signal", 0)
    macd_hist = latest.get("MACD_Hist", 0)
    prev_hist = prev.get("MACD_Hist", 0)

    if macd_val > macd_sig and macd_hist > 0:
        if macd_hist > prev_hist:
            scores["MACD"] = {"score": 14, "max": 15, "reason": "Bullish crossover, momentum increasing"}
        else:
            scores["MACD"] = {"score": 8, "max": 15, "reason": "Bullish but momentum slowing"}
    elif macd_val < macd_sig and macd_hist < 0:
        if macd_hist < prev_hist:
            scores["MACD"] = {"score": -14, "max": 15, "reason": "Bearish crossover, momentum increasing"}
        else:
            scores["MACD"] = {"score": -6, "max": 15, "reason": "Bearish but momentum slowing"}
    else:
        scores["MACD"] = {"score": 0, "max": 15, "reason": "Neutral / transitioning"}

    # ── 3. Moving Average Score (weight: 15) ─────────────────────────────
    price = latest["Close"]
    ma_score = 0
    ma_reasons = []

    for period in [10, 20, 50]:
        sma_key = f"SMA_{period}"
        ema_key = f"EMA_{period}"
        if sma_key in latest and not pd.isna(latest[sma_key]):
            if price > latest[sma_key]:
                ma_score += 2.5
                ma_reasons.append(f"Above SMA{period}")
            else:
                ma_score -= 2.5
                ma_reasons.append(f"Below SMA{period}")
        if ema_key in latest and not pd.isna(latest[ema_key]):
            if price > latest[ema_key]:
                ma_score += 2.5
            else:
                ma_score -= 2.5

    # Check for golden/death cross (SMA10 vs SMA50)
    if "SMA_10" in latest and "SMA_50" in latest:
        if not pd.isna(latest["SMA_10"]) and not pd.isna(latest["SMA_50"]):
            if latest["SMA_10"] > latest["SMA_50"]:
                ma_score += 3
                ma_reasons.append("Golden cross pattern")
            else:
                ma_score -= 3
                ma_reasons.append("Death cross pattern")

    ma_score = max(-15, min(15, ma_score))
    scores["Moving_Avg"] = {"score": round(ma_score, 1), "max": 15, "reason": "; ".join(ma_reasons[:3])}

    # ── 4. Bollinger Band Score (weight: 10) ─────────────────────────────
    bb_pct = latest.get("BB_Pct", 0.5)
    if bb_pct < 0:
        scores["Bollinger"] = {"score": 9, "max": 10, "reason": "Price below lower band (squeeze buy)"}
    elif bb_pct < 0.2:
        scores["Bollinger"] = {"score": 6, "max": 10, "reason": "Near lower band (potential bounce)"}
    elif bb_pct > 1:
        scores["Bollinger"] = {"score": -9, "max": 10, "reason": "Price above upper band (overextended)"}
    elif bb_pct > 0.8:
        scores["Bollinger"] = {"score": -5, "max": 10, "reason": "Near upper band (caution)"}
    else:
        scores["Bollinger"] = {"score": 2, "max": 10, "reason": f"Mid-band range ({bb_pct:.2f})"}

    # ── 5. Volume Score (weight: 10) ─────────────────────────────────────
    vol_ratio = latest.get("Vol_Ratio", 1)
    daily_ret = latest.get("Daily_Return", 0)

    if vol_ratio > 1.5 and daily_ret > 0:
        scores["Volume"] = {"score": 9, "max": 10, "reason": f"High volume bullish ({vol_ratio:.1f}x avg)"}
    elif vol_ratio > 1.5 and daily_ret < 0:
        scores["Volume"] = {"score": -8, "max": 10, "reason": f"High volume bearish ({vol_ratio:.1f}x avg)"}
    elif vol_ratio > 1.0 and daily_ret > 0:
        scores["Volume"] = {"score": 5, "max": 10, "reason": "Above avg volume, positive"}
    elif vol_ratio < 0.5:
        scores["Volume"] = {"score": -2, "max": 10, "reason": "Very low volume (low conviction)"}
    else:
        scores["Volume"] = {"score": 1, "max": 10, "reason": f"Normal volume ({vol_ratio:.1f}x avg)"}

    # ── 6. Stochastic Score (weight: 10) ─────────────────────────────────
    stoch_k = latest.get("Stoch_K", 50)
    stoch_d = latest.get("Stoch_D", 50)

    if stoch_k < 20 and stoch_k > stoch_d:
        scores["Stochastic"] = {"score": 9, "max": 10, "reason": "Oversold bullish crossover"}
    elif stoch_k < 20:
        scores["Stochastic"] = {"score": 6, "max": 10, "reason": f"Oversold ({stoch_k:.1f})"}
    elif stoch_k > 80 and stoch_k < stoch_d:
        scores["Stochastic"] = {"score": -9, "max": 10, "reason": "Overbought bearish crossover"}
    elif stoch_k > 80:
        scores["Stochastic"] = {"score": -5, "max": 10, "reason": f"Overbought ({stoch_k:.1f})"}
    else:
        scores["Stochastic"] = {"score": 2, "max": 10, "reason": f"Neutral ({stoch_k:.1f})"}

    # ── 7. Trend Strength ADX (weight: 10) ───────────────────────────────
    adx = latest.get("ADX", 20)
    adx_pos = latest.get("ADX_Pos", 0)
    adx_neg = latest.get("ADX_Neg", 0)

    if adx > 25 and adx_pos > adx_neg:
        scores["ADX"] = {"score": 9, "max": 10, "reason": f"Strong uptrend (ADX={adx:.0f})"}
    elif adx > 25 and adx_neg > adx_pos:
        scores["ADX"] = {"score": -9, "max": 10, "reason": f"Strong downtrend (ADX={adx:.0f})"}
    elif adx < 20:
        scores["ADX"] = {"score": 0, "max": 10, "reason": f"Weak/no trend (ADX={adx:.0f})"}
    else:
        scores["ADX"] = {"score": 3, "max": 10, "reason": f"Moderate trend (ADX={adx:.0f})"}

    # ── 8. Support/Resistance Proximity (weight: 10) ─────────────────────
    sr_score = 0
    sr_reason = "No clear S/R signal"
    if sr_levels["supports"] and sr_levels["resistances"]:
        nearest_support = sr_levels["supports"][0]
        nearest_resistance = sr_levels["resistances"][0]
        dist_to_support = (price - nearest_support) / price * 100
        dist_to_resistance = (nearest_resistance - price) / price * 100

        reward_risk = dist_to_resistance / max(dist_to_support, 0.1)

        if dist_to_support < 1.5 and reward_risk > 2:
            sr_score = 9
            sr_reason = f"Near support, R:R={reward_risk:.1f}"
        elif dist_to_support < 2 and reward_risk > 1.5:
            sr_score = 6
            sr_reason = f"Close to support, R:R={reward_risk:.1f}"
        elif dist_to_resistance < 1:
            sr_score = -7
            sr_reason = f"Near resistance ({nearest_resistance:.2f})"
        else:
            sr_score = 2
            sr_reason = f"R:R ratio = {reward_risk:.1f}"

    scores["S/R_Levels"] = {"score": sr_score, "max": 10, "reason": sr_reason}

    # ── 9. Recent Momentum (weight: 5) ───────────────────────────────────
    weekly_ret = latest.get("Weekly_Return", 0)
    if not pd.isna(weekly_ret):
        if 1 < weekly_ret < 8:
            scores["Momentum"] = {"score": 4, "max": 5, "reason": f"Positive momentum ({weekly_ret:.1f}%)"}
        elif weekly_ret > 8:
            scores["Momentum"] = {"score": -2, "max": 5, "reason": f"Overextended ({weekly_ret:.1f}%)"}
        elif -5 < weekly_ret < -1:
            scores["Momentum"] = {"score": 3, "max": 5, "reason": f"Pullback opportunity ({weekly_ret:.1f}%)"}
        elif weekly_ret < -5:
            scores["Momentum"] = {"score": -4, "max": 5, "reason": f"Falling knife ({weekly_ret:.1f}%)"}
        else:
            scores["Momentum"] = {"score": 1, "max": 5, "reason": f"Flat ({weekly_ret:.1f}%)"}
    else:
        scores["Momentum"] = {"score": 0, "max": 5, "reason": "No data"}

    # ── Aggregate ────────────────────────────────────────────────────────
    total_score = sum(s["score"] for s in scores.values())
    max_possible = sum(s["max"] for s in scores.values())

    # Normalize to 0-100 scale
    normalized = ((total_score + max_possible) / (2 * max_possible)) * 100
    normalized = max(0, min(100, normalized))

    if normalized >= 75:
        signal = "STRONG BUY"
    elif normalized >= 60:
        signal = "BUY"
    elif normalized >= 45:
        signal = "HOLD / NEUTRAL"
    elif normalized >= 30:
        signal = "SELL"
    else:
        signal = "STRONG SELL"

    return {
        "total_score": round(total_score, 1),
        "normalized_score": round(normalized, 1),
        "max_possible": max_possible,
        "signal": signal,
        "details": scores,
    }


def compute_entry_exit(df: pd.DataFrame, sr_levels: dict, signal_data: dict) -> dict:
    """Compute recommended entry price, target price, and stop loss."""
    if df is None or len(df) < 5:
        return {}

    latest = df.iloc[-1]
    price = latest["Close"]
    atr = latest.get("ATR", price * 0.02)

    signal = signal_data.get("signal", "HOLD")

    if "BUY" in signal:
        entry = round(price * 0.998, 2)  # Slightly below current for limit order

        # Target: nearest resistance or ATR-based
        if sr_levels["resistances"]:
            target = sr_levels["resistances"][0]
            if (target - price) / price * 100 < 1:
                target = sr_levels["resistances"][1] if len(sr_levels["resistances"]) > 1 else price + 2 * atr
        else:
            target = round(price + 2 * atr, 2)

        # Stop loss: nearest support or ATR-based
        if sr_levels["supports"]:
            stop_loss = sr_levels["supports"][0]
            if (price - stop_loss) / price * 100 > 5:
                stop_loss = round(price - 1.5 * atr, 2)
        else:
            stop_loss = round(price - 1.5 * atr, 2)

        target = round(target, 2)
        stop_loss = round(stop_loss, 2)
        risk = price - stop_loss
        reward = target - price
        rr_ratio = round(reward / max(risk, 0.01), 2)
        potential_profit_pct = round((target - entry) / entry * 100, 2)
        potential_loss_pct = round((entry - stop_loss) / entry * 100, 2)

        return {
            "action": "BUY",
            "entry_price": entry,
            "target_price": target,
            "stop_loss": stop_loss,
            "risk_reward": rr_ratio,
            "potential_profit_pct": potential_profit_pct,
            "potential_loss_pct": potential_loss_pct,
        }
    else:
        return {
            "action": "AVOID",
            "entry_price": None,
            "target_price": None,
            "stop_loss": None,
            "risk_reward": 0,
            "potential_profit_pct": 0,
            "potential_loss_pct": 0,
        }
