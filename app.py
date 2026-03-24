"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INDIAN EQUITY TRADING SIGNAL SYSTEM — Short-Term (5-Day) Weekly Trader    ║
║  All FOSS  •  NVIDIA NIM AI  •  Full Technical Analysis  •  NSE Stocks    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
import warnings

warnings.filterwarnings("ignore")

from stock_universe import STOCK_UNIVERSE, SECTORS
from technical_engine import (
    compute_all_indicators,
    find_support_resistance,
    compute_signal_scores,
    compute_entry_exit,
)
from sentiment_engine import (
    fetch_news_from_rss,
    get_news_sentiment_for_stock,
    basic_sentiment_analysis,
    nim_analyze_stock,
    nim_market_overview,
    nim_portfolio_optimizer,
    call_nvidia_nim,
)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLING
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TradeSignal India — Weekly Equity Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700;800&display=swap');

    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.4rem;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 300;
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    .signal-card {
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, rgba(0,200,100,0.12), rgba(0,200,100,0.04));
        border-left: 4px solid #00c864;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(255,60,60,0.12), rgba(255,60,60,0.04));
        border-left: 4px solid #ff3c3c;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(255,200,0,0.12), rgba(255,200,0,0.04));
        border-left: 4px solid #ffc800;
    }

    .metric-box {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .confidence-high { color: #00e676; font-weight: 700; font-size: 1.8rem; }
    .confidence-med  { color: #ffab00; font-weight: 700; font-size: 1.8rem; }
    .confidence-low  { color: #ff5252; font-weight: 700; font-size: 1.8rem; }
    
    .ticker-tag {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #111128 100%);
    }
    
    .stProgress .st-bo { background-color: #7b2ff7; }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "scan_complete" not in st.session_state:
    st.session_state.scan_complete = False
if "news_cache" not in st.session_state:
    st.session_state.news_cache = None
if "market_overview" not in st.session_state:
    st.session_state.market_overview = None


# ════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance for an Indian stock."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker: str) -> dict:
    """Fetch stock metadata."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "marketCap": info.get("marketCap", 0),
            "industry": info.get("industry", "Unknown"),
            "dayHigh": info.get("dayHigh", 0),
            "dayLow": info.get("dayLow", 0),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0),
            "averageVolume": info.get("averageVolume", 0),
            "beta": info.get("beta", 1),
        }
    except Exception:
        return {}


def fetch_all_news():
    """Fetch and cache news articles."""
    if st.session_state.news_cache is None:
        st.session_state.news_cache = fetch_news_from_rss(max_articles=60)
    return st.session_state.news_cache


# ════════════════════════════════════════════════════════════════════════════
#  FULL STOCK ANALYSIS PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def analyze_single_stock(ticker: str, stock_info: dict, news_articles: list, nim_api_key: str = "") -> dict:
    """Run the complete analysis pipeline on a single stock."""
    name = stock_info.get("name", ticker)
    sector = stock_info.get("sector", "Unknown")
    cap = stock_info.get("cap", "Unknown")

    # 1. Fetch price data
    df = fetch_stock_data(ticker)
    if df is None or len(df) < 20:
        return None

    # 2. Compute technical indicators
    df = compute_all_indicators(df)

    # 3. Support/Resistance
    sr_levels = find_support_resistance(df)

    # 4. Signal scoring
    signal_data = compute_signal_scores(df, sr_levels)

    # 5. Entry/Exit computation
    entry_exit = compute_entry_exit(df, sr_levels, signal_data)

    # 6. News sentiment
    news_sentiment = get_news_sentiment_for_stock(name, news_articles)

    # Adjust score based on news
    news_modifier = 0
    if news_sentiment["sentiment_label"] == "BULLISH":
        news_modifier = 5
    elif news_sentiment["sentiment_label"] == "BEARISH":
        news_modifier = -5

    # 7. Latest data point
    latest = df.iloc[-1]
    current_price = latest["Close"]

    # Build stock data dict for NIM
    nim_stock_data = {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "cap": cap,
        "current_price": current_price,
        "rsi": round(latest.get("RSI_14", 50), 1),
        "macd_hist": round(latest.get("MACD_Hist", 0), 4),
        "bb_pct": round(latest.get("BB_Pct", 0.5), 3),
        "adx": round(latest.get("ADX", 20), 1),
        "vol_ratio": round(latest.get("Vol_Ratio", 1), 2),
        "vs_sma10": "Above" if current_price > latest.get("SMA_10", current_price) else "Below",
        "vs_sma20": "Above" if current_price > latest.get("SMA_20", current_price) else "Below",
        "vs_sma50": "Above" if current_price > latest.get("SMA_50", current_price) else "Below",
        "daily_return": round(latest.get("Daily_Return", 0), 2),
        "weekly_return": round(latest.get("Weekly_Return", 0), 2),
        "monthly_return": round(latest.get("Monthly_Return", 0), 2),
        "supports": sr_levels["supports"][:3],
        "resistances": sr_levels["resistances"][:3],
        "pivot": sr_levels["pivot"],
        "signal_score": signal_data["normalized_score"],
        "news_sentiment": news_sentiment["sentiment_label"],
        "headlines": "; ".join([a["title"][:80] for a in news_sentiment["articles"][:3]]) if news_sentiment["articles"] else "None",
    }

    # 8. NVIDIA NIM AI Analysis (optional)
    ai_data = {"ai_analysis": "", "ai_confidence_modifier": 0}
    if nim_api_key:
        ai_data = nim_analyze_stock(nim_api_key, nim_stock_data)

    # 9. Final confidence score
    base_confidence = signal_data["normalized_score"]
    final_confidence = base_confidence + news_modifier + ai_data.get("ai_confidence_modifier", 0)
    final_confidence = max(0, min(100, final_confidence))

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "cap": cap,
        "current_price": round(current_price, 2),
        "df": df,
        "signal_data": signal_data,
        "sr_levels": sr_levels,
        "entry_exit": entry_exit,
        "news_sentiment": news_sentiment,
        "ai_data": ai_data,
        "final_confidence": round(final_confidence, 1),
        "final_signal": signal_data["signal"],
        "technical_details": nim_stock_data,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO ALLOCATION
# ════════════════════════════════════════════════════════════════════════════

def allocate_portfolio(capital: float, candidates: list, nim_api_key: str = "") -> list:
    """Allocate capital across top candidates. Uses NIM if available, else rule-based."""

    if not candidates:
        return []

    # Sort by confidence descending
    candidates = sorted(candidates, key=lambda x: x["final_confidence"], reverse=True)

    # Filter: only 90%+ confidence for primary signals
    top_picks = [c for c in candidates if c["final_confidence"] >= 65 and "BUY" in c["final_signal"]]

    if not top_picks:
        # Relax to 60+ if nothing at 65
        top_picks = [c for c in candidates if c["final_confidence"] >= 55 and "BUY" in c["final_signal"]]

    top_picks = top_picks[:5]  # Max 5 positions

    if not top_picks:
        return []

    # Try NIM for smart allocation
    if nim_api_key and top_picks:
        nim_candidates = [{
            "ticker": c["ticker"],
            "price": c["current_price"],
            "score": c["final_confidence"],
            "signal": c["final_signal"],
            "target": c["entry_exit"].get("target_price", 0),
            "stop_loss": c["entry_exit"].get("stop_loss", 0),
            "rr_ratio": c["entry_exit"].get("risk_reward", 0),
        } for c in top_picks]

        nim_result = nim_portfolio_optimizer(nim_api_key, capital, nim_candidates)

        try:
            if nim_result:
                json_str = nim_result
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                parsed = json.loads(json_str.strip())
                allocations = parsed.get("allocations", [])
                if allocations:
                    result = []
                    for alloc in allocations:
                        match = next((c for c in top_picks if c["ticker"] == alloc["ticker"]), None)
                        if match:
                            pct = alloc.get("allocation_pct", 0)
                            amount = capital * pct / 100
                            shares = int(amount / match["current_price"])
                            result.append({
                                **match,
                                "allocation_pct": pct,
                                "allocated_amount": round(amount, 2),
                                "shares_to_buy": shares,
                                "nim_reason": alloc.get("reason", ""),
                            })
                    if result:
                        return result
        except Exception:
            pass

    # Fallback: Rule-based allocation
    total_score = sum(c["final_confidence"] for c in top_picks)
    result = []
    for c in top_picks:
        pct = (c["final_confidence"] / total_score) * 100 if total_score > 0 else 100 / len(top_picks)
        pct = min(pct, 40)  # Cap at 40%
        amount = capital * pct / 100
        shares = int(amount / c["current_price"]) if c["current_price"] > 0 else 0

        result.append({
            **c,
            "allocation_pct": round(pct, 1),
            "allocated_amount": round(amount, 2),
            "shares_to_buy": shares,
            "nim_reason": "",
        })

    # Normalize allocations
    total_alloc = sum(r["allocation_pct"] for r in result)
    if total_alloc > 100:
        for r in result:
            r["allocation_pct"] = round(r["allocation_pct"] / total_alloc * 95, 1)
            r["allocated_amount"] = round(capital * r["allocation_pct"] / 100, 2)
            r["shares_to_buy"] = int(r["allocated_amount"] / r["current_price"]) if r["current_price"] > 0 else 0

    return result


# ════════════════════════════════════════════════════════════════════════════
#  CHARTING
# ════════════════════════════════════════════════════════════════════════════

def create_stock_chart(df: pd.DataFrame, ticker: str, sr_levels: dict, entry_exit: dict) -> go.Figure:
    """Create a comprehensive candlestick chart with indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price & Indicators", "Volume", "RSI"),
    )

    # Use last 60 days for display
    display_df = df.tail(60).copy()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=display_df.index, open=display_df["Open"], high=display_df["High"],
        low=display_df["Low"], close=display_df["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff5252",
        name="Price",
    ), row=1, col=1)

    # Moving Averages
    colors = {"SMA_10": "#00bcd4", "SMA_20": "#ff9800", "SMA_50": "#e040fb"}
    for ma, color in colors.items():
        if ma in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df[ma],
                name=ma, line=dict(color=color, width=1.5),
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in display_df.columns:
        fig.add_trace(go.Scatter(
            x=display_df.index, y=display_df["BB_Upper"],
            name="BB Upper", line=dict(color="rgba(255,255,255,0.2)", dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=display_df.index, y=display_df["BB_Lower"],
            name="BB Lower", line=dict(color="rgba(255,255,255,0.2)", dash="dot"),
            fill="tonexty", fillcolor="rgba(123,47,247,0.05)",
        ), row=1, col=1)

    # Support/Resistance Lines
    for s in sr_levels.get("supports", [])[:3]:
        fig.add_hline(y=s, line_dash="dash", line_color="#00e676",
                      annotation_text=f"S: ₹{s}", row=1, col=1)
    for r in sr_levels.get("resistances", [])[:3]:
        fig.add_hline(y=r, line_dash="dash", line_color="#ff5252",
                      annotation_text=f"R: ₹{r}", row=1, col=1)

    # Entry/Exit markers
    if entry_exit.get("action") == "BUY" and entry_exit.get("entry_price"):
        fig.add_hline(y=entry_exit["entry_price"], line_dash="solid",
                      line_color="#00d2ff", line_width=2,
                      annotation_text=f"Entry: ₹{entry_exit['entry_price']}", row=1, col=1)
        if entry_exit.get("target_price"):
            fig.add_hline(y=entry_exit["target_price"], line_dash="solid",
                          line_color="#00e676", line_width=2,
                          annotation_text=f"Target: ₹{entry_exit['target_price']}", row=1, col=1)
        if entry_exit.get("stop_loss"):
            fig.add_hline(y=entry_exit["stop_loss"], line_dash="solid",
                          line_color="#ff5252", line_width=2,
                          annotation_text=f"SL: ₹{entry_exit['stop_loss']}", row=1, col=1)

    # Volume
    vol_colors = ["#00e676" if c >= o else "#ff5252"
                  for c, o in zip(display_df["Close"], display_df["Open"])]
    fig.add_trace(go.Bar(
        x=display_df.index, y=display_df["Volume"],
        marker_color=vol_colors, name="Volume", opacity=0.7,
    ), row=2, col=1)

    if "Vol_SMA_20" in display_df.columns:
        fig.add_trace(go.Scatter(
            x=display_df.index, y=display_df["Vol_SMA_20"],
            name="Vol SMA20", line=dict(color="#ff9800", width=1),
        ), row=2, col=1)

    # RSI
    if "RSI_14" in display_df.columns:
        fig.add_trace(go.Scatter(
            x=display_df.index, y=display_df["RSI_14"],
            name="RSI 14", line=dict(color="#7b2ff7", width=2),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,82,82,0.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,230,118,0.5)", row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(123,47,247,0.05)", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=650,
        title=dict(text=f"{ticker}", font=dict(size=18, family="JetBrains Mono")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=60, b=30),
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    return fig


def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a gauge chart for confidence score."""
    if confidence >= 90:
        bar_color = "#00e676"
    elif confidence >= 70:
        bar_color = "#00d2ff"
    elif confidence >= 50:
        bar_color = "#ffab00"
    else:
        bar_color = "#ff5252"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 36, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10}},
            "bar": {"color": bar_color, "thickness": 0.8},
            "bgcolor": "rgba(255,255,255,0.05)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 30], "color": "rgba(255,82,82,0.15)"},
                {"range": [30, 60], "color": "rgba(255,171,0,0.15)"},
                {"range": [60, 80], "color": "rgba(0,210,255,0.15)"},
                {"range": [80, 100], "color": "rgba(0,230,118,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "value": 90,
            },
        },
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def create_allocation_pie(allocations: list, capital: float) -> go.Figure:
    """Create a pie chart showing portfolio allocation."""
    labels = [a["ticker"].replace(".NS", "") for a in allocations]
    values = [a["allocated_amount"] for a in allocations]

    remaining = capital - sum(values)
    if remaining > 0:
        labels.append("Cash Reserve")
        values.append(remaining)

    colors = ["#00e676", "#00d2ff", "#7b2ff7", "#ff9800", "#e040fb", "#607d8b"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(labels)]),
        textfont=dict(family="JetBrains Mono", size=13),
        textinfo="label+percent",
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[dict(
            text=f"₹{capital:,.0f}",
            x=0.5, y=0.5, font_size=18,
            font_family="JetBrains Mono",
            showarrow=False, font_color="white",
        )],
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="main-title">📊 TradeSignal</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Indian Equity • Weekly Signals</p>', unsafe_allow_html=True)

    st.divider()

    # Capital Input
    st.markdown("#### 💰 Your Capital")
    capital = st.number_input(
        "Enter total capital (₹)",
        min_value=1000,
        max_value=10000000,
        value=13000,
        step=1000,
        format="%d",
        help="Amount available for this week's trades"
    )

    st.divider()

    # NVIDIA NIM API
    st.markdown("#### 🤖 NVIDIA NIM API")
    nim_api_key = st.text_input(
        "NIM API Key",
        type="password",
        help="Get your free API key from build.nvidia.com"
    )

    nim_model = st.selectbox(
        "NIM Model",
        ["meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct",
         "nvidia/llama-3.1-nemotron-70b-instruct", "mistralai/mixtral-8x22b-instruct-v0.1"],
        index=0,
    )

    st.divider()

    # Scan Configuration
    st.markdown("#### ⚙️ Scan Settings")
    scan_scope = st.selectbox(
        "Stock Universe",
        ["All Stocks (~130)", "Large Cap Only (~50)", "Mid Cap Only (~50)",
         "Small Cap Only (~30)", "Custom Sector"],
    )

    selected_sectors = []
    if scan_scope == "Custom Sector":
        selected_sectors = st.multiselect("Select Sectors", sorted(SECTORS))

    min_confidence = st.slider("Min Confidence for Signal", 50, 95, 65, 5,
                               help="Only show signals above this confidence")

    max_positions = st.slider("Max Positions", 1, 10, 5,
                              help="Maximum stocks in portfolio")

    st.divider()

    # Run Scan Button
    run_scan = st.button("🚀 SCAN MARKET & GENERATE SIGNALS", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════════════════
#  STOCK FILTERING
# ════════════════════════════════════════════════════════════════════════════

def get_filtered_universe(scope: str, sectors: list) -> dict:
    """Filter stock universe based on user selection."""
    universe = STOCK_UNIVERSE.copy()

    if scope == "Large Cap Only (~50)":
        return {k: v for k, v in universe.items() if v["cap"] == "Large"}
    elif scope == "Mid Cap Only (~50)":
        return {k: v for k, v in universe.items() if v["cap"] == "Mid"}
    elif scope == "Small Cap Only (~30)":
        return {k: v for k, v in universe.items() if v["cap"] == "Small"}
    elif scope == "Custom Sector" and sectors:
        return {k: v for k, v in universe.items() if v["sector"] in sectors}
    return universe


# ════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">📊 TradeSignal India</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Weekly Equity Signal System — Short-Term 5-Day Trading on NSE</p>', unsafe_allow_html=True)

# Top-level info
today = datetime.now()
monday = today - timedelta(days=today.weekday())
friday = monday + timedelta(days=4)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("📅 Trading Week", f"{monday.strftime('%d %b')} — {friday.strftime('%d %b %Y')}")
with col_info2:
    st.metric("💰 Capital", f"₹{capital:,.0f}")
with col_info3:
    st.metric("🎯 Min Confidence", f"{min_confidence}%")
with col_info4:
    st.metric("🤖 NIM AI", "✅ Active" if nim_api_key else "❌ Off")

st.divider()


# ════════════════════════════════════════════════════════════════════════════
#  MARKET SCANNER
# ════════════════════════════════════════════════════════════════════════════

if run_scan:
    universe = get_filtered_universe(scan_scope, selected_sectors)
    total_stocks = len(universe)

    st.markdown(f"### 🔍 Scanning {total_stocks} Stocks...")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()

    # Fetch news first
    status_text.text("📰 Fetching latest market news...")
    news_articles = fetch_all_news()

    # Scan all stocks
    all_results = []
    errors = 0

    for idx, (ticker, info) in enumerate(universe.items()):
        progress = (idx + 1) / total_stocks
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {info['name']} ({ticker}) ... [{idx+1}/{total_stocks}]")

        try:
            result = analyze_single_stock(ticker, info, news_articles, nim_api_key)
            if result:
                all_results.append(result)
        except Exception as e:
            errors += 1
            continue

        # Small delay to avoid rate limiting
        if idx % 10 == 0 and idx > 0:
            time.sleep(0.5)

    progress_bar.progress(1.0)
    status_text.text(f"✅ Scan complete! Analyzed {len(all_results)}/{total_stocks} stocks ({errors} errors)")

    # Store results
    st.session_state.scan_results = all_results
    st.session_state.scan_complete = True

    # Generate market overview with NIM
    if nim_api_key:
        status_text.text("🤖 Generating AI market overview...")
        top_scores = sorted(all_results, key=lambda x: x["final_confidence"], reverse=True)[:10]
        market_summary = f"""
Top scoring stocks: {', '.join([f'{r["ticker"]}({r["final_confidence"]:.0f}%)' for r in top_scores[:5]])}
Sectors showing strength: {', '.join(set(r['sector'] for r in top_scores[:5]))}
Average confidence of top 10: {np.mean([r['final_confidence'] for r in top_scores]):.1f}%
Stocks with BUY signal: {len([r for r in all_results if 'BUY' in r['final_signal']])}
Stocks with SELL signal: {len([r for r in all_results if 'SELL' in r['final_signal']])}
"""
        overview = nim_market_overview(nim_api_key, market_summary)
        st.session_state.market_overview = overview

    status_text.empty()


# ════════════════════════════════════════════════════════════════════════════
#  DISPLAY RESULTS
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.scan_complete and st.session_state.scan_results:
    results = st.session_state.scan_results

    # Market Overview
    if st.session_state.market_overview:
        with st.expander("🤖 AI Market Overview (NVIDIA NIM)", expanded=True):
            st.markdown(st.session_state.market_overview)

    # ── TABS ──────────────────────────────────────────────────────────────
    tab_signals, tab_portfolio, tab_details, tab_screener, tab_news = st.tabs([
        "🎯 Trading Signals", "💼 Portfolio Allocation", "📈 Stock Details",
        "🔬 Full Screener", "📰 News & Sentiment"
    ])

    # ── TAB 1: TRADING SIGNALS ───────────────────────────────────────────
    with tab_signals:
        st.markdown("### 🎯 This Week's Trading Signals")
        st.markdown(f"*Showing signals with ≥{min_confidence}% confidence for 5-day holding period*")

        buy_signals = [r for r in results
                       if "BUY" in r["final_signal"] and r["final_confidence"] >= min_confidence]
        buy_signals = sorted(buy_signals, key=lambda x: x["final_confidence"], reverse=True)

        if not buy_signals:
            st.warning(f"⚠️ No stocks meet the {min_confidence}% confidence threshold this week. "
                       "Consider lowering the minimum confidence or expanding the scan scope.")
            # Show top candidates anyway
            st.markdown("**Top candidates (below threshold):**")
            top_all = sorted([r for r in results if "BUY" in r["final_signal"]],
                             key=lambda x: x["final_confidence"], reverse=True)[:5]
            for r in top_all:
                st.markdown(f"- {r['name']} ({r['ticker']}): {r['final_confidence']:.0f}% confidence")
        else:
            for i, sig in enumerate(buy_signals[:max_positions]):
                signal_class = "signal-buy" if "BUY" in sig["final_signal"] else "signal-hold"
                entry = sig["entry_exit"]

                st.markdown(f"""
                <div class="signal-card {signal_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span class="ticker-tag">{sig["ticker"].replace('.NS', '')}</span>
                            &nbsp;—&nbsp; {sig["name"]}
                            <span style="color:#888; font-size:0.85rem;">&nbsp;|&nbsp;{sig["sector"]} • {sig["cap"]} Cap</span>
                        </div>
                        <div style="text-align:right;">
                            <span class="{'confidence-high' if sig['final_confidence'] >= 90 else 'confidence-med' if sig['final_confidence'] >= 70 else 'confidence-low'}">
                                {sig['final_confidence']:.0f}%
                            </span>
                            <br><span style="color:#888; font-size:0.75rem;">CONFIDENCE</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                with col_a:
                    st.metric("Current Price", f"₹{sig['current_price']:.2f}")
                with col_b:
                    st.metric("📥 Entry Price", f"₹{entry.get('entry_price', 0):.2f}" if entry.get('entry_price') else "N/A")
                with col_c:
                    st.metric("🎯 Target", f"₹{entry.get('target_price', 0):.2f}" if entry.get('target_price') else "N/A",
                              delta=f"+{entry.get('potential_profit_pct', 0):.1f}%" if entry.get('potential_profit_pct') else None)
                with col_d:
                    st.metric("🛑 Stop Loss", f"₹{entry.get('stop_loss', 0):.2f}" if entry.get('stop_loss') else "N/A",
                              delta=f"-{entry.get('potential_loss_pct', 0):.1f}%" if entry.get('potential_loss_pct') else None,
                              delta_color="inverse")
                with col_e:
                    st.metric("⚖️ Risk:Reward", f"{entry.get('risk_reward', 0):.1f}")

                # AI Analysis
                if sig["ai_data"].get("ai_analysis"):
                    with st.expander(f"🤖 AI Analysis — {sig['ticker'].replace('.NS', '')}"):
                        st.write(sig["ai_data"]["ai_analysis"])
                        if sig["ai_data"].get("ai_key_factors"):
                            st.markdown("**Key Factors:** " + " • ".join(sig["ai_data"]["ai_key_factors"]))
                        if sig["ai_data"].get("ai_risk_factors"):
                            st.markdown("**Risks:** " + " • ".join(sig["ai_data"]["ai_risk_factors"]))

                # Technical breakdown
                with st.expander(f"📊 Technical Breakdown — {sig['ticker'].replace('.NS', '')}"):
                    details = sig["signal_data"]["details"]
                    score_df = pd.DataFrame([
                        {"Indicator": k, "Score": v["score"], "Max": v["max"], "Reason": v["reason"]}
                        for k, v in details.items()
                    ])
                    st.dataframe(score_df, hide_index=True, use_container_width=True)

                    # S/R Levels
                    sr = sig["sr_levels"]
                    col_sr1, col_sr2 = st.columns(2)
                    with col_sr1:
                        st.markdown("**Support Levels:** " + ", ".join([f"₹{s}" for s in sr["supports"][:3]]) if sr["supports"] else "None")
                    with col_sr2:
                        st.markdown("**Resistance Levels:** " + ", ".join([f"₹{r}" for r in sr["resistances"][:3]]) if sr["resistances"] else "None")

                st.divider()

    # ── TAB 2: PORTFOLIO ALLOCATION ──────────────────────────────────────
    with tab_portfolio:
        st.markdown("### 💼 Recommended Portfolio Allocation")
        st.markdown(f"*Capital: ₹{capital:,.0f} | Max {max_positions} positions | 5-day holding*")

        buy_signals = [r for r in results
                       if "BUY" in r["final_signal"] and r["final_confidence"] >= min_confidence]
        buy_signals = sorted(buy_signals, key=lambda x: x["final_confidence"], reverse=True)

        allocations = allocate_portfolio(capital, buy_signals, nim_api_key)

        if not allocations:
            st.warning("No qualifying stocks for allocation. Lower the confidence threshold or expand scan scope.")
        else:
            # Summary metrics
            total_invested = sum(a["allocated_amount"] for a in allocations)
            cash_reserve = capital - total_invested
            avg_confidence = np.mean([a["final_confidence"] for a in allocations])
            max_profit = sum(
                a["shares_to_buy"] * (a["entry_exit"].get("target_price", a["current_price"]) - a["current_price"])
                for a in allocations
            )
            max_loss = sum(
                a["shares_to_buy"] * (a["current_price"] - a["entry_exit"].get("stop_loss", a["current_price"]))
                for a in allocations
            )

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.metric("Total Invested", f"₹{total_invested:,.0f}")
            with col_p2:
                st.metric("Cash Reserve", f"₹{cash_reserve:,.0f}")
            with col_p3:
                st.metric("Max Potential Profit", f"₹{max_profit:,.0f}",
                          delta=f"+{max_profit/capital*100:.1f}%")
            with col_p4:
                st.metric("Max Risk (SL Hit)", f"₹{max_loss:,.0f}",
                          delta=f"-{max_loss/capital*100:.1f}%", delta_color="inverse")

            # Pie chart and allocation table side by side
            col_chart, col_table = st.columns([1, 1.5])

            with col_chart:
                fig_pie = create_allocation_pie(allocations, capital)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_table:
                alloc_data = []
                for a in allocations:
                    alloc_data.append({
                        "Stock": f"{a['name']} ({a['ticker'].replace('.NS', '')})",
                        "Sector": a["sector"],
                        "Price": f"₹{a['current_price']:.2f}",
                        "Shares": a["shares_to_buy"],
                        "Amount": f"₹{a['allocated_amount']:,.0f}",
                        "Alloc%": f"{a['allocation_pct']:.1f}%",
                        "Target": f"₹{a['entry_exit'].get('target_price', 0):.2f}",
                        "SL": f"₹{a['entry_exit'].get('stop_loss', 0):.2f}",
                        "Confidence": f"{a['final_confidence']:.0f}%",
                    })
                st.dataframe(pd.DataFrame(alloc_data), hide_index=True, use_container_width=True)

            # Detailed order sheet
            st.markdown("#### 📋 Order Sheet (Copy for Trading)")
            st.markdown("*Execute these orders at market open on Monday:*")

            for a in allocations:
                st.code(
                    f"{'BUY':6s} {a['ticker'].replace('.NS', ''):15s} | "
                    f"Qty: {a['shares_to_buy']:4d} | "
                    f"Entry: ₹{a['entry_exit'].get('entry_price', a['current_price']):>10.2f} | "
                    f"Target: ₹{a['entry_exit'].get('target_price', 0):>10.2f} | "
                    f"SL: ₹{a['entry_exit'].get('stop_loss', 0):>10.2f} | "
                    f"Confidence: {a['final_confidence']:.0f}%",
                    language=None
                )

    # ── TAB 3: STOCK DETAILS ─────────────────────────────────────────────
    with tab_details:
        st.markdown("### 📈 Detailed Stock Analysis")

        # Stock selector from results
        stock_options = {f"{r['name']} ({r['ticker'].replace('.NS', '')}) — {r['final_confidence']:.0f}%": r
                        for r in sorted(results, key=lambda x: x["final_confidence"], reverse=True)}

        selected_stock_key = st.selectbox("Select a stock for detailed analysis", list(stock_options.keys()))

        if selected_stock_key:
            stock = stock_options[selected_stock_key]

            # Header metrics
            col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
            with col_h1:
                st.metric("Price", f"₹{stock['current_price']:.2f}")
            with col_h2:
                daily_ret = stock["technical_details"].get("daily_return", 0)
                st.metric("Today", f"{daily_ret:+.2f}%",
                          delta=f"{daily_ret:+.2f}%")
            with col_h3:
                weekly_ret = stock["technical_details"].get("weekly_return", 0)
                st.metric("5-Day", f"{weekly_ret:+.2f}%")
            with col_h4:
                st.metric("Signal", stock["final_signal"])
            with col_h5:
                st.metric("Confidence", f"{stock['final_confidence']:.1f}%")

            # Chart
            fig = create_stock_chart(stock["df"], stock["ticker"], stock["sr_levels"], stock["entry_exit"])
            st.plotly_chart(fig, use_container_width=True)

            # Confidence gauge and technical details
            col_gauge, col_tech = st.columns([1, 2])

            with col_gauge:
                st.markdown("#### Confidence Score")
                fig_gauge = create_confidence_gauge(stock["final_confidence"])
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Key levels
                st.markdown("#### Key Levels")
                sr = stock["sr_levels"]
                levels_data = []
                for s in sr.get("supports", [])[:3]:
                    levels_data.append({"Type": "🟢 Support", "Price": f"₹{s:.2f}",
                                        "Distance": f"{(stock['current_price']-s)/stock['current_price']*100:.1f}%"})
                levels_data.append({"Type": "⚪ Pivot", "Price": f"₹{sr.get('pivot', 0):.2f}", "Distance": "—"})
                for r in sr.get("resistances", [])[:3]:
                    levels_data.append({"Type": "🔴 Resistance", "Price": f"₹{r:.2f}",
                                        "Distance": f"{(r-stock['current_price'])/stock['current_price']*100:.1f}%"})
                st.dataframe(pd.DataFrame(levels_data), hide_index=True, use_container_width=True)

            with col_tech:
                st.markdown("#### Technical Indicator Scores")
                details = stock["signal_data"]["details"]

                # Create a horizontal bar chart of scores
                indicators = list(details.keys())
                score_values = [details[k]["score"] for k in indicators]
                max_values = [details[k]["max"] for k in indicators]
                reasons = [details[k]["reason"] for k in indicators]

                colors = ["#00e676" if s > 0 else "#ff5252" if s < 0 else "#ffab00" for s in score_values]

                fig_scores = go.Figure()
                fig_scores.add_trace(go.Bar(
                    y=indicators, x=score_values,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{s:+.0f}/{m}" for s, m in zip(score_values, max_values)],
                    textposition="outside",
                    hovertext=reasons,
                ))

                fig_scores.update_layout(
                    template="plotly_dark",
                    height=350,
                    xaxis_title="Score",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,10,26,0.8)",
                    margin=dict(l=100, r=60, t=10, b=30),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_scores, use_container_width=True)

                # Moving Averages Table
                st.markdown("#### Moving Averages")
                latest = stock["df"].iloc[-1]
                ma_data = []
                for period in [5, 10, 20, 50]:
                    for ma_type in ["SMA", "EMA"]:
                        key = f"{ma_type}_{period}"
                        if key in latest and not pd.isna(latest[key]):
                            val = latest[key]
                            diff_pct = (stock["current_price"] - val) / val * 100
                            ma_data.append({
                                "MA": f"{ma_type} {period}",
                                "Value": f"₹{val:.2f}",
                                "Price Diff": f"{diff_pct:+.2f}%",
                                "Position": "✅ Above" if diff_pct > 0 else "❌ Below",
                            })
                if ma_data:
                    st.dataframe(pd.DataFrame(ma_data), hide_index=True, use_container_width=True)

            # AI Analysis
            if stock["ai_data"].get("ai_analysis"):
                st.markdown("#### 🤖 NVIDIA NIM AI Analysis")
                st.info(stock["ai_data"]["ai_analysis"])

                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    if stock["ai_data"].get("ai_key_factors"):
                        st.markdown("**Key Factors:**")
                        for f in stock["ai_data"]["ai_key_factors"]:
                            st.markdown(f"- ✅ {f}")
                with col_ai2:
                    if stock["ai_data"].get("ai_risk_factors"):
                        st.markdown("**Risk Factors:**")
                        for f in stock["ai_data"]["ai_risk_factors"]:
                            st.markdown(f"- ⚠️ {f}")

    # ── TAB 4: FULL SCREENER ─────────────────────────────────────────────
    with tab_screener:
        st.markdown("### 🔬 Full Market Screener Results")

        # Filter controls
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_signal = st.multiselect("Signal Filter", ["STRONG BUY", "BUY", "HOLD / NEUTRAL", "SELL", "STRONG SELL"],
                                           default=["STRONG BUY", "BUY"])
        with col_f2:
            filter_cap = st.multiselect("Cap Filter", ["Large", "Mid", "Small"],
                                        default=["Large", "Mid", "Small"])
        with col_f3:
            sort_by = st.selectbox("Sort By", ["Confidence ↓", "Price ↓", "Daily Return ↓", "Volume Ratio ↓"])

        # Build screener DataFrame
        screener_data = []
        for r in results:
            if r["final_signal"] in filter_signal and r["cap"] in filter_cap:
                td = r["technical_details"]
                screener_data.append({
                    "Ticker": r["ticker"].replace(".NS", ""),
                    "Name": r["name"],
                    "Sector": r["sector"],
                    "Cap": r["cap"],
                    "Price": r["current_price"],
                    "Signal": r["final_signal"],
                    "Confidence": r["final_confidence"],
                    "RSI": td.get("rsi", 0),
                    "MACD Hist": td.get("macd_hist", 0),
                    "Vol Ratio": td.get("vol_ratio", 0),
                    "1D Return%": td.get("daily_return", 0),
                    "5D Return%": td.get("weekly_return", 0),
                    "20D Return%": td.get("monthly_return", 0),
                    "vs SMA10": td.get("vs_sma10", ""),
                    "vs SMA50": td.get("vs_sma50", ""),
                    "News": r["news_sentiment"]["sentiment_label"],
                })

        if screener_data:
            screener_df = pd.DataFrame(screener_data)

            # Sort
            sort_map = {
                "Confidence ↓": ("Confidence", False),
                "Price ↓": ("Price", False),
                "Daily Return ↓": ("1D Return%", False),
                "Volume Ratio ↓": ("Vol Ratio", False),
            }
            sort_col, sort_asc = sort_map[sort_by]
            screener_df = screener_df.sort_values(sort_col, ascending=sort_asc)

            st.dataframe(
                screener_df,
                hide_index=True,
                use_container_width=True,
                height=600,
                column_config={
                    "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f%%"),
                    "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "1D Return%": st.column_config.NumberColumn("1D Ret%", format="%.2f%%"),
                    "5D Return%": st.column_config.NumberColumn("5D Ret%", format="%.2f%%"),
                    "20D Return%": st.column_config.NumberColumn("20D Ret%", format="%.2f%%"),
                },
            )

            st.markdown(f"*Showing {len(screener_df)} of {len(results)} stocks*")

            # Sector heatmap
            st.markdown("#### Sector Confidence Heatmap")
            sector_avg = screener_df.groupby("Sector")["Confidence"].mean().sort_values(ascending=False)
            fig_heat = px.bar(
                x=sector_avg.values, y=sector_avg.index,
                orientation="h",
                color=sector_avg.values,
                color_continuous_scale=["#ff5252", "#ffab00", "#00e676"],
                labels={"x": "Avg Confidence %", "y": "Sector"},
            )
            fig_heat.update_layout(
                template="plotly_dark",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,10,26,0.8)",
                showlegend=False,
                margin=dict(l=120, r=30, t=10, b=30),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── TAB 5: NEWS & SENTIMENT ──────────────────────────────────────────
    with tab_news:
        st.markdown("### 📰 Market News & Sentiment Analysis")

        news_articles = st.session_state.news_cache or []

        if not news_articles:
            st.info("News will be fetched during market scan. Click 'Scan Market' to load.")
        else:
            # Overall market sentiment
            all_sentiments = [basic_sentiment_analysis(a["title"]) for a in news_articles]
            avg_pol = np.mean([s["polarity"] for s in all_sentiments])

            col_n1, col_n2, col_n3 = st.columns(3)
            with col_n1:
                st.metric("Total Articles", len(news_articles))
            with col_n2:
                st.metric("Market Sentiment",
                          "🟢 Bullish" if avg_pol > 0.1 else "🔴 Bearish" if avg_pol < -0.1 else "🟡 Neutral")
            with col_n3:
                st.metric("Sentiment Score", f"{avg_pol:.3f}")

            # Sentiment distribution
            labels_count = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
            for s in all_sentiments:
                labels_count[s["label"]] += 1

            fig_sent = go.Figure(go.Bar(
                x=list(labels_count.keys()),
                y=list(labels_count.values()),
                marker_color=["#00e676", "#ffab00", "#ff5252"],
            ))
            fig_sent.update_layout(
                template="plotly_dark", height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,10,26,0.8)",
                margin=dict(l=30, r=30, t=10, b=30),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            # News articles list
            st.markdown("#### Latest Headlines")
            for article in news_articles[:30]:
                sent = basic_sentiment_analysis(article["title"])
                emoji = "🟢" if sent["label"] == "POSITIVE" else "🔴" if sent["label"] == "NEGATIVE" else "🟡"
                st.markdown(
                    f"{emoji} **{article['title'][:120]}**  \n"
                    f"<small style='color:#888;'>{article.get('source', '')} | {article.get('published', '')}</small>",
                    unsafe_allow_html=True
                )

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome! Here's how to use TradeSignal India:
    
    **Step 1:** Enter your capital in the sidebar (e.g., ₹13,000)
    
    **Step 2:** (Optional) Add your NVIDIA NIM API key for AI-powered analysis  
    → Get a free key at [build.nvidia.com](https://build.nvidia.com)
    
    **Step 3:** Configure scan settings (stock universe, confidence threshold)
    
    **Step 4:** Click **"🚀 SCAN MARKET & GENERATE SIGNALS"** to analyze the entire Indian market
    
    **Step 5:** Review signals in the **Trading Signals** tab and execute orders from the **Portfolio Allocation** tab
    
    ---
    
    #### 🔍 What This System Analyzes:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📊 Technical Indicators**
        - RSI (7, 14 period)
        - MACD with histogram
        - Bollinger Bands
        - Stochastic Oscillator
        - ADX Trend Strength
        - Williams %R
        """)
    with col2:
        st.markdown("""
        **📈 Price Analysis**
        - SMA/EMA (10, 20, 50 day)
        - Support & Resistance levels
        - Pivot Points (Classic)
        - Volume analysis & ratio
        - 1D / 5D / 20D returns
        - Candlestick patterns
        """)
    with col3:
        st.markdown("""
        **🤖 AI & Sentiment**
        - News from ET, Moneycontrol, Mint
        - TextBlob sentiment analysis
        - NVIDIA NIM deep analysis
        - AI portfolio optimization
        - Market outlook generation
        - Risk factor identification
        """)

    st.markdown("""
    ---
    #### ⚠️ Disclaimer
    *This is an educational tool for market analysis. All trading involves risk. Past signals do not 
    guarantee future performance. Always do your own due diligence before investing. The confidence 
    scores represent the system's technical analysis strength, not guaranteed outcomes.*
    """)


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    '<p style="text-align:center; color:#555; font-size:0.8rem;">'
    'TradeSignal India v1.0 — Built with 100% FOSS Tools — '
    'yfinance • ta • TextBlob • Plotly • Streamlit • NVIDIA NIM'
    '</p>',
    unsafe_allow_html=True,
)
