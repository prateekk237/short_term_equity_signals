# 📊 TradeSignal India — AI-Powered Weekly Equity Signal System

A **100% FOSS** (Free & Open-Source Software) trading signal system for the **Indian stock market (NSE)** designed for **short-term (5-day) weekly trading** in equities. Uses **NVIDIA NIM API** for AI-powered deep analysis.

---

## 🎯 What It Does

Every **Monday morning**, you enter your available capital (e.g. ₹13,000). The system:

1. **Scans 130+ NSE stocks** across all sectors, market caps (Large/Mid/Small)
2. **Computes 9 technical indicator categories** with weighted scoring
3. **Fetches live news** from Economic Times, Moneycontrol, LiveMint
4. **Runs sentiment analysis** on news headlines
5. **Uses NVIDIA NIM AI** (optional) for deep stock-by-stock analysis
6. **Generates BUY signals** with entry price, target, stop-loss
7. **Allocates your capital** optimally across top picks
8. **Gives confidence scores** (only shows 90%+ when analysis is strong)

You execute the trades Monday → sell by Friday → reinvest next Monday.

---

## 🛠️ Tech Stack (All FOSS)

| Component | Library | Purpose |
|---|---|---|
| UI | **Streamlit** | Interactive web dashboard |
| Stock Data | **yfinance** | NSE OHLCV price data |
| Technicals | **ta** (Technical Analysis) | RSI, MACD, BB, ADX, etc. |
| Charts | **Plotly** | Interactive candlestick charts |
| News | **feedparser** | RSS news from Indian sources |
| Sentiment | **TextBlob** | NLP sentiment analysis |
| AI Analysis | **NVIDIA NIM API** | Deep AI-powered stock analysis |
| Data | **pandas / numpy** | Data processing |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone or download this folder, then:
cd trading_system
pip install -r requirements.txt
```

### 2. (Optional) Get NVIDIA NIM API Key

- Go to [build.nvidia.com](https://build.nvidia.com)
- Sign up for free
- Get an API key for any LLM endpoint (e.g., Llama 3.1 70B)
- The system works WITHOUT the key (uses technical + sentiment only), but NIM adds powerful AI analysis

### 3. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📖 How to Use (Weekly Workflow)

### Monday Morning:

1. **Open the app** → Enter your capital in the sidebar (e.g., ₹13,000)
2. **Paste your NIM API key** (optional but recommended)
3. **Select scan scope** → "All Stocks" for full market scan
4. **Set minimum confidence** → 65% default (raise to 90% for ultra-safe signals)
5. **Click "🚀 SCAN MARKET & GENERATE SIGNALS"**
6. **Wait 2-5 minutes** for full scan (130+ stocks)

### Review Results:

| Tab | What You See |
|---|---|
| 🎯 Trading Signals | Top BUY picks with entry/target/SL/confidence |
| 💼 Portfolio Allocation | How to split your ₹13K across stocks + order sheet |
| 📈 Stock Details | Deep dive into any stock with full chart |
| 🔬 Full Screener | All 130 stocks sortable by any metric |
| 📰 News & Sentiment | Market news with sentiment scores |

### Execute:

- Copy the **Order Sheet** from Portfolio tab
- Place **limit buy orders** at the Entry Price on your broker (Zerodha, Groww, etc.)
- Set **stop-loss** and **target** as shown
- By Friday: exit all positions (sell at target or market close)

---

## 📊 Technical Indicators Used

### Scoring System (Total: 100 points)

| Indicator | Weight | What It Measures |
|---|---|---|
| RSI (14) | 15 pts | Overbought/Oversold momentum |
| MACD | 15 pts | Trend direction & momentum |
| Moving Averages | 15 pts | SMA/EMA 10, 20, 50 cross analysis |
| Bollinger Bands | 10 pts | Volatility & mean reversion |
| Volume Analysis | 10 pts | Volume confirmation of moves |
| Stochastic | 10 pts | Short-term momentum extremes |
| ADX | 10 pts | Trend strength |
| Support/Resistance | 10 pts | Risk:Reward from key levels |
| Momentum | 5 pts | Recent price trajectory |

### Additional Analysis:
- **Support/Resistance**: Classic pivot points + swing high/low clustering
- **News Sentiment**: ±5 point modifier based on bullish/bearish news
- **NIM AI**: ±10 point modifier from deep AI analysis

---

## 🤖 NVIDIA NIM Integration

When you provide an API key, the system uses NIM for:

1. **Stock-by-Stock Analysis**: Each candidate gets AI review of all technicals + news
2. **Market Overview**: Weekly outlook with sector rotation insights
3. **Portfolio Optimization**: AI-powered capital allocation across top picks
4. **Risk Assessment**: Identifies risk factors humans might miss

### Supported NIM Models:
- `meta/llama-3.1-70b-instruct` (recommended)
- `meta/llama-3.1-8b-instruct` (faster, less accurate)
- `nvidia/llama-3.1-nemotron-70b-instruct`
- `mistralai/mixtral-8x22b-instruct-v0.1`

---

## 📁 Project Structure

```
trading_system/
├── app.py                 # Main Streamlit application
├── stock_universe.py      # 130+ NSE stocks across all sectors
├── technical_engine.py    # Technical indicators & signal scoring
├── sentiment_engine.py    # News fetching, sentiment, NIM API
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚠️ Important Disclaimer

This is an **educational and analytical tool**. It does NOT guarantee profits.

- All trading involves **risk of capital loss**
- Confidence scores represent **technical analysis strength**, not guaranteed outcomes
- Always perform your own **due diligence** before investing
- Past signals do **not** guarantee future performance
- The system ignores **extreme circumstances** (black swan events, circuit breakers, etc.)
- Use **stop-losses** religiously to protect capital

---

## 🔧 Customization

### Add More Stocks
Edit `stock_universe.py` → add any NSE stock with `.NS` suffix:
```python
"NEWSTOCK.NS": {"name": "New Stock Ltd", "sector": "IT", "cap": "Mid"},
```

### Adjust Signal Weights
Edit `technical_engine.py` → modify the `compute_signal_scores()` function weights.

### Add News Sources
Edit `sentiment_engine.py` → add RSS feed URLs to `NEWS_RSS_FEEDS` list.

---

*Built with ❤️ using 100% Free & Open-Source Software*
