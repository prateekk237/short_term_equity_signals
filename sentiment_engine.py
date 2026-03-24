"""
News Sentiment Engine + NVIDIA NIM API Integration.
Fetches Indian market news from RSS feeds and uses NIM for AI-powered analysis.
"""

import json
import requests
import feedparser
from datetime import datetime, timedelta
from textblob import TextBlob


# ════════════════════════════════════════════════════════════════════════════
#  NEWS FETCHING
# ════════════════════════════════════════════════════════════════════════════

NEWS_RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.livemint.com/rss/markets",
]


def fetch_news_from_rss(max_articles: int = 50) -> list:
    """Fetch latest market news from Indian financial RSS feeds."""
    articles = []
    for feed_url in NEWS_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_articles // len(NEWS_RSS_FEEDS)]:
                published = ""
                if hasattr(entry, "published"):
                    published = entry.published
                elif hasattr(entry, "updated"):
                    published = entry.updated

                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", ""))[:500],
                    "link": entry.get("link", ""),
                    "published": published,
                    "source": feed_url.split("/")[2],
                })
        except Exception:
            continue
    return articles


def basic_sentiment_analysis(text: str) -> dict:
    """Basic sentiment using TextBlob as fallback when NIM is not available."""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        if polarity > 0.2:
            label = "POSITIVE"
        elif polarity < -0.2:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {
            "label": label,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
        }
    except Exception:
        return {"label": "NEUTRAL", "polarity": 0, "subjectivity": 0}


def get_news_sentiment_for_stock(stock_name: str, articles: list) -> dict:
    """Filter and analyze news sentiment for a specific stock."""
    keywords = stock_name.lower().split()
    relevant = []

    for article in articles:
        text = (article.get("title", "") + " " + article.get("summary", "")).lower()
        if any(kw in text for kw in keywords if len(kw) > 3):
            sentiment = basic_sentiment_analysis(article["title"] + " " + article["summary"])
            relevant.append({**article, "sentiment": sentiment})

    if not relevant:
        return {
            "article_count": 0,
            "avg_sentiment": 0,
            "sentiment_label": "NO NEWS",
            "articles": [],
        }

    avg_polarity = sum(a["sentiment"]["polarity"] for a in relevant) / len(relevant)

    if avg_polarity > 0.15:
        label = "BULLISH"
    elif avg_polarity < -0.15:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "article_count": len(relevant),
        "avg_sentiment": round(avg_polarity, 3),
        "sentiment_label": label,
        "articles": relevant[:5],
    }


# ════════════════════════════════════════════════════════════════════════════
#  NVIDIA NIM API INTEGRATION
# ════════════════════════════════════════════════════════════════════════════

NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def call_nvidia_nim(api_key: str, prompt: str, system_prompt: str = "", model: str = "meta/llama-3.1-70b-instruct") -> str:
    """Call NVIDIA NIM API for AI analysis."""
    if not api_key:
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    try:
        response = requests.post(NIM_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[NIM API Error: {str(e)}]"


def nim_analyze_stock(api_key: str, stock_data: dict) -> dict:
    """Use NVIDIA NIM to perform AI analysis on a stock's technical and news data."""
    if not api_key:
        return {"ai_analysis": "NIM API key not configured", "ai_confidence_modifier": 0}

    system_prompt = """You are an expert Indian stock market analyst specializing in short-term 
(5-day) equity trading on NSE. You analyze technical indicators, volume patterns, support/resistance 
levels, and news sentiment to give precise trading recommendations. Be data-driven and quantitative. 
Respond ONLY in valid JSON format with these keys:
- "analysis": string (2-3 sentence analysis)
- "recommendation": "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL"
- "confidence_modifier": integer from -10 to +10 (adjustment to base confidence)
- "key_factors": list of strings (top 3 factors driving this recommendation)
- "risk_factors": list of strings (top 2 risks for this trade)
- "expected_5day_move_pct": float (expected price change in 5 trading days)"""

    prompt = f"""Analyze this Indian stock for a 5-day short-term trade:

Stock: {stock_data.get('name', 'Unknown')} ({stock_data.get('ticker', 'Unknown')})
Sector: {stock_data.get('sector', 'Unknown')}
Current Price: ₹{stock_data.get('current_price', 0):.2f}
Market Cap Type: {stock_data.get('cap', 'Unknown')}

Technical Indicators:
- RSI(14): {stock_data.get('rsi', 'N/A')}
- MACD Histogram: {stock_data.get('macd_hist', 'N/A')}
- Bollinger Band %: {stock_data.get('bb_pct', 'N/A')}
- ADX: {stock_data.get('adx', 'N/A')}
- Volume Ratio (vs 20d avg): {stock_data.get('vol_ratio', 'N/A')}

Moving Averages:
- Price vs SMA10: {stock_data.get('vs_sma10', 'N/A')}
- Price vs SMA20: {stock_data.get('vs_sma20', 'N/A')}
- Price vs SMA50: {stock_data.get('vs_sma50', 'N/A')}

Returns:
- 1-Day: {stock_data.get('daily_return', 'N/A')}%
- 5-Day: {stock_data.get('weekly_return', 'N/A')}%
- 20-Day: {stock_data.get('monthly_return', 'N/A')}%

Support/Resistance:
- Key Supports: {stock_data.get('supports', [])}
- Key Resistances: {stock_data.get('resistances', [])}
- Pivot: {stock_data.get('pivot', 'N/A')}

Technical Signal Score: {stock_data.get('signal_score', 0)}/100
News Sentiment: {stock_data.get('news_sentiment', 'N/A')}
Recent Headlines: {stock_data.get('headlines', 'None')}

Please analyze for a 5-day holding period and respond in JSON only."""

    raw_response = call_nvidia_nim(api_key, prompt, system_prompt)

    try:
        # Try to extract JSON from the response
        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        result = json.loads(json_str.strip())
        return {
            "ai_analysis": result.get("analysis", ""),
            "ai_recommendation": result.get("recommendation", "HOLD"),
            "ai_confidence_modifier": result.get("confidence_modifier", 0),
            "ai_key_factors": result.get("key_factors", []),
            "ai_risk_factors": result.get("risk_factors", []),
            "ai_expected_move": result.get("expected_5day_move_pct", 0),
        }
    except (json.JSONDecodeError, IndexError, KeyError):
        return {
            "ai_analysis": raw_response[:500] if raw_response else "Analysis unavailable",
            "ai_recommendation": "HOLD",
            "ai_confidence_modifier": 0,
            "ai_key_factors": [],
            "ai_risk_factors": [],
            "ai_expected_move": 0,
        }


def nim_market_overview(api_key: str, market_data: str) -> str:
    """Use NIM to generate an overall Indian market outlook."""
    if not api_key:
        return "Configure NVIDIA NIM API key for AI-powered market overview."

    system_prompt = """You are an expert Indian market strategist. Provide a concise weekly market 
outlook for short-term equity traders. Focus on actionable insights, sector rotation opportunities, 
and key events. Be specific about Indian market context (Nifty, Sensex, FII/DII flows, RBI policy)."""

    prompt = f"""Based on the current Indian market conditions, provide a weekly trading outlook:

{market_data}

Cover:
1. Overall market direction for the week
2. Top 3 sectors to watch
3. Key events/catalysts this week
4. Risk factors to monitor
5. Recommended trading approach (aggressive/moderate/conservative)"""

    return call_nvidia_nim(api_key, prompt, system_prompt)


def nim_portfolio_optimizer(api_key: str, capital: float, candidates: list) -> str:
    """Use NIM to optimize portfolio allocation across shortlisted candidates."""
    if not api_key or not candidates:
        return ""

    system_prompt = """You are a portfolio optimizer for short-term Indian equity trading.
Given a capital amount and stock candidates with their analysis, suggest optimal allocation.
Respond ONLY in valid JSON format with key "allocations" containing a list of objects with:
- "ticker": string
- "allocation_pct": float (percentage of capital)
- "shares": int (approximate shares to buy)
- "reason": string (1 sentence)"""

    candidates_text = "\n".join([
        f"- {c['ticker']}: ₹{c['price']:.2f}, Score: {c['score']}/100, "
        f"Signal: {c['signal']}, Target: ₹{c.get('target', 0):.2f}, "
        f"StopLoss: ₹{c.get('stop_loss', 0):.2f}, R:R={c.get('rr_ratio', 0):.1f}"
        for c in candidates
    ])

    prompt = f"""Optimize this portfolio for 5-day holding period:

Capital: ₹{capital:,.0f}
Risk Tolerance: Moderate (preserve capital, aim for 2-5% weekly return)

Candidate Stocks (already filtered for quality):
{candidates_text}

Rules:
- Maximum 5 stocks in portfolio
- No single stock > 40% of capital
- Consider risk:reward ratio and diversification
- Only allocate to stocks with score >= 65
- Keep some cash if no strong candidates (min 0%)

Respond in JSON only."""

    return call_nvidia_nim(api_key, prompt, system_prompt)
