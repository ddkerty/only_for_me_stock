# fmp_api.py ─ 최종

import os, logging, requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY") or st.secrets.get("FMP_API_KEY")
BASE_URL    = "https://financialmodelingprep.com/api/v3"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "TechnutStock/1.0"})

def _request(endpoint: str, params: dict | None = None):
    if not FMP_API_KEY:
        raise EnvironmentError("FMP_API_KEY not set")
    params = params or {}
    params["apikey"] = FMP_API_KEY
    url = f"{BASE_URL}/{endpoint}"

    resp = SESSION.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

# ──────── 래퍼 함수 ────────
@st.cache_data(ttl=3600)
def get_price_data(ticker: str, days: int = 60):
    data = _request(f"historical-price-full/{ticker}",
                    {"serietype": "line", "timeseries": days})
    return data.get("historical", [])

@st.cache_data(ttl=86400)
def get_profile(ticker: str):
    data = _request(f"profile/{ticker}")
    return data[0] if data else {}

@st.cache_data(ttl=1800)
def get_news(ticker: str, limit: int = 10):
    return _request("stock_news", {"tickers": ticker, "limit": limit})
