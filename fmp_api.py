# fmp_api.py

import os
import requests
import logging
from dotenv import load_dotenv
import streamlit as st

# 환경 변수 불러오기
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

BASE_URL = "https://financialmodelingprep.com/api/v3"

# 공통 요청 함수
def fetch_fmp(endpoint, params={}):
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY가 .env 또는 st.secrets에 설정되어 있지 않습니다.")
    try:
        params["apikey"] = FMP_API_KEY
        url = f"{BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"[FMP 요청 오류] {endpoint}: {e}")
        return None
    
@st.cache_data(ttl=3600)
def get_price_data_fmp(ticker, days=60):
    endpoint = f"historical-price-full/{ticker}"
    params = {
        "serietype": "line",
        "timeseries": days
    }
    data = fetch_fmp(endpoint, params)
    if data and "historical" in data:
        return data["historical"]
    return []

@st.cache_data(ttl=86400)
def get_fundamentals_fmp(ticker):
    endpoint = f"profile/{ticker}"
    data = fetch_fmp(endpoint)
    return data[0] if isinstance(data, list) and data else {}

@st.cache_data(ttl=1800)
def get_news_fmp(ticker, limit=10):
    endpoint = "stock_news"
    params = {"tickers": ticker, "limit": limit}
    data = fetch_fmp(endpoint, params)
    return data if data else []
