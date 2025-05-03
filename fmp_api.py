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
def get_price_data(ticker: str, start_date: str, end_date: str):
    """
    지정된 기간의 일별 주가 데이터(OHLCV)를 FMP API로부터 가져옵니다.
    Args:
        ticker (str): 주식 티커
        start_date (str): 시작일 (YYYY-MM-DD 형식)
        end_date (str): 종료일 (YYYY-MM-DD 형식)
    Returns:
        list: FMP API의 'historical' 데이터를 담은 리스트 (없으면 빈 리스트)
    """
    logging.info(f"fmp_api: Requesting historical price for {ticker} from {start_date} to {end_date}")
    try:
        # _request 함수에 "from"과 "to" 파라미터를 정확히 전달
        data = _request(f"historical-price-full/{ticker}", {"from": start_date, "to": end_date})
        # FMP 응답 구조에 따라 'historical' 키 확인 (V3 기준)
        return data.get("historical", [])
    except requests.exceptions.RequestException as req_err:
        logging.error(f"FMP API historical price request failed for {ticker}: {req_err}")
        return [] # 오류 시 빈 리스트 반환
    except Exception as e:
        logging.error(f"Error processing historical price for {ticker}: {e}")
        return []
    
# get_intraday_data 함수도 추가되었는지 확인 (app.py 기술 분석 탭용)
# @st.cache_data(ttl=3600) # 필요시 캐싱
def get_intraday_data(ticker: str, interval: str, from_date: str, to_date: str):
    """지정된 기간 및 간격의 분봉/시간봉 데이터를 FMP API로부터 가져옵니다."""
    logging.info(f"fmp_api: Requesting intraday price for {ticker} ({interval}) from {from_date} to {to_date}")
    try:
        # FMP V3/V4 문서 확인 후 올바른 엔드포인트 및 파라미터 사용
        # 예시: V4 '/historical-chart/{interval}/{ticker}'
        # BASE_URL 이 V3 이면 URL 수정 필요할 수 있음: BASE_URL.replace('/v3', '/v4')
        data = _request(f"historical-chart/{interval}/{ticker}", {"from": from_date, "to": to_date})
        return data if isinstance(data, list) else []
    except requests.exceptions.RequestException as req_err:
        logging.error(f"FMP API intraday request failed for {ticker} ({interval}): {req_err}")
        return []
    except Exception as e:
        logging.error(f"Error processing intraday price for {ticker} ({interval}): {e}")
        return []

@st.cache_data(ttl=86400)
def get_profile(ticker: str):
    data = _request(f"profile/{ticker}")
    return data[0] if data else {}

@st.cache_data(ttl=1800)
def get_news(ticker: str, limit: int = 10):
    return _request("stock_news", {"tickers": ticker, "limit": limit})

@st.cache_data(ttl=3600)
def get_income_statement(ticker: str, limit: int = 4):
    return _request(f"income-statement/{ticker}", {"period": "quarter", "limit": limit})

@st.cache_data(ttl=3600)
def get_balance_sheet(ticker: str, limit: int = 4):
    return _request(f"balance-sheet-statement/{ticker}", {"period": "quarter", "limit": limit})

@st.cache_data(ttl=1800)
def get_quote_bulk(symbols: list[str]):
    symbols_str = ",".join(symbols)
    return _request(f"quote/{symbols_str}")

