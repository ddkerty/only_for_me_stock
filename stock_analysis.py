# stock_analysis.py (FMP Macro Data & FRED 제거 적용 버전)

import os
import logging
import pandas as pd
import numpy as np
import fmp_api # FMP API 모듈
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob # 뉴스 감성 분석용
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
# from fredapi import Fred # --- FRED API 제거 ---
import traceback
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import warnings
import locale
import re
import pandas_ta as ta
import streamlit as st # caching decorator 사용 위해 임포트

# Import specific FMP functions for clarity
from fmp_api import (
    get_income_statement, get_balance_sheet, get_profile, get_quote_bulk,
    get_price_data, get_intraday_data # 가격 데이터 함수 임포트 (fmp_api.py에 구현 가정)
)

# 경고 메시지 및 로깅 설정
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 경로 설정 ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
    logging.info(f"__file__ 없음. CWD 사용: {BASE_DIR}")

CHARTS_FOLDER = os.path.join(BASE_DIR, "charts")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
FORECAST_FOLDER = os.path.join(BASE_DIR, "forecast")

# .env 로드 (FMP 키는 fmp_api.py에서 처리)
try:
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env 로드 성공: {dotenv_path}")
    else:
        logging.info(f".env 파일 없음 (정상일 수 있음): {dotenv_path}")
except Exception as e:
    logging.error(f".env 로드 오류: {e}")

# --- FRED_API_KEY 관련 코드 제거 ---
# FMP API Key는 fmp_api.py 모듈 내에서 로드 및 관리됨

# --- 데이터 가져오기 함수들 ---
def get_fear_greed_index():
    """공포-탐욕 지수 가져오기"""
    # (기존 코드 유지)
    url = "https://api.alternative.me/fng/?limit=1&format=json&date_format=world"
    value, classification = None, None
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            value_str = data[0].get('value')
            classification_str = data[0].get('value_classification')
            if value_str is not None and classification_str is not None:
                try:
                    value = int(value_str)
                    classification = classification_str
                    logging.info(f"F&G 성공: {value} ({classification})")
                except (ValueError, TypeError):
                    logging.warning(f"F&G 값 변환 오류: {value_str}")
            else:
                logging.warning("F&G 데이터 구조 오류.")
        else:
            logging.warning("F&G 데이터 형식 오류.")
    except requests.exceptions.RequestException as e:
        logging.error(f"F&G API 요청 오류: {e}")
    except Exception as e:
        logging.error(f"F&G 처리 오류: {e}")
    return value, classification

# fmp_api.get_price_data 가 start_date, end_date 를 받도록 수정되었다고 가정
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date=None, end_date=None, period="1y"):
    """주가 데이터 가져오기 (OHLCV 포함) - fmp_api 사용"""
    # (이전 답변의 수정된 코드 유지 - start_date, end_date 계산 및 fmp_api.get_price_data 호출)
    logging.info(f"Fetching stock data for {ticker} using fmp_api...")
    try:
        if not end_date: end_dt = datetime.now().date()
        else: end_dt = pd.to_datetime(end_date).date()

        if not start_date:
            if period:
                period_map = {'d': 1, 'w': 7, 'mo': 30, 'y': 365}; match = re.match(r"(\d+)([dwmyo]+)", period.lower())
                if match: num, unit = match.groups(); delta_days = int(num) * period_map.get(unit, 30); start_dt = end_dt - timedelta(days=delta_days)
                else: start_dt = end_dt - relativedelta(years=1)
            else: start_dt = end_dt - relativedelta(years=1)
        else: start_dt = pd.to_datetime(start_date).date()

        start_date_str = start_dt.strftime('%Y-%m-%d'); end_date_str = end_dt.strftime('%Y-%m-%d')

        # --- Fetch data using CORRECTED fmp_api.get_price_data ---
        historical_data = fmp_api.get_price_data(ticker, start_date=start_date_str, end_date=end_date_str)
        # ---------------------------------------------------------

        if not historical_data: logging.warning(f"{ticker} 주가 데이터 비어있음 (fmp_api, 기간: {start_date_str}~{end_date_str})."); return None
        df = pd.DataFrame(historical_data)
        if df.empty: logging.warning(f"{ticker} 주가 데이터 변환 후 비어있음 (fmp_api)."); return None

        # --- Data Cleaning and Formatting ---
        rename_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df = df.rename(columns=rename_map)
        if 'Date' not in df.columns: logging.error(f"{ticker}: FMP 응답에 'Date' 컬럼 누락."); return None
        df['Date'] = pd.to_datetime(df['Date']); df = df.set_index('Date'); df = df.sort_index()
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']; cols_to_use = [col for col in df.columns if col in required_cols]; df = df[cols_to_use]
        for col in required_cols:
             if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
             else: df[col] = np.nan; logging.warning(f"{ticker}: 필수 컬럼 '{col}' 누락 -> NaN으로 채움.")
        initial_len = len(df); df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # 가격 데이터 없으면 제거
        if len(df) < initial_len: logging.info(f"{ticker}: 가격 데이터 NaN 값으로 {initial_len - len(df)} 행 제거")
        df = df[(df.index >= pd.to_datetime(start_date_str)) & (df.index <= pd.to_datetime(end_date_str))]
        if df.empty: logging.warning(f"{ticker}: 최종 처리 후 데이터 없음 (기간: {start_date_str}~{end_date_str})."); return None

        logging.info(f"{ticker} 주가 가져오기 성공 (fmp_api). 최종 기간: {df.index.min()}~{df.index.max()}, Shape: {df.shape}")
        return df
    except requests.exceptions.RequestException as req_err: logging.error(f"티커 '{ticker}' fmp_api 요청 실패: {req_err}"); return None
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (주가 데이터): {env_err}"); return None
    except Exception as e: logging.error(f"티커 '{ticker}' 주가 데이터 처리 실패 (fmp_api): {e}\n{traceback.format_exc()}"); return None

@st.cache_data(ttl=1800)
def get_macro_data_fmp():
    """FMP 기반 최신 매크로 지표 (VIX, 10Y, 13W, DXY)"""
    # (이전 답변의 코드 유지)
    logging.info("Fetching latest macro data from FMP (VIX, TNX, IRX, DXY)...")
    tickers = ["^VIX", "^TNX", "^IRX", "DXY"]
    try:
        quote_data = get_quote_bulk(tickers)
        if not quote_data: logging.warning("FMP 매크로 데이터(bulk quote) 없음."); return {}
        result = {}
        for item in quote_data:
            sym = item.get("symbol"); price = item.get("price")
            if sym and price is not None:
                try: result[sym] = round(float(price), 3)
                except (ValueError, TypeError): logging.warning(f"FMP 매크로 값 변환 오류: Symbol={sym}, Price={price}"); result[sym] = None
            else: logging.debug(f"FMP 매크로 데이터 누락: Symbol={sym}, Price={price}")
        macro_map = {"^VIX": "VIX", "^TNX": "US10Y", "^IRX": "US13W", "DXY": "DXY"}
        final_result = {macro_map.get(sym): val for sym, val in result.items() if sym in macro_map}
        logging.info(f"FMP 매크로 데이터 로딩 성공: {final_result}")
        return final_result
    except requests.exceptions.RequestException as req_err: logging.error(f"FMP 매크로 데이터 요청 실패: {req_err}"); return {}
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (매크로 데이터): {env_err}"); return {}
    except Exception as e: logging.error(f"FMP 매크로 데이터 처리 오류: {e}\n{traceback.format_exc()}"); return {}

# --- 기본적 분석 데이터 가져오기 ---
def format_market_cap(mc):
    # (기존 코드 유지)
    if isinstance(mc, (int, float)):
        if mc >= 1e12: return f"${mc / 1e12:.2f} T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f} B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f} M"
        else: return f"${mc:.2f}"
    return "N/A"

@st.cache_data(ttl=86400)
def get_fundamental_data(ticker):
    """FMP 기반 기업 기본 정보 가져오기"""
    # (기존 코드 유지 - PER/EPS N/A 문제는 FMP 데이터 자체일 수 있음)
    logging.info(f"{ticker}: FMP 프로필 조회...")
    try:
        data = get_profile(ticker) # Uses fmp_api.py function
        if not data or not isinstance(data, dict): logging.warning(f"{ticker}: FMP 프로필 데이터 없음."); return {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
        fundamentals = {
            "시가총액": format_market_cap(data.get("mktCap")),
            "PER": f"{data.get('pe'):.2f}" if data.get("pe") else "N/A",
            "EPS": f"{data.get('eps'):.2f}" if data.get("eps") else "N/A",
            "배당수익률": f"{data.get('lastDiv'):.2f}" if data.get("lastDiv") is not None else "N/A",
            "베타": f"{data.get('beta'):.2f}" if data.get("beta") is not None else "N/A",
            "업종": data.get("sector") or "N/A",
            "산업": data.get("industry") or "N/A",
            "요약": data.get("description") or "N/A"
        }
        logging.info(f"{ticker}: FMP 프로필 파싱 완료.")
        return fundamentals
    except requests.exceptions.RequestException as req_err: logging.error(f"FMP 프로필 요청 실패 ({ticker}): {req_err}"); return {}
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (프로필, {ticker}): {env_err}"); return {}
    except Exception as e: logging.error(f"FMP 프로필 처리 오류 ({ticker}): {e}\n{traceback.format_exc()}"); return {}

# --- FMP 기반 재무 추세 함수들 ---
@st.cache_data(ttl=3600)
def get_operating_margin_trend(ticker, num_periods=4):
    """FMP 기반 영업이익률(%) 추세"""
    # (기존 FMP 기반 코드 유지 - 데이터 없음 문제는 FMP 응답 확인 필요)
    logging.info(f"{ticker}: FMP 영업이익률 추세 조회 ({num_periods} 분기)...")
    try:
        data = get_income_statement(ticker, limit=num_periods)
        records = []
        if not data: logging.warning(f"{ticker}: FMP 손익계산서 데이터 없음 (영업이익률)"); return []
        for item in reversed(data):
            revenue = item.get("revenue"); op_income = item.get("operatingIncome"); date = item.get("date")
            if revenue is not None and op_income is not None and revenue != 0 and date:
                margin = round((op_income / revenue) * 100, 2); records.append({"Date": date, "Op Margin (%)": margin})
            else: logging.debug(f"{ticker} 영업이익률 계산 불가 (Date: {date}, Revenue: {revenue}, OpIncome: {op_income})")
        return records
    except requests.exceptions.RequestException as req_err: logging.error(f"FMP 손익계산서 요청 실패 (영업이익률, {ticker}): {req_err}"); return []
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (영업이익률, {ticker}): {env_err}"); return []
    except Exception as e: logging.error(f"{ticker}: 영업이익률 계산 오류 - {e}"); return []

@st.cache_data(ttl=3600)
def get_roe_trend(ticker, num_periods=4):
    """FMP 기반 최근 분기별 ROE(%) 추세"""
    # (기존 FMP 기반 코드 유지)
    logging.info(f"{ticker}: FMP ROE 추세 조회 ({num_periods} 분기)...")
    try:
        income_data = get_income_statement(ticker, limit=num_periods); balance_data = get_balance_sheet(ticker, limit=num_periods)
        if not income_data or not balance_data: logging.warning(f"{ticker}: ROE 계산 위한 FMP 데이터 부족."); return []
        equity_map = {b['date']: b.get('totalStockholdersEquity') or b.get('totalEquity') for b in balance_data if 'date' in b}
        records = []
        for item in income_data:
            date = item.get("date"); ni = item.get("netIncome"); equity = equity_map.get(date)
            if ni is not None and equity is not None and equity != 0 and date:
                roe = round((ni / equity) * 100, 2); records.append({"Date": date, "ROE (%)": roe})
            else: logging.debug(f"{ticker} ROE 계산 불가 (Date: {date}, NI: {ni}, Equity: {equity})")
        return list(reversed(records))
    except requests.exceptions.RequestException as req_err: logging.error(f"FMP 재무제표 요청 실패 (ROE, {ticker}): {req_err}"); return []
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (ROE, {ticker}): {env_err}"); return []
    except Exception as e: logging.error(f"{ticker}: ROE 계산 오류 - {e}"); return []

@st.cache_data(ttl=3600)
def get_debt_to_equity_trend(ticker, num_periods=4):
    """FMP 기반 부채비율(D/E Ratio) 추세 계산"""
    # (기존 FMP 기반 코드 유지)
    logging.info(f"{ticker}: FMP 부채비율 추세 조회 ({num_periods} 분기)...")
    try:
        bs_data = get_balance_sheet(ticker, limit=num_periods); records = []
        if not bs_data: logging.warning(f"{ticker}: FMP 재무상태표 데이터 없음 (부채비율)"); return []
        for item in reversed(bs_data):
            date = item.get("date"); equity = item.get("totalStockholdersEquity") or item.get("totalEquity"); debt = item.get("totalDebt")
            if debt is None: debt = (item.get("shortTermDebt") or 0) + (item.get("longTermDebt") or 0)
            if equity is not None and equity != 0 and debt is not None and date:
                ratio = round(debt / equity, 2); records.append({"Date": date, "D/E Ratio": ratio})
            else: logging.debug(f"{ticker} 부채비율 계산 불가 (Date: {date}, Equity: {equity}, Debt: {debt})")
        return records
    except requests.exceptions.RequestException as req_err: logging.error(f"FMP 재무상태표 요청 실패 (부채비율, {ticker}): {req_err}"); return []
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (부채비율, {ticker}): {env_err}"); return []
    except Exception as e: logging.error(f"{ticker}: 부채비율 계산 오류 - {e}"); return []

@st.cache_data(ttl=3600)
def get_current_ratio_trend(ticker, num_periods=4):
    """FMP 기반 유동비율(Current Ratio) 추세 계산"""
    # (기존 FMP 기반 코드 유지)
    logging.info(f"{ticker}: FMP 유동비율 추세 조회 ({num_periods} 분기)...")
    try:
        bs_data = get_balance_sheet(ticker, limit=num_periods); records = []
        if not bs_data: logging.warning(f"{ticker}: FMP 재무상태표 데이터 없음 (유동비율)"); return []
        for item in reversed(bs_data):
            date = item.get("date"); current_assets = item.get("totalCurrentAssets"); current_liabilities = item.get("totalCurrentLiabilities")
            if current_assets is not None and current_liabilities is not None and current_liabilities != 0 and date:
                ratio = round(current_assets / current_liabilities, 2); records.append({"Date": date, "Current Ratio": ratio})
            else: logging.debug(f"{ticker} 유동비율 계산 불가 (Date: {date}, Assets: {current_assets}, Liab: {current_liabilities})")
        return records
    except requests.exceptions.RequestException as req_err: logging.error(f"FMP 재무상태표 요청 실패 (유동비율, {ticker}): {req_err}"); return []
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (유동비율, {ticker}): {env_err}"); return []
    except Exception as e: logging.error(f"{ticker}: 유동비율 계산 오류 - {e}"); return []

# --- 분석 및 시각화 함수들 ---
def plot_stock_chart(ticker, start_date=None, end_date=None, period="1y"):
    """주가 차트 Figure 객체 반환 (fmp_api 데이터 사용)"""
    # (기존 FMP 기반 코드 유지)
    logging.info(f"Plotting stock chart for {ticker} using fmp_api data...")
    df = get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period)
    if df is None or df.empty: logging.error(f"{ticker} 차트 실패: FMP 데이터 없음"); return None
    try:
        required_plot_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_plot_cols): logging.error(f"{ticker} 차트 실패: 필수 컬럼 부족 ({[c for c in required_plot_cols if c not in df.columns]})"); return None
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1)
        fig.update_layout(title=f'{ticker} 주가/거래량 차트', yaxis_title='가격', yaxis2_title='거래량', xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="가격", row=1, col=1); fig.update_yaxes(title_text="거래량", row=2, col=1)
        logging.info(f"{ticker} 차트 생성 완료"); return fig
    except Exception as e: logging.error(f"{ticker} 차트 생성 오류: {e}"); return None

def get_news_sentiment(ticker): # Removed api_key parameter
    """뉴스 감정 분석 (FMP API + TextBlob)"""
    # (기존 FMP + TextBlob 코드 유지)
    logging.info(f"Fetching and analyzing news for {ticker} using fmp_api & TextBlob...")
    try:
        articles = fmp_api.get_news(ticker, limit=20) # Uses fmp_api.py function
        if not articles: logging.info(f"{ticker}: 관련 뉴스 없음 (fmp_api)."); return ["관련 뉴스 없음."]
        output = []; total_polarity = 0.0; analyzed_count = 0
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'N/A'); text_content = article.get('text', '') or ""; published_date = article.get('publishedDate', ''); source_site = article.get('site', '')
            prefix = f"{i}. ";
            if published_date: try: date_str = pd.to_datetime(published_date).strftime('%Y-%m-%d %H:%M'); prefix += f"[{date_str}] " except: prefix += f"[{published_date}] "
            if source_site: prefix += f"({source_site}) "
            try:
                analysis_text = text_content if text_content and len(text_content) > 10 else title
                if analysis_text and analysis_text != 'N/A' and analysis_text != '[Removed]':
                    blob = TextBlob(analysis_text); polarity = blob.sentiment.polarity
                    output.append(f"{prefix}{title} | 감정: {polarity:.2f}"); total_polarity += polarity; analyzed_count += 1
                else: output.append(f"{prefix}{title} | 내용 부족/없음")
            except Exception as text_e: logging.warning(f"뉴스 감성 분석 오류({title}): {text_e}"); output.append(f"{prefix}{title} | 감성 분석 오류")
        avg_polarity = total_polarity / analyzed_count if analyzed_count > 0 else 0
        logging.info(f"{ticker} 뉴스 분석 완료 (fmp_api, 평균: {avg_polarity:.2f})"); output.insert(0, f"총 {analyzed_count}개 분석 | 평균 감성: {avg_polarity:.2f}")
        return output
    except requests.exceptions.RequestException as req_err: logging.error(f"fmp_api 뉴스 요청 실패: {req_err}"); return [f"뉴스 API 요청 실패 (fmp_api): {req_err}"]
    except EnvironmentError as env_err: logging.error(f"FMP API 키 오류 (뉴스): {env_err}"); return ["FMP API 키 설정 오류로 뉴스 분석 실패."]
    except Exception as e: logging.error(f"뉴스 분석 중 예외 발생 (fmp_api): {e}\n{traceback.format_exc()}"); return ["뉴스 분석 중 오류 발생."]

# --- Prophet Forecasting (FMP Macro Data 적용) ---
def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, changepoint_prior_scale=0.05): # fred_key 제거
    """Prophet 예측 (FMP Stock Data + FMP Macro Data + TA Regressors)"""
    logging.info(f"{ticker}: Prophet 예측 시작 (FMP Data, cp_prior={changepoint_prior_scale})...")
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d")

    # 1. 초기 주가 데이터 로딩 (FMP 사용 - get_stock_data 호출)
    df_stock_initial = get_stock_data(ticker, start_date=start_date, end_date=end_date)
    if df_stock_initial is None or df_stock_initial.empty:
        logging.error(f"{ticker}: Prophet 예측 위한 FMP 주가 데이터 로드 실패.")
        return None, None, None, None # 4 Nones 반환

    try: # 데이터 처리 부분 추가 예외 처리
        if isinstance(df_stock_initial.index, pd.DatetimeIndex): df_stock_processed = df_stock_initial.reset_index()
        else: logging.error("주가 데이터 인덱스가 DatetimeIndex가 아님."); return None, None, None, None
        required_cols_prophet = ["Date", "Close", "Open", "High", "Low", "Volume"]
        cols_exist = [col for col in required_cols_prophet if col in df_stock_processed.columns]; df_stock_processed = df_stock_processed[cols_exist].copy()
        if "Date" not in df_stock_processed.columns: logging.error(f"{ticker}: Prophet용 'Date' 컬럼 없음."); return None, None, None, None
        df_stock_processed["Date"] = pd.to_datetime(df_stock_processed["Date"])
        if "Close" in df_stock_processed.columns and df_stock_processed["Close"].isnull().any():
            rows_before = len(df_stock_processed); df_stock_processed.dropna(subset=["Close"], inplace=True)
            logging.info(f"{ticker}: 'Close' NaN 값으로 {rows_before - len(df_stock_processed)} 행 제거됨.")
        if df_stock_processed.empty or "Close" not in df_stock_processed.columns or df_stock_processed["Close"].isnull().all():
            logging.error(f"{ticker}: 유효한 'Close' 데이터 없음."); return None, None, None, None
        logging.info(f"{ticker}: 초기 주가 (FMP) 로딩/처리 완료 (Shape: {df_stock_processed.shape}).")
    except Exception as proc_err:
        logging.error(f"{ticker}: 주가 데이터 처리 중 오류: {proc_err}\n{traceback.format_exc()}")
        return None, None, None, None

    # 2. FMP 기반 매크로 데이터 로딩 및 병합 (수정된 로직)
    macro_data_dict = get_macro_data_fmp() # FMP 함수 호출
    macro_cols = ["VIX", "US10Y", "US13W", "DXY"]
    df_merged = df_stock_processed.copy()
    if macro_data_dict:
        logging.info(f"{ticker}: FMP 매크로 데이터 로딩 성공: {macro_data_dict}")
        for col in macro_cols:
            value = macro_data_dict.get(col)
            if value is not None:
                df_merged[col] = value # 모든 날짜에 동일 값 적용
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce') # 숫자 변환
            else:
                df_merged[col] = np.nan # 없으면 NaN 컬럼
                logging.warning(f"{ticker}: FMP 매크로 데이터 '{col}' 누락/None.")
        # NaN 처리 (단일 값이므로 ffill/bfill보다는 그냥 두거나 0으로 채우는게 나을 수 있음)
        # df_merged[macro_cols] = df_merged[macro_cols].fillna(0) # 예시: 0으로 채우기
        df_merged[macro_cols] = df_merged[macro_cols].ffill().bfill() # 안전하게 ffill/bfill
        logging.info(f"{ticker}: FMP 매크로 변수 병합 및 NaN 처리 완료.")
    else:
        logging.warning(f"{ticker}: FMP 매크로 데이터 없음. 주가 데이터만 사용.")
        for col in macro_cols: df_merged[col] = np.nan # 컬럼은 만듦

    # 3. 기술적 지표 계산 (pandas_ta 사용)
    logging.info(f"{ticker}: 기술적 지표 계산 시작...")
    tech_indicators_to_add = []
    base_ta_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # pandas_ta는 소문자 컬럼명 선호 - 임시로 변경
    df_merged_ta = df_merged.rename(columns={c: c.lower() for c in base_ta_cols if c in df_merged.columns})
    if not all(c.lower() in df_merged_ta.columns for c in base_ta_cols):
         logging.warning(f"{ticker}: TA 계산 위한 기본 컬럼 부족. 건너<0xEB>니다.") # 수정: 건너<0xEB>니다
    else:
        try:
            df_merged_ta.ta.rsi(close='close', length=14, append=True, col_names=('RSI_14',))
            df_merged_ta.ta.macd(close='close', fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9','MACDs_12_26_9'))
            logging.info(f"{ticker}: RSI, MACD 계산 완료.")
            # 원본 컬럼명 복구
            df_merged['RSI_14'] = df_merged_ta.get('RSI_14'); df_merged['MACDs_12_26_9'] = df_merged_ta.get('MACDs_12_26_9')
            tech_indicators_candidates = ['RSI_14', 'MACDs_12_26_9']
            for ti in tech_indicators_candidates:
                if ti in df_merged.columns:
                    if df_merged[ti].isnull().any():
                        df_merged[ti] = df_merged[ti].ffill().bfill(); logging.info(f"{ticker}: '{ti}' NaN ffill/bfill 처리.")
                    if df_merged[ti].isnull().any(): logging.warning(f"{ticker}: '{ti}' 처리 후 NaN 존재 -> 제외.")
                    else: tech_indicators_to_add.append(ti)
                else: logging.warning(f"{ticker}: 기술 지표 '{ti}' 생성 안 됨.")
        except Exception as ta_err: logging.error(f"{ticker}: TA 계산 오류: {ta_err}\n{traceback.format_exc()}"); tech_indicators_to_add = []

    # 4. 최종 데이터 검증 및 Prophet 준비
    if df_merged.empty or len(df_merged) < 30 or 'Date' not in df_merged.columns or 'Close' not in df_merged.columns:
        logging.error(f"Prophet 실패: 최종 데이터 부족/형식오류 ({len(df_merged)} 행).")
        return None, None, None, None
    logging.info(f"Prophet 학습 데이터 준비 완료 (Shape: {df_merged.shape})")
    # ... (데이터 저장 로직은 이전과 동일) ...

    # --- Prophet 모델링 ---
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"}); df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=changepoint_prior_scale)

    # Regressors 추가 (FMP 매크로 + TA)
    all_regressors = []
    macro_cols_available = [col for col in macro_cols if col in df_prophet.columns and df_prophet[col].notna().any()]
    if macro_cols_available:
        for col in macro_cols_available:
            if df_prophet[col].isnull().any(): df_prophet[col] = df_prophet[col].fillna(0); logging.warning(f"{ticker}: Macro '{col}' NaN->0")
            m.add_regressor(col)
        all_regressors.extend(macro_cols_available); logging.info(f"{ticker}: FMP 매크로 Regressors: {macro_cols_available}")
    if tech_indicators_to_add:
        for col in tech_indicators_to_add:
            if df_prophet[col].isnull().any(): df_prophet[col] = df_prophet[col].fillna(0); logging.warning(f"{ticker}: TA '{col}' NaN->0")
            m.add_regressor(col)
        all_regressors.extend(tech_indicators_to_add); logging.info(f"{ticker}: TA Regressors: {tech_indicators_to_add}")

    # 5. 학습, 예측, CV 실행
    forecast_dict, fig_fcst, cv_path, mape = None, None, None, None
    try:
        logging.info(f"{ticker}: Prophet 학습 시작 (Regressors: {all_regressors})...")
        final_regressors = [reg for reg in all_regressors if reg in df_prophet.columns]
        cols_for_fit = ['ds', 'y'] + final_regressors; missing_fit_cols = [col for col in cols_for_fit if col not in df_prophet.columns]
        if missing_fit_cols: logging.error(f"Prophet fit 실패: 학습 컬럼 부족: {missing_fit_cols}"); return None, None, None, None
        prophet_train_df = df_prophet[cols_for_fit].dropna()
        if len(prophet_train_df) < 30: logging.error(f"Prophet fit 실패: 유효 데이터 부족 ({len(prophet_train_df)})."); return None, None, None, None
        m.fit(prophet_train_df); logging.info(f"{ticker}: Prophet 학습 완료.")
        # ... (CV 로직은 이전과 동일) ...
        try: # CV Block
            data_len_days = (prophet_train_df['ds'].max() - prophet_train_df['ds'].min()).days; initial_cv_days = max(180, int(data_len_days * 0.5)); period_cv_days = max(30, int(initial_cv_days * 0.2)); horizon_cv_days = forecast_days
            initial_cv, period_cv, horizon_cv = f'{initial_cv_days} days', f'{period_cv_days} days', f'{horizon_cv_days} days'
            if data_len_days >= initial_cv_days + horizon_cv_days:
                logging.info(f"Prophet CV 시작 (initial='{initial_cv}', period='{period_cv}', horizon='{horizon_cv}')...")
                df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel=None); logging.info("CV 완료.")
                df_p = performance_metrics(df_cv); mape = df_p["mape"].mean() * 100; logging.info(f"Prophet CV 평균 MAPE: {mape:.2f}%")
                fig_cv = plot_cross_validation_metric(df_cv, metric='mape'); plt.title(f'{ticker} CV MAPE (Params: cp={changepoint_prior_scale})')
                cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_cp{changepoint_prior_scale}.png"); fig_cv.savefig(cv_path); plt.close(fig_cv); logging.info(f"CV MAPE 차트 저장 완료: {cv_path}")
            else: logging.warning(f"{ticker}: 데이터 기간 부족 ({data_len_days} < {initial_cv_days + horizon_cv_days} days)하여 CV를 건너<0xEB>니다."); cv_path = None # 수정: 건너<0xEB>니다
        except Exception as cv_e: logging.error(f"Prophet CV 중 오류 발생: {cv_e}\n{traceback.format_exc()}"); cv_path = None; mape = None
        # Future Prediction
        logging.info("미래 예측 시작..."); future = m.make_future_dataframe(periods=forecast_days)
        if final_regressors:
             last_known_regressors = prophet_train_df[final_regressors].iloc[-1]
             for col in final_regressors: future[col] = last_known_regressors[col]
             if future[final_regressors].isnull().any().any(): logging.warning(f"미래 예측 Df NaN Regressor -> 0 채움."); future[final_regressors] = future[final_regressors].fillna(0)
        predict_cols = ['ds'] + final_regressors; missing_predict_cols = [col for col in predict_cols if col not in future.columns]
        if missing_predict_cols: logging.error(f"Predict 실패: 예측 데이터 컬럼 부족: {missing_predict_cols}"); return None, None, None, mape
        forecast = m.predict(future[predict_cols]); logging.info("미래 예측 완료.")
        # ... (CSV 저장 로직은 이전과 동일) ...
        fig_fcst = plot_plotly(m, forecast); fig_fcst.update_layout(title=f'{ticker} Price Forecast (cp={changepoint_prior_scale})', margin=dict(l=20,r=20,t=40,b=20)); logging.info(f"예측 결과 Figure 생성 완료.")
        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records'); [rec.update({'ds': rec['ds'].strftime('%Y-%m-%d')}) for rec in forecast_dict]
        return forecast_dict, fig_fcst, cv_path, mape
    except Exception as e: logging.error(f"Prophet 학습/예측 오류: {e}\n{traceback.format_exc()}"); return None, None, cv_path, mape

# --- 메인 분석 함수 ---
def analyze_stock(ticker, news_key=None, fred_key=None, analysis_period_years=2, forecast_days=30, num_trend_periods=4, changepoint_prior_scale=0.05): # fred_key 인자 유지 (Prophet 호출용), news_key 제거
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다. (FMP API 기반, Prophet은 FMP Macro)"""
    logging.info(f"--- {ticker} 주식 분석 시작 (FMP, cp_prior={changepoint_prior_scale}) ---")
    output_results = {}
    try: # 날짜 설정
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0); start_date = end_date - relativedelta(years=analysis_period_years)
        start_date_str = start_date.strftime("%Y-%m-%d"); end_date_str = end_date.strftime("%Y-%m-%d")
        logging.info(f"분석 기간: {start_date_str} ~ {end_date_str}")
    except Exception as e: logging.error(f"날짜 설정 오류: {e}"); return {"error": f"날짜 설정 오류: {e}"}

    # Stock data (FMP)
    df_stock_full = get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str)
    stock_data_valid = df_stock_full is not None and not df_stock_full.empty and 'Close' in df_stock_full.columns
    if stock_data_valid: output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty and pd.notna(df_stock_full['Close'].iloc[-1]) else "N/A"; output_results['data_points'] = len(df_stock_full)
    else: output_results['current_price'] = "N/A"; output_results['data_points'] = 0; logging.warning(f"{ticker}: 분석 기간 내 유효한 주가 정보 없음 (FMP).")
    output_results['analysis_period_start'] = start_date_str; output_results['analysis_period_end'] = end_date_str

    # Analyses (FMP 기반)
    output_results['stock_chart_fig'] = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str) or None
    output_results['fundamentals'] = get_fundamental_data(ticker) or {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods) or []
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods) or []
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend(ticker, num_periods=num_trend_periods) or []
    output_results['current_ratio_trend'] = get_current_ratio_trend(ticker, num_periods=num_trend_periods) or []
    output_results['news_sentiment'] = get_news_sentiment(ticker) or ["뉴스 분석 실패"] # FMP + TextBlob
    fg_value, fg_class = get_fear_greed_index(); output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    # Prophet Forecast (FMP Macro 사용)
    output_results['prophet_forecast'] = "예측 불가"; output_results['forecast_fig'] = None; output_results['cv_plot_path'] = None; output_results['mape'] = None; output_results['warn_high_mape'] = False
    if stock_data_valid and output_results['data_points'] > 30:
        # --- fred_key 인자 제거 ---
        forecast_result = run_prophet_forecast(
            ticker, start_date=start_date_str, end_date=end_date_str,
            forecast_days=forecast_days,
            changepoint_prior_scale=changepoint_prior_scale
        )
        # -------------------------
        if forecast_result and isinstance(forecast_result, tuple) and len(forecast_result) == 4:
            fc_list, fc_fig, cv_path, mape_val = forecast_result
            output_results['prophet_forecast'] = fc_list if fc_list else "예측 결과 없음"; output_results['forecast_fig'] = fc_fig
            output_results['cv_plot_path'] = cv_path; output_results['mape'] = f"{mape_val:.2f}%" if isinstance(mape_val, (int, float)) else ("CV 실패/건너<0xEB>뜀" if cv_path is None else "MAPE 계산 불가") # 수정: 건너<0xEB>뜀
            output_results['warn_high_mape'] = isinstance(mape_val, (int, float)) and mape_val > 20
        else: output_results['prophet_forecast'] = "예측 실행 오류/결과 없음"; logging.error(f"{ticker}: run_prophet_forecast 함수 비정상 반환: {forecast_result}")
    else: msg = f"데이터 부족({output_results['data_points']})" if not stock_data_valid or output_results['data_points'] <= 30 else "주가 정보 없음"; output_results['prophet_forecast'] = f"{msg}으로 예측 불가"; logging.warning(f"{ticker}: {msg} - Prophet 예측 건너<0xEB>니다.") # 수정: 건너<0xEB>니다

    logging.info(f"--- {ticker} 주식 분석 완료 (FMP 기반) ---")
    return output_results

# --- 메인 실행 부분 (테스트용) ---
if __name__ == "__main__":
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    print("Testing with FULL FMP API integration...")
    target_ticker = "AAPL"; cps_test = 0.1
    print(f"Attempting analysis for: {target_ticker} (using FMP for all data except F&G)")
    # --- fred_key 제거 ---
    test_results = analyze_stock(ticker=target_ticker, analysis_period_years=1, forecast_days=15, num_trend_periods=4, changepoint_prior_scale=cps_test)
    # --------------------
    print(f"\n--- 테스트 실행 결과 요약 ({target_ticker}, Changepoint Prior: {cps_test}) ---")
    if test_results and isinstance(test_results, dict) and "error" not in test_results:
        for key, value in test_results.items():
            if 'fig' in key and value is not None: print(f"- {key.replace('_',' ').title()}: Plotly Figure 생성됨")
            elif key == 'fundamentals' and isinstance(value, dict): print("- Fundamentals (FMP):"); [print(f"    - {k}: {v}"[:120]) for k, v in value.items()]
            elif '_trend' in key and isinstance(value, list): print(f"- {key.replace('_',' ').title()} ({len(value)} 분기) [FMP]:"); [print(f"    - {item}") for item in value[:3]]; print("     ...") if len(value) > 3 else None
            elif key == 'prophet_forecast': status = value if isinstance(value, str) else f"{len(value)}일 예측 생성됨 (첫 날: {value[0]})" if isinstance(value, list) and value else "예측 결과 없음"; print(f"- Prophet Forecast (FMP Macro): {status}")
            elif key == 'news_sentiment': status = value[0] if isinstance(value, list) and value else "뉴스 분석 실패/오류"; print(f"- News Sentiment (FMP+TextBlob): {status}")
            elif key == 'cv_plot_path': print(f"- Cv Plot Path: {value if value else '생성 안 됨/실패/건너뜀'}")
            elif key == 'mape': print(f"- MAPE: {value if value else 'N/A'}")
            else: print(f"- {key.replace('_',' ').title()}: {value}")
    elif test_results and "error" in test_results: print(f"분석 중 오류 발생: {test_results['error']}")
    else: print("테스트 분석 실패 (결과 없음 또는 알 수 없는 오류).")
    print("\n--- 테스트 실행 종료 ---")