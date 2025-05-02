# stock_analysis.py (Modified to use fmp_api where possible)

import os
import logging
import pandas as pd
import numpy as np
import fmp_api # Import the new FMP API module
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from fredapi import Fred
import traceback
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import warnings
import locale
import re
import pandas_ta as ta
from fmp_api import get_income_statement
from fmp_api import get_income_statement, get_balance_sheet

# 경고 메시지 및 로깅 설정
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 경로 및 API 키 설정 ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
    logging.info(f"__file__ 없음. CWD 사용: {BASE_DIR}")

CHARTS_FOLDER = os.path.join(BASE_DIR, "charts")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
FORECAST_FOLDER = os.path.join(BASE_DIR, "forecast")

try:
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env 로드 성공: {dotenv_path}")
    else:
        logging.info(f".env 파일 없음 (정상일 수 있음): {dotenv_path}")
except Exception as e:
    logging.error(f".env 로드 오류: {e}")

# NEWS_API_KEY = os.getenv("NEWS_API_KEY") # No longer needed if using fmp_api for news
FRED_API_KEY = os.getenv("FRED_API_KEY")
# FMP API Key is loaded within fmp_api.py

# if not NEWS_API_KEY: logging.warning("NEWS_API_KEY 없음.") # No longer needed
if not FRED_API_KEY: logging.warning("FRED_API_KEY 없음.")
if FRED_API_KEY: logging.info("API 키 로드 시도 완료 (FRED). FMP Key handled in fmp_api.")

# --- 데이터 가져오기 함수들 ---
def get_fear_greed_index():
    """공포-탐욕 지수 가져오기"""
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

def get_stock_data(ticker, start_date=None, end_date=None, period="1y"):
    """주가 데이터 가져오기 (OHLCV 포함) - Updated for fmp_api"""
    logging.info(f"Fetching stock data for {ticker} using fmp_api...")
    try:
        # Determine number of days needed based on start/end dates or period
        days = None
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            days = (end - start).days
        elif period:
            # Approximate days based on period string (e.g., '1y' -> 365)
            period_map = {'d': 1, 'w': 7, 'mo': 30, 'y': 365}
            match = re.match(r"(\d+)([dwmyo]+)", period.lower())
            if match:
                num, unit = match.groups()
                days = int(num) * period_map.get(unit, 30) # Default to 30 days if unit unknown
            else: # Default period if parsing fails
                 days = 365


        if days is None or days <= 0:
            days = 365 # Default to 1 year if calculation fails
            logging.warning(f"Could not determine days from period/dates, defaulting to {days} days.")

        # Fetch data using fmp_api
        # Note: fmp_api.get_price_data uses the 'historical-price-full' endpoint
        # which should contain OHLCV data.
        historical_data = fmp_api.get_price_data(ticker, days=days)

        if not historical_data:
            logging.warning(f"{ticker} 주가 데이터 비어있음 (fmp_api).")
            return None

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(historical_data)

        if df.empty:
            logging.warning(f"{ticker} 주가 데이터 변환 후 비어있음 (fmp_api).")
            return None

        # --- Data Cleaning and Formatting ---
        # Rename columns to match yfinance structure (adjust if FMP names differ)
        # Common FMP names: 'date', 'open', 'high', 'low', 'close', 'volume'
        rename_map = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            # 'adjClose': 'Adj Close', # If available and needed
            'volume': 'Volume'
        }
        df = df.rename(columns=rename_map)

        # Convert 'Date' column to datetime objects and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index() # Ensure chronological order

        # Select necessary columns (case-insensitive check)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        cols_to_use = [col for col in df.columns if col in required_cols]
        df = df[cols_to_use] # Keep only existing required columns

        # Ensure numeric types and handle errors
        for col in required_cols:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
             else:
                 # Add missing required columns as NaN
                 df[col] = np.nan
                 logging.warning(f"{ticker}: 필수 컬럼 '{col}' 누락 -> NaN으로 채움 (fmp_api).")


        # Drop rows where essential TA columns are NaN (excluding Close)
        essential_ta_cols = ['Open', 'High', 'Low', 'Volume']
        initial_len = len(df)
        df.dropna(subset=[col for col in essential_ta_cols if col in df.columns], inplace=True)
        if len(df) < initial_len:
             logging.warning(f"{ticker}: TA 계산 위한 필수 컬럼(O,H,L,V) NaN 값으로 {initial_len - len(df)} 행 제거")

        # Filter by date range if start_date/end_date were provided
        if start_date and end_date:
             start_dt = pd.to_datetime(start_date)
             end_dt = pd.to_datetime(end_date)
             df = df[(df.index >= start_dt) & (df.index <= end_dt)]


        logging.info(f"{ticker} 주가 가져오기 성공 (fmp_api). Final shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as req_err:
         logging.error(f"티커 '{ticker}' fmp_api 요청 실패: {req_err}")
         return None
    except Exception as e:
        logging.error(f"티커 '{ticker}' 주가 데이터 처리 실패 (fmp_api): {e}")
        logging.error(traceback.format_exc())
        return None


def get_macro_data(start_date, end_date=None, fred_key=None):
    """매크로 지표 데이터 가져오기 (Unchanged, still uses yfinance/fred)"""
    # This function remains unchanged as fmp_api.py doesn't cover macro indices yet.
    logging.info("Fetching macro data using yfinance and Fred...")
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    yf_tickers = {"^VIX": "VIX", "^TNX": "US10Y", "^IRX": "US13W", "DX-Y.NYB": "DXY"}
    fred_series = {"FEDFUNDS": "FedFunds"}
    expected_cols = ['Date'] + list(yf_tickers.values()) + list(fred_series.values())
    df_macro = pd.DataFrame()
    all_yf_data = []

    for tk, label in yf_tickers.items():
        try:
            # Using yfinance download directly here
            tmp = yf.download(tk, start=start_date, end=end_date, progress=False, timeout=15)
            if not tmp.empty:
                tmp = tmp[['Close']].rename(columns={"Close": label})
                tmp.index = pd.to_datetime(tmp.index).tz_localize(None)
                all_yf_data.append(tmp)
                logging.info(f"{label} 성공 (yfinance)")
            else:
                logging.warning(f"{label} 비어있음 (yfinance).")
        except Exception as e:
            logging.error(f"{label} 실패 (yfinance): {e}")

    if all_yf_data:
        df_macro = pd.concat(all_yf_data, axis=1)
        if isinstance(df_macro.columns, pd.MultiIndex):
            df_macro.columns = df_macro.columns.get_level_values(-1)

    if fred_key:
        try:
            fred = Fred(api_key=fred_key)
            fred_data = []
            for series_id, label in fred_series.items():
                s = fred.get_series(series_id, start_date=start_date, end_date=end_date).rename(label)
                s.index = pd.to_datetime(s.index).tz_localize(None)
                fred_data.append(s)
            if fred_data:
                df_fred = pd.concat(fred_data, axis=1)
                if not df_macro.empty:
                    df_macro = df_macro.merge(df_fred, left_index=True, right_index=True, how='outer')
                else:
                    df_macro = df_fred
                logging.info("FRED 병합/가져오기 성공")
        except Exception as e:
            logging.error(f"FRED 실패: {e}")
    else:
        logging.warning("FRED 키 없어 스킵.")

    if not df_macro.empty:
        for col in expected_cols:
            if col != 'Date' and col not in df_macro.columns:
                df_macro[col] = pd.NA
                logging.warning(f"매크로 '{col}' 없어 NaN 추가.")
        for col in df_macro.columns:
            if col != 'Date':
                df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
        # Apply ffill().bfill() more carefully
        if 'Date' in df_macro.columns:
             df_macro = df_macro.set_index('Date')
             df_macro = df_macro.sort_index().ffill().bfill()
             df_macro = df_macro.reset_index()
        else: # if Date column was lost or never existed
             df_macro = df_macro.sort_index().ffill().bfill() # Sort by index if no Date
             df_macro = df_macro.reset_index() # Add numerical index if Date missing
             # Consider logging a warning here if Date column is critical later

        # Ensure Date column exists and is datetime
        if 'Date' not in df_macro.columns and 'index' in df_macro.columns:
             df_macro = df_macro.rename(columns={'index': 'Date'}) # Rename if reset_index created 'index'

        if 'Date' in df_macro.columns:
            df_macro["Date"] = pd.to_datetime(df_macro["Date"])
            logging.info("매크로 처리 완료.")
            # Ensure all expected columns are present before returning
            final_cols = [col for col in expected_cols if col in df_macro.columns]
            return df_macro[final_cols]
        else:
            logging.error("매크로 데이터 처리 후 'Date' 컬럼이 없습니다.")
            return pd.DataFrame(columns=expected_cols)

    else:
        logging.warning("매크로 가져오기 최종 실패.")
        return pd.DataFrame(columns=expected_cols)


# --- 기본적 분석 데이터 가져오기 함수들 ---
def format_market_cap(mc):
    """시가총액 숫자 포맷팅"""
    if isinstance(mc, (int, float)) and mc > 0:
        if mc >= 1e12: return f"${mc / 1e12:.2f} T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f} B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f} M"
        else: return f"${mc:,.0f}"
    return "N/A"

# find_financial_statement_item remains the same for now, used by yfinance-based functions below
def find_financial_statement_item(index, keywords, exact_match_keywords=None, case_sensitive=False):
    """재무제표 인덱스 항목 찾기"""
    if not isinstance(index, pd.Index): return None
    flags = 0 if case_sensitive else re.IGNORECASE

    if exact_match_keywords:
        for exact_key in exact_match_keywords:
            if exact_key in index:
                logging.debug(f"정확 매칭: '{exact_key}' for {keywords}")
                return exact_key

    pattern_keywords = [re.escape(k) for k in keywords]
    pattern = r'\b' + r'\b.*\b'.join(pattern_keywords) + r'\b'
    matches = []
    for item in index:
        if isinstance(item, str):
            try:
                if re.search(pattern, item, flags=flags):
                    matches.append(item)
            except Exception as e:
                logging.warning(f"항목명 검색 정규식 오류({keywords}, item='{item}'): {e}")
                continue

    if matches:
        best_match = min(matches, key=len)
        logging.debug(f"포함 매칭: '{best_match}' for {keywords}")
        return best_match

    logging.warning(f"재무 항목 찾기 최종 실패: {keywords}")
    return None


def get_fundamental_data(ticker):
    """Fetch key fundamental data using fmp_api.get_profile"""
    logging.info(f"{ticker}: Fetching profile using fmp_api...")
    fundamentals = {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약", "회사명", "현재가"]}
    try:
        profile = fmp_api.get_profile(ticker)

        if not profile:
            logging.warning(f"'{ticker}' 유효한 profile 없음 (fmp_api).")
            return fundamentals

        # Map FMP profile fields to our fundamentals dictionary
        fundamentals["회사명"] = profile.get("companyName", "N/A")
        fundamentals["현재가"] = f"{profile.get('price'):.2f}" if isinstance(profile.get('price'), (int, float)) else "N/A"
        fundamentals["시가총액"] = format_market_cap(profile.get("mktCap"))
        # FMP often provides PE directly
        pe_val = profile.get('pe')
        fundamentals["PER"] = f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else "N/A"
        # FMP often provides EPS directly
        eps_val = profile.get('eps')
        fundamentals["EPS"] = f"{eps_val:.2f}" if isinstance(eps_val, (int, float)) else "N/A"
        # FMP dividend yield might be 'lastDiv' / 'price' or a direct 'dividendYield' field
        # Assuming FMP provides 'dividendYield' (adjust if needed)
        div_yield = profile.get('dividendYield') # FMP often returns this as percentage * 100 or decimal
        if isinstance(div_yield, (int, float)):
             # Check if it looks like a percentage already (e.g., > 0.5) or needs multiplying
             fundamentals["배당수익률"] = f"{div_yield:.2f}%" # Adjust if FMP gives decimal
        else: # Calculate if possible
            last_div = profile.get('lastDiv')
            price = profile.get('price')
            if isinstance(last_div, (int, float)) and isinstance(price, (int, float)) and price > 0:
                 fundamentals["배당수익률"] = f"{(last_div / price) * 100:.2f}%"
            else:
                 fundamentals["배당수익률"] = "N/A"


        beta_val = profile.get('beta')
        fundamentals["베타"] = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A"

        fundamentals["업종"] = profile.get("sector", "N/A")
        fundamentals["산업"] = profile.get("industry", "N/A")
        fundamentals["요약"] = profile.get("description", "N/A") # FMP uses 'description'

        logging.info(f"{ticker} profile fetch 성공 (fmp_api).")
        return fundamentals
    except requests.exceptions.RequestException as req_err:
         logging.error(f"{ticker} profile fetch 실패 (fmp_api request error): {req_err}")
         return fundamentals
    except Exception as e:
        logging.error(f"{ticker} profile 처리 실패 (fmp_api): {e}")
        return fundamentals


# --- Financial Statement Trend Functions (Still using yfinance) ---
# These need to be adapted to use FMP financial statement endpoints if full replacement is desired.


def get_operating_margin_trend(ticker, num_periods=4):
    """FMP 기반 영업이익률(%) 추세"""
    try:
        data = get_income_statement(ticker, limit=num_periods)
        records = []
        for item in reversed(data):  # 최근 → 오래된 순으로 정렬
            revenue = item.get("revenue")
            op_income = item.get("operatingIncome")
            if revenue and op_income and revenue != 0:
                margin = round((op_income / revenue) * 100, 2)
                records.append({"Date": item["date"], "Op Margin (%)": margin})
        return records
    except Exception as e:
        logging.error(f"{ticker}: 영업이익률 계산 오류 - {e}")
        return []

from fmp_api import get_income_statement, get_balance_sheet

def get_roe_trend(ticker, num_periods=4):
    """FMP 기반 최근 분기별 ROE(%) 추세"""
    try:
        income_data = get_income_statement(ticker, limit=num_periods)
        balance_data = get_balance_sheet(ticker, limit=num_periods)

        if not income_data or not balance_data:
            return []

        # date 기준으로 정렬된 dict 생성 (→ 빠른 매칭용)
        equity_map = {b['date']: b.get('totalStockholdersEquity') or b.get('totalEquity') for b in balance_data}

        records = []
        for item in income_data:
            date = item.get("date")
            ni = item.get("netIncome")
            equity = equity_map.get(date)
            if ni is not None and equity and equity != 0:
                roe = round((ni / equity) * 100, 2)
                records.append({"Date": date, "ROE (%)": roe})
        return list(reversed(records))  # 최근 → 과거 순서
    except Exception as e:
        logging.error(f"{ticker}: ROE 계산 오류 - {e}")
        return []

from fmp_api import get_balance_sheet

def get_debt_to_equity_trend(ticker, num_periods=4):
    """FMP 기반 부채비율(D/E Ratio) 추세 계산"""
    try:
        bs_data = get_balance_sheet(ticker, limit=num_periods)
        if not bs_data:
            return []

        records = []
        for item in reversed(bs_data):  # 최근 → 오래된 순으로
            date = item.get("date")
            equity = item.get("totalStockholdersEquity") or item.get("totalEquity")
            debt = item.get("totalDebt")

            # fallback: 총부채가 없을 경우, 단기+장기로 계산
            if debt is None:
                st_debt = item.get("shortTermDebt") or 0
                lt_debt = item.get("longTermDebt") or 0
                debt = st_debt + lt_debt

            if equity and equity != 0 and debt is not None:
                ratio = round(debt / equity, 2)
                records.append({"Date": date, "D/E Ratio": ratio})
        return records
    except Exception as e:
        logging.error(f"{ticker}: 부채비율 계산 오류 - {e}")
        return []

def get_current_ratio_trend(ticker, num_periods=4):
    """FMP 기반 유동비율(Current Ratio) 추세 계산"""
    try:
        bs_data = get_balance_sheet(ticker, limit=num_periods)
        if not bs_data:
            return []

        records = []
        for item in reversed(bs_data):  # 최근 → 오래된 순서
            date = item.get("date")
            current_assets = item.get("totalCurrentAssets")
            current_liabilities = item.get("totalCurrentLiabilities")

            if current_assets and current_liabilities and current_liabilities != 0:
                ratio = round(current_assets / current_liabilities, 2)
                records.append({"Date": date, "Current Ratio": ratio})
        return records
    except Exception as e:
        logging.error(f"{ticker}: 유동비율 계산 오류 - {e}")
        return []

# --- 분석 및 시각화 함수들 ---
def plot_stock_chart(ticker, start_date=None, end_date=None, period="1y"):
    """주가 차트 Figure 객체 반환 (Uses get_stock_data, now fmp_api based)"""
    logging.info(f"Plotting stock chart for {ticker}...")
    # Calls the updated get_stock_data which now uses fmp_api
    df = get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period)
    if df is None or df.empty:
        logging.error(f"{ticker} 차트 실패: 데이터 없음")
        return None
    try:
        # Check if necessary columns exist after fmp_api fetch and processing
        required_plot_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_plot_cols if col not in df.columns]
        if missing_cols:
             logging.error(f"{ticker} 차트 실패: 필수 컬럼 부족: {missing_cols}")
             # Add placeholder columns if needed for plotting, though plot might be misleading
             # for col in missing_cols: df[col] = df['Close'] # Example placeholder
             return None # Or return None if essential columns missing


        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1)
        fig.update_layout(title=f'{ticker} 주가/거래량 차트', yaxis_title='가격', yaxis2_title='거래량', xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        logging.info(f"{ticker} 차트 생성 완료")
        return fig
    except Exception as e:
        logging.error(f"{ticker} 차트 생성 오류: {e}")
        return None

def get_news_sentiment(ticker, api_key=None): # api_key param no longer used
    """뉴스 감정 분석 (Updated for fmp_api)"""
    logging.info(f"Fetching news for {ticker} using fmp_api...")
    # Uses fmp_api.get_news() - doesn't need a separate API key argument
    try:
        # Fetch news using fmp_api (limit can be adjusted if needed)
        articles = fmp_api.get_news(ticker, limit=20) # Use limit from fmp_api

        if not articles:
            logging.info(f"{ticker}: 관련 뉴스 없음 (fmp_api).")
            return ["관련 뉴스 없음."]

        output, total_pol, count = [], 0, 0
        # FMP news structure might differ from NewsAPI. Common fields:
        # 'symbol', 'publishedDate', 'title', 'image', 'site', 'text', 'url'
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'N/A')
            # Use 'text' field from FMP for content/description
            text_content = article.get('text', '') or ""
            published_date = article.get('publishedDate', '')
            source_site = article.get('site', '')

            # Prepend date and source if available
            prefix = f"{i}. "
            if published_date:
                try: # Format date nicely
                    date_str = pd.to_datetime(published_date).strftime('%Y-%m-%d %H:%M')
                    prefix += f"[{date_str}] "
                except: # Fallback if date format is unexpected
                     prefix += f"[{published_date}] "

            if source_site:
                prefix += f"({source_site}) "


            if text_content and text_content != "[Removed]": # Check against common removal patterns
                try:
                    # Use title if text is very short or missing
                    analysis_text = text_content if len(text_content) > 50 else (title if title != 'N/A' else "")
                    if not analysis_text:
                         output.append(f"{prefix}{title} | 내용 분석 불가")
                         continue

                    blob = TextBlob(analysis_text)
                    pol = blob.sentiment.polarity
                    output.append(f"{prefix}{title} | 감정: {pol:.2f}")
                    total_pol += pol
                    count += 1
                except Exception as text_e:
                    logging.warning(f"뉴스 처리 오류({title}): {text_e}")
                    output.append(f"{prefix}{title} | 감정 분석 오류")
            else:
                 output.append(f"{prefix}{title} | 내용 없음")


        avg_pol = total_pol / count if count > 0 else 0
        logging.info(f"{ticker} 뉴스 분석 완료 (fmp_api, 평균: {avg_pol:.2f})")
        output.insert(0, f"총 {count}개 분석 | 평균 감성: {avg_pol:.2f}")
        return output
    except requests.exceptions.RequestException as req_err:
        logging.error(f"fmp_api 뉴스 요청 실패: {req_err}")
        return [f"뉴스 API 요청 실패 (fmp_api): {req_err}"]
    except Exception as e:
        logging.error(f"뉴스 분석 오류 (fmp_api): {e}")
        return ["뉴스 분석 중 오류 발생."]


# --- Prophet Forecasting ---
# Needs careful checking as it relies on get_stock_data and get_macro_data outputs
def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, fred_key=None, changepoint_prior_scale=0.05):
    """Prophet 예측 (기술 지표 Regressor + 파라미터 적용)"""
    logging.info(f"{ticker}: Prophet 예측 시작 (changepoint_prior_scale={changepoint_prior_scale})...")
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # 1. 초기 주가 데이터 로딩 (Uses updated get_stock_data -> fmp_api)
    df_stock_initial = None
    try:
        # Now uses fmp_api via get_stock_data
        df_stock_initial = get_stock_data(ticker, start_date=start_date, end_date=end_date)
        if df_stock_initial is None or df_stock_initial.empty:
            return None, None, None, None # Return 4 Nones

        # Reset index to get 'Date' column, select necessary columns
        # Ensure 'Date' column exists after reset_index
        if isinstance(df_stock_initial.index, pd.DatetimeIndex):
             df_stock_processed = df_stock_initial.reset_index()
        else: # If index is not datetime (shouldn't happen with current get_stock_data)
             logging.error("Stock data index is not DatetimeIndex before processing.")
             # Attempt reset anyway or handle error
             df_stock_processed = df_stock_initial.reset_index()


        required_cols_prophet = ["Date", "Close", "Open", "High", "Low", "Volume"]
        cols_exist = [col for col in required_cols_prophet if col in df_stock_processed.columns]
        df_stock_processed = df_stock_processed[cols_exist].copy()

        if "Date" not in df_stock_processed.columns:
            logging.error(f"{ticker}: 'Date' column missing after processing stock data for Prophet.")
            return None, None, None, None


        df_stock_processed["Date"] = pd.to_datetime(df_stock_processed["Date"])

        # Drop rows where 'Close' is NaN as Prophet needs 'y'
        if "Close" in df_stock_processed.columns and df_stock_processed["Close"].isnull().any():
            rows_before = len(df_stock_processed)
            df_stock_processed.dropna(subset=["Close"], inplace=True)
            logging.warning(f"{ticker}: 'Close' 컬럼 NaN 값으로 인해 {rows_before - len(df_stock_processed)} 행 제거됨.")

        if df_stock_processed.empty or "Close" not in df_stock_processed.columns or df_stock_processed["Close"].isnull().all():
            logging.error(f"{ticker}: 'Close'가 유효한 데이터가 없습니다.")
            return None, None, None, None
        logging.info(f"{ticker}: 초기 주가 데이터 로딩 및 기본 처리 완료 (Shape: {df_stock_processed.shape}).")

    except Exception as get_data_err:
        logging.error(f"{ticker}: 초기 주가 로딩/처리 중 오류: {get_data_err}")
        logging.error(traceback.format_exc())
        return None, None, None, None

    # 2. 매크로 데이터 로딩 및 병합 (Unchanged, uses yfinance/fred)
    df_macro = get_macro_data(start_date=start_date, end_date=end_date, fred_key=fred_key)
    macro_cols = ["VIX", "US10Y", "US13W", "DXY", "FedFunds"]
    df_merged = df_stock_processed # Start with processed stock data

    # Merge only if df_macro is not empty and has 'Date' column
    if not df_macro.empty and 'Date' in df_macro.columns:
        try:
            df_macro['Date'] = pd.to_datetime(df_macro['Date'])
            # Merge based on 'Date'
            df_merged = pd.merge(df_stock_processed, df_macro, on="Date", how="left")
            logging.info(f"{ticker}: 주가/매크로 데이터 병합 완료.")
            # Fill NaNs in macro columns after merging
            for col in macro_cols:
                if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').ffill().bfill()
            logging.info(f"{ticker}: 매크로 변수 NaN 처리 완료.")
        except Exception as merge_err:
            logging.error(f"{ticker}: 데이터 병합 오류: {merge_err}")
            # Fallback to using only stock data if merge fails
            df_merged = df_stock_processed
            logging.warning(f"{ticker}: 매크로 병합 실패. 주가 데이터만 사용하여 진행.")
    else:
        logging.warning(f"{ticker}: 매크로 데이터 없음 또는 형식 오류. 주가 데이터만 사용.")
        # df_merged is already df_stock_processed

    # 3. 기술적 지표 계산
    logging.info(f"{ticker}: 기술적 지표 계산 시작...")
    tech_indicators_to_add = []
    # Ensure required base columns exist before calculating TAs
    base_ta_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df_merged.columns for col in base_ta_cols):
         logging.warning(f"{ticker}: TA 계산 위한 기본 컬럼 부족. 건너<0xEB>니다.") # 수정: 건너<0xEB>니다
    else:
        try:
            # Use lowercase names expected by pandas_ta by default
            df_merged.ta.rsi(close='close', length=14, append=True, col_names=('RSI_14',))
            df_merged.ta.macd(close='close', fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9','MACDs_12_26_9')) # Specify all names
            logging.info(f"{ticker}: RSI, MACD 계산 완료.")

            # Add only the signal line ('MACDs') as regressor if needed
            tech_indicators_candidates = ['RSI_14', 'MACDs_12_26_9']

            # Check, ffill/bfill, and validate TAs
            for ti in tech_indicators_candidates:
                if ti in df_merged.columns:
                    if df_merged[ti].isnull().any():
                        logging.info(f"{ticker}: 기술 지표 '{ti}'의 초기 NaN 값을 ffill/bfill 처리합니다.")
                        df_merged[ti] = df_merged[ti].ffill().bfill()
                    if df_merged[ti].isnull().any():
                        logging.warning(f"{ticker}: 기술 지표 '{ti}'에 처리 후에도 NaN 존재하여 Regressor에서 제외합니다.")
                    else:
                        tech_indicators_to_add.append(ti)
                else:
                    logging.warning(f"{ticker}: 기술 지표 '{ti}'가 생성되지 않았습니다.")
        except Exception as ta_err:
            logging.error(f"{ticker}: 기술적 지표 계산 중 오류: {ta_err}")
            logging.error(traceback.format_exc())
            tech_indicators_to_add = [] # Reset on error

    # 4. 최종 데이터 검증 및 Prophet 준비
    if df_merged.empty or len(df_merged) < 30 or 'Date' not in df_merged.columns or 'Close' not in df_merged.columns:
        logging.error(f"Prophet 실패: 최종 데이터 부족/형식오류({len(df_merged)}).")
        return None, None, None, None

    logging.info(f"Prophet 학습 데이터 준비 완료 (Shape: {df_merged.shape})")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    data_csv = os.path.join(DATA_FOLDER, f"{ticker}_prophet_tech_input_cp{changepoint_prior_scale}.csv")
    try:
        df_merged.to_csv(data_csv, index=False)
        logging.info(f"Prophet 학습 데이터 저장 완료: {data_csv}")
    except Exception as e:
        logging.error(f"학습 데이터 저장 실패: {e}")

    # --- Prophet 모델링 ---
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale
    )

    # Add Regressors
    all_regressors = []
    # Macro Regressors
    macro_cols_available = [col for col in macro_cols if col in df_prophet.columns and pd.api.types.is_numeric_dtype(df_prophet[col]) and df_prophet[col].isnull().sum() == 0]
    if macro_cols_available:
        for col in macro_cols_available: m.add_regressor(col)
        all_regressors.extend(macro_cols_available)
        logging.info(f"{ticker}: 매크로 Regressors 추가됨: {macro_cols_available}")

    # Technical Indicator Regressors
    if tech_indicators_to_add:
        for col in tech_indicators_to_add: m.add_regressor(col)
        all_regressors.extend(tech_indicators_to_add)
        logging.info(f"{ticker}: 기술 지표 Regressors 추가됨: {tech_indicators_to_add}")

    # 5. 학습, 예측, CV 실행
    forecast_dict, fig_fcst, cv_path, mape = None, None, None, None
    try:
        logging.info(f"{ticker}: Prophet 모델 학습 시작 (Regressors: {all_regressors})...")
        # Ensure all regressor columns actually exist in the final df_prophet
        final_regressors = [reg for reg in all_regressors if reg in df_prophet.columns]
        cols_for_fit = ['ds', 'y'] + final_regressors
        missing_fit_cols = [col for col in cols_for_fit if col not in df_prophet.columns]
        if missing_fit_cols:
             logging.error(f"Prophet fit 실패: 다음 컬럼이 학습 데이터에 없음: {missing_fit_cols}")
             return None, None, None, None

        # Drop rows where any regressor needed for fitting is NaN
        prophet_train_df = df_prophet[cols_for_fit].dropna()
        if len(prophet_train_df) < 30:
            logging.error(f"Prophet fit 실패: Regressor 포함 후 유효 데이터 부족 ({len(prophet_train_df)}).")
            return None, None, None, None

        m.fit(prophet_train_df)
        logging.info(f"{ticker}: Prophet 학습 완료.")
        os.makedirs(FORECAST_FOLDER, exist_ok=True)

        # Cross Validation (Error handling included)
        try:
            data_len_days = (prophet_train_df['ds'].max() - prophet_train_df['ds'].min()).days
            initial_cv_days = max(180, int(data_len_days * 0.5))
            period_cv_days = max(30, int(initial_cv_days * 0.2))
            horizon_cv_days = forecast_days
            initial_cv, period_cv, horizon_cv = f'{initial_cv_days} days', f'{period_cv_days} days', f'{horizon_cv_days} days'

            # Ensure sufficient data length for CV settings
            if data_len_days >= initial_cv_days + horizon_cv_days: # Simplified check
                logging.info(f"Prophet CV 시작 (initial='{initial_cv}', period='{period_cv}', horizon='{horizon_cv}')...")
                # Run CV without parallel for broader compatibility first
                df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel=None)
                logging.info("CV 완료.")
                df_p = performance_metrics(df_cv)
                mape = df_p["mape"].mean() * 100
                logging.info(f"Prophet CV 평균 MAPE: {mape:.2f}%")
                fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
                plt.title(f'{ticker} CV MAPE (Params: cp={changepoint_prior_scale})')
                cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_cp{changepoint_prior_scale}.png")
                fig_cv.savefig(cv_path)
                plt.close(fig_cv)
                logging.info(f"CV MAPE 차트 저장 완료: {cv_path}")
            else:
                logging.warning(f"{ticker}: 데이터 기간 부족 ({data_len_days} < {initial_cv_days + horizon_cv_days} days)하여 CV를 건너<0xEB>니다.") # 수정: 건너<0xEB>니다
                cv_path = None
        except Exception as cv_e:
            logging.error(f"Prophet CV 중 오류 발생: {cv_e}")
            logging.error(traceback.format_exc())
            cv_path = None
            mape = None # Ensure mape is None if CV fails


        # Future Prediction
        logging.info("미래 예측 시작...")
        future = m.make_future_dataframe(periods=forecast_days)

        # Prepare future regressors
        if final_regressors:
             # Use the *training* data (prophet_train_df) which has regressors processed
             temp_reg = prophet_train_df[['ds'] + final_regressors].copy()
             future = future.merge(temp_reg, on='ds', how='left')
             # Fill future NaNs using ffill then the last known value
             for col in final_regressors:
                 if col in future.columns:
                     last_known_val = temp_reg[col].iloc[-1] if not temp_reg[col].empty else 0
                     future[col] = future[col].ffill().fillna(last_known_val)
             # Final check for NaNs in future regressors
             if future[final_regressors].isnull().any().any():
                  logging.warning(f"미래 예측 데이터프레임에 NaN Regressor 값 존재. 0으로 채움.")
                  future[final_regressors] = future[final_regressors].fillna(0)


        # Predict
        # Ensure future dataframe has all columns model was trained on
        predict_cols = ['ds'] + final_regressors
        missing_predict_cols = [col for col in predict_cols if col not in future.columns]
        if missing_predict_cols:
             logging.error(f"Predict 실패: 다음 컬럼이 예측 데이터에 없음: {missing_predict_cols}")
             return None, None, None, mape # Return mape if CV succeeded

        forecast = m.predict(future[predict_cols])
        logging.info("미래 예측 완료.")

        csv_fn = os.path.join(FORECAST_FOLDER, f"{ticker}_forecast_cp{changepoint_prior_scale}.csv")
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy().assign(ds=lambda dfx: dfx['ds'].dt.strftime('%Y-%m-%d')).to_csv(csv_fn, index=False)
        logging.info(f"예측 결과 데이터 저장 완료: {csv_fn}")

        fig_fcst = plot_plotly(m, forecast)
        fig_fcst.update_layout(title=f'{ticker} Price Forecast (cp={changepoint_prior_scale})', margin=dict(l=20,r=20,t=40,b=20))
        logging.info(f"예측 결과 Figure 생성 완료.")

        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records')
        for rec in forecast_dict: rec['ds'] = rec['ds'].strftime('%Y-%m-%d')

        # Return all 4 values
        return forecast_dict, fig_fcst, cv_path, mape
    except Exception as e:
        logging.error(f"Prophet 학습/예측 단계에서 오류 발생: {e}")
        logging.error(traceback.format_exc())
         # Return mape if available from successful CV step
        return None, None, cv_path, mape # Keep cv_path if it exists


# --- 메인 분석 함수 ---
# Takes news_key but doesn't use it if fmp_api is used for news
def analyze_stock(ticker, news_key, fred_key, analysis_period_years=2, forecast_days=30, num_trend_periods=4, changepoint_prior_scale=0.05):
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다."""
    logging.info(f"--- {ticker} 주식 분석 시작 (changepoint_prior={changepoint_prior_scale}) ---")
    output_results = {}
    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - relativedelta(years=analysis_period_years)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        logging.info(f"분석 기간: {start_date_str} ~ {end_date_str}")
    except Exception as e:
        logging.error(f"날짜 설정 오류: {e}")
        return {"error": f"날짜 설정 오류: {e}"}

    # Stock data (uses fmp_api via get_stock_data)
    df_stock_full = get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str)
    stock_data_valid = df_stock_full is not None and not df_stock_full.empty and 'Close' in df_stock_full.columns

    if stock_data_valid:
        output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty and pd.notna(df_stock_full['Close'].iloc[-1]) else "N/A"
        output_results['data_points'] = len(df_stock_full)
    else:
        output_results['current_price'] = "N/A"
        output_results['data_points'] = 0
        logging.warning(f"{ticker}: 분석 기간 내 유효한 주가 정보 없음.")

    output_results['analysis_period_start'] = start_date_str
    output_results['analysis_period_end'] = end_date_str

    # Run analyses
    # Stock Chart (uses fmp_api data)
    output_results['stock_chart_fig'] = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str) or None
    # Fundamentals (uses fmp_api profile)
    output_results['fundamentals'] = get_fundamental_data(ticker) or {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약", "회사명", "현재가"]}
    # Financial Trends (still yfinance - needs FMP integration)
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods) or []
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods) or []
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend(ticker, num_periods=num_trend_periods) or []
    output_results['current_ratio_trend'] = get_current_ratio_trend(ticker, num_periods=num_trend_periods) or []
    # News Sentiment (uses fmp_api news)
    output_results['news_sentiment'] = get_news_sentiment(ticker) or ["뉴스 분석 실패"] # No API key needed here
    # Fear & Greed (external API)
    fg_value, fg_class = get_fear_greed_index()
    output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    # Prophet Forecast (uses fmp_api stock data, yfinance/fred macro data)
    output_results['prophet_forecast'] = "예측 불가 (초기화)"
    output_results['forecast_fig']    = None
    output_results['cv_plot_path']    = None
    output_results['mape']            = None
    output_results['warn_high_mape']  = False


    if stock_data_valid and output_results['data_points'] > 30:
        forecast_result = run_prophet_forecast(
            ticker, start_date=start_date_str, end_date=end_date_str,
            forecast_days=forecast_days, fred_key=fred_key,
            changepoint_prior_scale=changepoint_prior_scale
        )

        if forecast_result and isinstance(forecast_result, tuple) and len(forecast_result) == 4:
            fc_list, fc_fig, cv_path, mape_val = forecast_result
            output_results['prophet_forecast'] = fc_list if fc_list else "예측 결과 없음"
            output_results['forecast_fig']    = fc_fig # Can be None if plotting fails
            output_results['cv_plot_path']    = cv_path # Can be None if CV fails/skipped
            output_results['mape']            = f"{mape_val:.2f}%" if isinstance(mape_val, (int, float)) else ("CV 실패/건너<0xEB>뜀" if cv_path is None else "MAPE 계산 불가") # 수정: 건너<0xEB>뜀
            output_results['warn_high_mape']  = isinstance(mape_val, (int, float)) and mape_val > 20
        else:
            # Handle case where run_prophet_forecast returns None or incorrect format
             output_results['prophet_forecast'] = "예측 실행 오류/결과 없음"
             logging.error(f"{ticker}: run_prophet_forecast 함수가 비정상적인 값을 반환했습니다: {forecast_result}")

    else:
        msg = f"데이터 부족({output_results['data_points']})" if not stock_data_valid or output_results['data_points'] <= 30 else "주가 정보 없음"
        output_results['prophet_forecast'] = f"{msg}으로 예측 불가"
        logging.warning(f"{ticker}: {msg} - Prophet 예측을 건너<0xEB>니다.") # 수정: 건너<0xEB>니다


    logging.info(f"--- {ticker} 주식 분석 완료 ---")
    return output_results


# --- 메인 실행 부분 (테스트용) ---
if __name__ == "__main__":
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    # Ensure fmp_api.py is in the same directory or Python path
    print("Testing with fmp_api integration...")

    target_ticker = "AAPL" # Use a common ticker like AAPL or MSFT
    # news_key = os.getenv("NEWS_API_KEY") # Not needed for fmp_api news
    fred_key = os.getenv("FRED_API_KEY")
    cps_test = 0.1

    print(f"FRED Key available: {'Yes' if fred_key else 'No'}")
    print(f"Attempting analysis for: {target_ticker}")

    # Call analyze_stock without news_key as it's handled by fmp_api now
    test_results = analyze_stock(
        ticker=target_ticker, news_key=None, fred_key=fred_key, # Pass None for news_key
        analysis_period_years=1, forecast_days=15, num_trend_periods=4, # Reduced periods for faster test
        changepoint_prior_scale=cps_test
    )

    print(f"\n--- 테스트 실행 결과 요약 ({target_ticker}, Changepoint Prior: {cps_test}) ---")
    if test_results and isinstance(test_results, dict) and "error" not in test_results:
        for key, value in test_results.items():
            if 'fig' in key and value is not None:
                print(f"- {key.replace('_',' ').title()}: Plotly Figure 생성됨 (표시 안 함)")
            elif key == 'fundamentals' and isinstance(value, dict):
                print("- Fundamentals (fmp_api):")
                for k, v in value.items():
                    if k == '요약' and isinstance(v, str) and len(v) > 100:
                        print(f"    - {k}: {v[:100]}...")
                    else:
                        print(f"    - {k}: {v}")
            elif '_trend' in key and isinstance(value, list):
                # These are still from yfinance in this version
                print(f"- {key.replace('_',' ').title()} ({len(value)} 분기) [yfinance]:")
                for item in value[:3]:
                    print(f"    - {item}")
                if len(value) > 3: print("     ...")
            elif key == 'prophet_forecast':
                forecast_status = "예측 실패 또는 오류"
                if isinstance(value, list) and value:
                     forecast_status = f"{len(value)}일 예측 생성됨 (첫 날: {value[0]})"
                elif isinstance(value, str):
                     forecast_status = value
                print(f"- Prophet Forecast: {forecast_status}")
            elif key == 'news_sentiment':
                news_status = "뉴스 분석 실패 또는 오류 (fmp_api)"
                if isinstance(value, list) and len(value)>1: # Check for header + articles
                     news_status = f"{len(value)-1}개 뉴스 분석됨 (fmp_api) (헤더: {value[0]})"
                elif isinstance(value, list) and len(value)==1: # Only header means no news found
                     news_status = f"관련 뉴스 없음 (fmp_api) (헤더: {value[0]})"
                print(f"- News Sentiment (fmp_api): {news_status}")
            elif key == 'cv_plot_path':
                 print(f"- Cv Plot Path: {value if value else '생성 안 됨/실패/건너<0xEB>뜀'}") # 수정: 건너<0xEB>뜀
            elif key == 'mape':
                  print(f"- MAPE: {value if value else 'N/A'}")
            else:
                 print(f"- {key.replace('_',' ').title()}: {value}")
    elif test_results and "error" in test_results:
        print(f"분석 중 오류 발생: {test_results['error']}")
    else:
        print("테스트 분석 실패 (결과 없음 또는 알 수 없는 오류).")

    print("\n--- 테스트 실행 종료 ---")