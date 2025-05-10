# stock_analysis.py (FMP Macro Data & FRED 제거 최종 적용 버전 - 수정됨 with Plotly enhancements & plot_stock_chart 추가)

import os
import logging
import pandas as pd
import numpy as np
import fmp_api # FMP API 모듈
import plotly.graph_objects as go # MODIFIED: Explicitly imported for custom charts
from plotly.subplots import make_subplots
from textblob import TextBlob # 뉴스 감성 분석용
import requests
from prophet import Prophet
# from prophet.plot import plot_plotly # MODIFIED: No longer using the default Prophet plotly
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
# from fredapi import Fred # --- FRED API 제거 ---
import traceback
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import warnings
# import locale # <-- 사용되지 않으므로 제거
import re
import pandas_ta as ta
import streamlit as st # caching decorator 사용 위해 임포트
# from fmp_api import get_price_data as get_stock_data # safe_load_fmp_price_data가 get_price_data를 직접 사용하므로, 이 alias는 주석 처리 또는 필요시 fmp_api에 맞게 조정


# Import specific FMP functions for clarity
from fmp_api import (
    get_income_statement, get_balance_sheet, get_profile, get_quote_bulk,
    get_price_data, get_intraday_data, # 가격 데이터 함수 임포트 (fmp_api.py에 구현 가정)
    # 다음 함수들은 fmp_api.py에 정의되어 있다고 가정합니다.
    get_macro_data_fmp, get_fundamental_data, get_operating_margin_trend,
    get_roe_trend, get_debt_to_equity_trend, get_current_ratio_trend,
    get_news_sentiment, get_fear_greed_index
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


def safe_load_fmp_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    FMP API에서 가격 데이터를 불러와 안전하게 DataFrame으로 변환하고 유효성 검사를 수행합니다.

    Args:
        ticker (str): 종목 티커
        start_date (str): 시작일 (YYYY-MM-DD)
        end_date (str): 종료일 (YYYY-MM-DD)

    Returns:
        pd.DataFrame: 정제된 가격 데이터프레임 또는 빈 데이터프레임 (오류 시)
    """
    try:
        # fmp_api.py 에서 get_price_data를 직접 사용
        raw_data = get_price_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if raw_data is None or not isinstance(raw_data, list) or len(raw_data) == 0:
            logging.error(f"{ticker}: get_price_data() 결과 없음 또는 형식 오류")
            return pd.DataFrame()

        df = pd.DataFrame(raw_data)
        # FMP API 응답이 'date', 'open', 'high', 'low', 'close', 'volume' 키를 포함한다고 가정
        required_cols = ['date', 'close', 'open', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"{ticker}: API 응답에서 필수 컬럼 누락: {missing_cols}. 사용 가능한 컬럼: {df.columns.tolist()}")
            # 'adjClose' 같은 대체 컬럼 사용 고려 가능
            if 'close' not in df.columns and 'adjClose' in df.columns:
                 df = df.rename(columns={'adjClose': 'close'}) # adjClose를 close로 사용
                 logging.info(f"{ticker}: 'close' 컬럼 없어 'adjClose' 사용.")
            else: # 여전히 close가 없으면 에러
                 return pd.DataFrame()


        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # 컬럼 이름 표준화 (첫 글자 대문자)
        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        
        # 모든 필수 OHLCV 컬럼이 있는지 다시 확인
        final_required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in final_required_cols):
            logging.error(f"{ticker}: 표준화 후 필수 컬럼 누락: {final_required_cols}")
            return pd.DataFrame()
            
        # NaN 값을 가진 행 제거 또는 적절히 처리 (여기서는 dropna)
        df = df[final_required_cols].dropna(subset=['Date', 'Close']) # Date와 Close는 필수
        df = df.fillna(method='ffill').fillna(method='bfill') # 나머지 NaN은 채움

        if df.empty:
            logging.error(f"{ticker}: 데이터 처리 후 빈 데이터프레임.")
            return pd.DataFrame()

        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        # 숫자형으로 변환
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # 숫자 변환 실패 시 NaN 제거

        return df

    except Exception as e:
        logging.error(f"{ticker}: 가격 데이터 로드 및 처리 중 예외 발생: {e}\n{traceback.format_exc()}")
        return pd.DataFrame()

# ✅ 해결 방법 1. stock 차트 함수가 있다면 아래처럼 정의부 추가 (추천)
def plot_stock_chart(ticker, start_date, end_date):
    """
    주가 차트 생성 함수 (가격 + 기간 기준, Plotly 시각화)
    """
    df = safe_load_fmp_price_data(ticker, start_date, end_date)
    if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        logging.warning(f"{ticker}: 주가 차트용 데이터 없음 또는 필수 컬럼 (OHLC) 누락.")
        # 빈 Figure 객체 또는 특정 메시지를 담은 Figure 반환 고려
        fig = go.Figure()
        fig.update_layout(title=f"{ticker}: 주가 데이터 없음", xaxis_title="날짜", yaxis_title="가격")
        return fig


    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='주가 캔들'
    ))
    fig.update_layout(
        title=f"{ticker} 주가 차트",
        xaxis_title="날짜", yaxis_title="가격 ($)", # 통화 기호 추가
        xaxis_rangeslider_visible=False
    )
    return fig


# --- Prophet Forecasting (FMP Macro Data 적용) ---
def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, changepoint_prior_scale=0.05, avg_price=None): # MODIFIED: Added avg_price
    """Prophet 예측 (FMP Stock Data + FMP Macro Data + TA Regressors) - Custom Plotly Chart"""
    logging.info(f"{ticker}: Prophet 예측 시작 (FMP Data, cp_prior={changepoint_prior_scale}, avg_p_viz={avg_price})...")
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d")

    # 1. 초기 주가 데이터 로딩 (FMP 사용, safe_load_fmp_price_data 사용)
    df_stock_initial = safe_load_fmp_price_data(ticker, start_date=start_date, end_date=end_date) # 수정됨
    if df_stock_initial is None or df_stock_initial.empty:
        logging.error(f"{ticker}: Prophet 예측 위한 FMP 주가 데이터 로드 실패.")
        return None, None, None, None # 4 Nones 반환
    try: # 데이터 처리
        if isinstance(df_stock_initial.index, pd.DatetimeIndex): df_stock_processed = df_stock_initial.reset_index()
        else: logging.error("주가 데이터 인덱스가 DatetimeIndex가 아님."); return None, None, None, None
        
        # safe_load_fmp_price_data에서 컬럼명 표준화 및 필수 컬럼 확인했으므로, 여기서는 Date와 Close 존재 여부 위주로 확인
        required_cols_prophet = ["Date", "Close"] # Prophet에는 최소 Date(ds)와 Close(y)가 필요
        # Open, High, Low, Volume은 TA 계산에 필요
        ta_base_cols = ["Open", "High", "Low", "Volume"] 
        
        cols_exist_prophet = [col for col in required_cols_prophet if col in df_stock_processed.columns]
        if not all(col in cols_exist_prophet for col in required_cols_prophet):
            logging.error(f"{ticker}: Prophet용 필수 컬럼 ('Date', 'Close') 누락. 사용 가능: {df_stock_processed.columns.tolist()}"); return None, None, None, None

        df_stock_processed["Date"] = pd.to_datetime(df_stock_processed["Date"])
        if "Close" in df_stock_processed.columns and df_stock_processed["Close"].isnull().any():
            rows_before = len(df_stock_processed); df_stock_processed.dropna(subset=["Close"], inplace=True)
            logging.info(f"{ticker}: 'Close' NaN 값으로 {rows_before - len(df_stock_processed)} 행 제거됨.")
        if df_stock_processed.empty or "Close" not in df_stock_processed.columns or df_stock_processed["Close"].isnull().all():
            logging.error(f"{ticker}: 유효한 'Close' 데이터 없음."); return None, None, None, None
        logging.info(f"{ticker}: 초기 주가 (FMP) 로딩/처리 완료 (Shape: {df_stock_processed.shape}).")
    except Exception as proc_err:
        logging.error(f"{ticker}: 주가 데이터 처리 중 오류: {proc_err}\n{traceback.format_exc()}"); return None, None, None, None

    # 2. FMP 기반 매크로 데이터 로딩 및 병합 (get_macro_data_fmp 호출)
    # 이 함수는 fmp_api.py에 정의되어 있다고 가정
    macro_data_dict = get_macro_data_fmp()
    macro_cols = ["VIX", "US10Y", "US13W", "DXY"] # 예시 매크로 변수
    df_merged = df_stock_processed.copy()
    if macro_data_dict and isinstance(macro_data_dict, dict): # dict 타입인지 확인
        logging.info(f"{ticker}: FMP 매크로 데이터 로딩 성공: { {k:v for k,v in macro_data_dict.items() if k in macro_cols} }") # 관련 값만 로깅
        for col in macro_cols:
            value = macro_data_dict.get(col)
            # Prophet Regressor는 숫자형이어야 함
            df_merged[col] = pd.to_numeric(value, errors='coerce') if value is not None else np.nan # 여기서 직접 값 할당
            
            if pd.isna(df_merged[col]).all() and value is not None :
                 logging.warning(f"{ticker}: FMP 매크로 데이터 '{col}' 값 ({value}) 숫자 변환 실패.")
            elif value is None:
                 logging.warning(f"{ticker}: FMP 매크로 데이터 '{col}' 누락/None.")
        # Prophet은 regressor의 NaN을 허용하지 않으므로 ffill/bfill
        df_merged[macro_cols] = df_merged[macro_cols].ffill().bfill() 
        # 그래도 NaN이 남아있다면, 해당 regressor는 사용하지 않거나 0으로 채워야 함 (Prophet 모델링 파트에서 처리)
        logging.info(f"{ticker}: FMP 매크로 변수 병합 및 NaN 1차 처리 완료.")
    else:
        logging.warning(f"{ticker}: FMP 매크로 데이터 없음 또는 형식 오류. 주가 데이터만 사용.")
        for col in macro_cols: df_merged[col] = np.nan # 컬럼은 생성하되 NaN으로

    # 3. 기술적 지표 계산 (pandas_ta 사용 - 소문자 컬럼명 주의)
    logging.info(f"{ticker}: 기술적 지표 계산 시작...")
    tech_indicators_to_add = []
    # safe_load_fmp_price_data 에서 이미 표준화된 컬럼명(Open, High, Low, Close, Volume) 사용
    base_ta_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
    if not all(c in df_merged.columns for c in base_ta_cols):
        missing_ta_cols = [c for c in base_ta_cols if c not in df_merged.columns]
        logging.warning(f"{ticker}: TA 계산 위한 기본 컬럼 부족 ({missing_ta_cols}). 건너뜁니다.")
    else:
        # pandas_ta는 소문자 컬럼명을 선호할 수 있으나, 최근 버전은 대소문자 구분 없이 잘 동작하기도 함.
        # 명시적으로 소문자로 변경하여 사용
        df_merged_ta_input = df_merged.rename(columns={c: c.lower() for c in base_ta_cols if c in df_merged.columns})
        
        try:
            if 'close' in df_merged_ta_input.columns: # 소문자 'close'로 확인
                 # RSI
                 df_merged_ta_input.ta.rsi(close='close', length=14, append=True, col_names=('RSI_14',))
                 # MACD (macd, macdh, macds 3개 컬럼 생성)
                 df_merged_ta_input.ta.macd(close='close', fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9','MACDs_12_26_9'))
                 logging.info(f"{ticker}: RSI, MACD 계산 완료.")
                 
                 # 계산된 지표를 원본 df_merged로 복사
                 df_merged['RSI_14'] = df_merged_ta_input.get('RSI_14')
                 df_merged['MACDs_12_26_9'] = df_merged_ta_input.get('MACDs_12_26_9') # MACD signal line 사용
                 
                 tech_indicators_candidates = ['RSI_14', 'MACDs_12_26_9']
                 for ti in tech_indicators_candidates:
                     if ti in df_merged.columns and df_merged[ti].notna().any(): # 컬럼이 존재하고 NaN이 아닌 값이 하나라도 있는지 확인
                         if df_merged[ti].isnull().any(): # NaN이 일부 있다면
                             df_merged[ti] = df_merged[ti].ffill().bfill() # NaN 채우기
                             logging.info(f"{ticker}: '{ti}' NaN ffill/bfill 처리.")
                         if df_merged[ti].isnull().any(): # 그래도 NaN이 있다면 (예: 전체가 NaN)
                             logging.warning(f"{ticker}: '{ti}' 처리 후에도 NaN 존재 -> 제외.")
                         else:
                             tech_indicators_to_add.append(ti)
                     else:
                         logging.warning(f"{ticker}: 기술 지표 '{ti}' 생성 안 됨 또는 전체가 NaN.")
            else:
                 logging.warning(f"{ticker}: TA 계산 위한 'close' 컬럼 없음(소문자 변환 후). 건너뜁니다.")
        except Exception as ta_err:
            logging.error(f"{ticker}: TA 계산 오류: {ta_err}\n{traceback.format_exc()}")
            tech_indicators_to_add = [] # 오류 발생 시 초기화

    # 4. 최종 데이터 검증 및 Prophet 준비
    if df_merged.empty or len(df_merged) < 30 or 'Date' not in df_merged.columns or 'Close' not in df_merged.columns:
        logging.error(f"Prophet 실패: 최종 데이터 부족/형식오류 ({len(df_merged)} 행).")
        return None, None, None, None
    logging.info(f"Prophet 학습 데이터 준비 완료 (Shape: {df_merged.shape}), 사용할 TA: {tech_indicators_to_add}")

    # --- Prophet 모델링 ---
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"}); df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=changepoint_prior_scale)
    
    all_regressors_for_fit = [] # 실제 학습에 사용될 regressor 리스트
    
    # 매크로 변수 추가
    macro_cols_for_prophet = [col for col in macro_cols if col in df_prophet.columns and df_prophet[col].notna().any()]
    if macro_cols_for_prophet:
        for col in macro_cols_for_prophet:
            if df_prophet[col].isnull().any(): # 이전 ffill/bfill 후에도 NaN이 남아있을 수 있음 (데이터 시작 부분)
                 df_prophet[col] = df_prophet[col].fillna(0); logging.warning(f"{ticker}: Macro Regressor '{col}' NaN 발견됨 -> 0으로 채움")
            m.add_regressor(col)
            all_regressors_for_fit.append(col)
        logging.info(f"{ticker}: FMP 매크로 Regressors 추가: {macro_cols_for_prophet}")

    # 기술적 지표 변수 추가
    tech_indicators_for_prophet = [col for col in tech_indicators_to_add if col in df_prophet.columns and df_prophet[col].notna().any()]
    if tech_indicators_for_prophet:
        for col in tech_indicators_for_prophet:
            if df_prophet[col].isnull().any(): # 이전 ffill/bfill 후에도 NaN이 남아있을 수 있음
                 df_prophet[col] = df_prophet[col].fillna(0); logging.warning(f"{ticker}: TA Regressor '{col}' NaN 발견됨 -> 0으로 채움")
            m.add_regressor(col)
            all_regressors_for_fit.append(col)
        logging.info(f"{ticker}: TA Regressors 추가: {tech_indicators_for_prophet}")

    # 5. 학습, 예측, CV 실행
    forecast_dict, fig_fcst, cv_path, mape = None, None, None, None
    try:
        logging.info(f"{ticker}: Prophet 학습 시작 (Regressors: {all_regressors_for_fit})...")
        
        # 학습 데이터에서 'ds', 'y' 및 모든 regressor 컬럼만 선택하고 NaN 행 제거
        cols_for_fit = ['ds', 'y'] + all_regressors_for_fit
        prophet_train_df = df_prophet[cols_for_fit].dropna() # NaN 있는 행 제거
        
        if len(prophet_train_df) < 30: # regressor 추가로 인해 데이터가 너무 줄었을 경우
             logging.error(f"Prophet fit 실패: 유효 데이터 부족 (NaN 제거 후 {len(prophet_train_df)} 행). Regressors: {all_regressors_for_fit}")
             # Regressor 없이 재시도 또는 에러 반환 결정 필요. 여기서는 에러 반환.
             return None, None, None, None 

        m.fit(prophet_train_df); logging.info(f"{ticker}: Prophet 학습 완료.")

        # Cross Validation Block
        try:
            data_len_days = (prophet_train_df['ds'].max() - prophet_train_df['ds'].min()).days
            initial_cv_days = max(60, int(data_len_days * 0.5)) # 최소 60일 또는 데이터의 50%
            period_cv_days = max(15, int(initial_cv_days * 0.2)) # 최소 15일 또는 initial의 20%
            horizon_cv_days = forecast_days # 예측 기간만큼 horizon

            initial_cv, period_cv, horizon_cv = f'{initial_cv_days} days', f'{period_cv_days} days', f'{horizon_cv_days} days'

            # CV 실행 가능한 최소 데이터 기간 확인
            if data_len_days >= initial_cv_days + horizon_cv_days:
                logging.info(f"Prophet CV 시작 (initial='{initial_cv}', period='{period_cv}', horizon='{horizon_cv}')...")
                df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel=None) # parallel=None으로 명시적 단일 프로세스
                logging.info("CV 완료.")
                df_p = performance_metrics(df_cv)
                mape = df_p["mape"].mean() * 100 if "mape" in df_p and not df_p["mape"].empty else None # MAPE 값 존재 및 비어있지 않은지 확인
                if mape is not None: logging.info(f"Prophet CV 평균 MAPE: {mape:.2f}%")
                else: logging.warning("CV 후 MAPE 계산 불가 (결과 없음 또는 빈 값).")

                fig_cv = plot_cross_validation_metric(df_cv, metric='mape') # Plotly 차트 객체 반환
                plt.title(f'{ticker} CV MAPE (Params: cp={changepoint_prior_scale})') # Matplotlib 타이틀 (Prophet 기본 plot이 matplotlib 기반일 경우)
                os.makedirs(FORECAST_FOLDER, exist_ok=True)
                cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_cp{changepoint_prior_scale}.png")
                # fig_cv가 matplotlib Figure 객체일 경우 savefig 사용
                if hasattr(fig_cv, 'savefig'):
                    fig_cv.savefig(cv_path); plt.close(fig_cv)
                else: # Plotly 객체일 경우, to_image 등으로 저장 (kaleido 필요)
                    try:
                        fig_cv.write_image(cv_path)
                    except Exception as img_e:
                        logging.warning(f"Plotly CV 차트 이미지 저장 실패: {img_e}. kaleido 설치 필요할 수 있음.")
                        cv_path = None # 저장 실패 시 경로 None

                if cv_path: logging.info(f"CV MAPE 차트 저장 완료: {cv_path}")

            else:
                logging.warning(f"{ticker}: 데이터 기간 부족 ({data_len_days}일 < {initial_cv_days + horizon_cv_days}일)하여 CV를 건너뜁니다.")
                cv_path = None; mape = None # CV 건너뛰면 mape도 없음
        except Exception as cv_e:
             logging.error(f"Prophet CV 중 오류 발생: {cv_e}\n{traceback.format_exc()}")
             cv_path = None; mape = None # CV 오류 시 mape도 없음

        # Future Prediction
        logging.info("미래 예측 시작...")
        future = m.make_future_dataframe(periods=forecast_days)
        
        # 미래 데이터프레임에 regressor 값 채우기
        if all_regressors_for_fit:
            if not prophet_train_df.empty: # 학습에 사용된 데이터프레임 기준
                 last_known_regressors = prophet_train_df[all_regressors_for_fit].iloc[-1]
                 for col in all_regressors_for_fit:
                     future[col] = last_known_regressors[col] # 마지막 값으로 미래 regressor 채움 (단순화된 방식)
                 
                 if future[all_regressors_for_fit].isnull().any().any():
                     logging.warning(f"미래 예측 Df에 NaN Regressor 존재 -> 0으로 채움.")
                     future[all_regressors_for_fit] = future[all_regressors_for_fit].fillna(0)
            else: # prophet_train_df가 비어있는 비정상적 상황 대비
                 logging.error("학습 데이터(prophet_train_df)가 비어있어 미래 Regressor 값을 설정할 수 없습니다.")
                 for col in all_regressors_for_fit:
                      future[col] = 0 # 안전장치로 0으로 채움
                 logging.warning("미래 Regressor 값을 0으로 채웁니다.")
        
        # 예측에 필요한 컬럼만 전달
        predict_cols = ['ds'] + all_regressors_for_fit
        forecast = m.predict(future[predict_cols]); logging.info("미래 예측 완료.")

        # ========== MODIFICATION START: Custom Prophet Plotly Chart ==========
        df_fcst_viz = forecast 
        
        fig_fcst = go.Figure()

        fig_fcst.add_trace(go.Scatter(
            x=df_fcst_viz['ds'], 
            y=df_fcst_viz['yhat_upper'],
            line=dict(width=0), 
            showlegend=False,
            name='상단 예측 범위' 
        ))

        fig_fcst.add_trace(go.Scatter(
            x=df_fcst_viz['ds'], 
            y=df_fcst_viz['yhat_lower'],
            fill='tonexty', 
            fillcolor='rgba(0,100,80,0.1)', 
            line=dict(width=0), 
            showlegend=False,
            name='하단 예측 범위' 
        ))
        
        fig_fcst.add_trace(go.Scatter(
            x=df_fcst_viz['ds'], 
            y=df_fcst_viz['yhat'],
            line=dict(color='blue'), 
            name='예측값'
        ))

        # 학습에 사용된 실제 값(prophet_train_df)을 차트에 추가
        if not prophet_train_df.empty:
            actual_data_trace = prophet_train_df[prophet_train_df['ds'] <= df_fcst_viz['ds'].max()] 
            fig_fcst.add_trace(go.Scatter(
                x=actual_data_trace['ds'],
                y=actual_data_trace['y'],
                mode='markers',
                marker=dict(color='black', size=3),
                name='실제 종가'
            ))

        if avg_price is not None and avg_price > 0:
            fig_fcst.add_hline(
                y=avg_price,
                line_dash="dot",
                line_color="red",
                annotation_text=f"평단가: ${avg_price:.2f}",
                annotation_position="bottom right" 
            )

        fig_fcst.update_layout(
            title=f'{ticker} 가격 예측 (Prophet | cp={changepoint_prior_scale})',
            xaxis_title="날짜",
            yaxis_title="가격 ($)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20) 
        )
        logging.info(f"Prophet 예측 커스텀 Plotly Figure 생성 완료.")
        # ========== MODIFICATION END ==========

        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records')
        for rec in forecast_dict:
             rec['ds'] = rec['ds'].strftime('%Y-%m-%d')

        return forecast_dict, fig_fcst, cv_path, mape

    except Exception as e:
        logging.error(f"Prophet 학습/예측 오류: {e}\n{traceback.format_exc()}")
        return None, None, cv_path, mape # CV는 성공했을 수 있으므로 cv_path와 mape는 반환


# --- 메인 분석 함수 ---
def analyze_stock(ticker, analysis_period_years=2, forecast_days=30, num_trend_periods=4, changepoint_prior_scale=0.05, avg_price=None): # MODIFIED: Added avg_price
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다. (FMP API 기반, Prophet은 FMP Macro)"""
    logging.info(f"--- {ticker} 주식 분석 시작 (FMP, cp_prior={changepoint_prior_scale}, avg_p_main={avg_price}) ---")
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

    # safe_load_fmp_price_data를 사용하여 주가 데이터 로드
    df_stock_full = safe_load_fmp_price_data(ticker, start_date=start_date_str, end_date=end_date_str)
    stock_data_valid = not df_stock_full.empty and 'Close' in df_stock_full.columns and df_stock_full['Close'].notna().any()


    current_price_str = "N/A"; data_points_count = 0
    if stock_data_valid:
        # df_stock_full.index는 DatetimeIndex여야 함
        last_close_series = df_stock_full['Close'].dropna()
        if not last_close_series.empty:
             last_close = last_close_series.iloc[-1]
             current_price_str = f"{last_close:.2f}"
        else:
             current_price_str = "N/A" # 모든 종가가 NaN인 경우
        
        data_points_count = len(df_stock_full) # 전체 데이터 포인트 수
        output_results['current_price'] = current_price_str
        output_results['data_points'] = data_points_count
    else:
        output_results['current_price'] = "N/A"; output_results['data_points'] = 0
        logging.warning(f"{ticker}: 분석 기간 내 유효한 주가 정보 없음 (FMP).")

    output_results['analysis_period_start'] = start_date_str
    output_results['analysis_period_end'] = end_date_str
    
    # plot_stock_chart 함수 호출 (위에서 정의됨)
    output_results['stock_chart_fig'] = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str) or None
    
    # fmp_api.py 에 정의된 함수들 호출 (가정)
    output_results['fundamentals'] = get_fundamental_data(ticker) or {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods) or []
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods) or []
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend(ticker, num_periods=num_trend_periods) or []
    output_results['current_ratio_trend'] = get_current_ratio_trend(ticker, num_periods=num_trend_periods) or []
    output_results['news_sentiment'] = get_news_sentiment(ticker) or ["뉴스 분석 실패"]
    
    fg_data = get_fear_greed_index() # 튜플 (value, classification) 또는 None 반환 가정
    if isinstance(fg_data, tuple) and len(fg_data) == 2:
        fg_value, fg_class = fg_data
        output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class}
    else:
        output_results['fear_greed_index'] = "N/A"


    output_results['prophet_forecast'] = "예측 불가"
    output_results['forecast_fig'] = None
    output_results['cv_plot_path'] = None
    output_results['mape'] = None
    output_results['warn_high_mape'] = False

    if stock_data_valid and data_points_count >= 30: # 최소 30일 데이터 필요
        forecast_result = run_prophet_forecast(
            ticker, start_date=start_date_str, end_date=end_date_str, # run_prophet_forecast가 내부적으로 safe_load_fmp_price_data 사용
            forecast_days=forecast_days,
            changepoint_prior_scale=changepoint_prior_scale,
            avg_price=avg_price # MODIFIED: Pass avg_price
        )
        if forecast_result and isinstance(forecast_result, tuple) and len(forecast_result) == 4:
            fc_list, fc_fig, cv_path, mape_val = forecast_result
            output_results['prophet_forecast'] = fc_list if fc_list else "예측 결과 없음"
            output_results['forecast_fig'] = fc_fig 
            output_results['cv_plot_path'] = cv_path
            if isinstance(mape_val, (int, float)): # mape_val이 숫자인지 확인
                 output_results['mape'] = f"{mape_val:.2f}%"
                 output_results['warn_high_mape'] = mape_val > 20 # 예: MAPE 20% 초과 시 경고
            elif cv_path is None and mape_val is None and fc_list: # CV는 실패/건너뛰었지만 예측은 성공한 경우
                 output_results['mape'] = "CV 실패/건너뜀" 
            else: # 그 외 경우 (예측 자체가 실패했거나 mape가 문자열 등)
                 output_results['mape'] = "MAPE 계산 불가"
        else:
            output_results['prophet_forecast'] = "예측 실행 오류/결과 없음"
            logging.error(f"{ticker}: run_prophet_forecast 함수 비정상 반환: {forecast_result}")
    else:
        msg = f"데이터 부족({data_points_count})" if data_points_count < 30 else "유효한 주가 정보 없음"
        output_results['prophet_forecast'] = f"{msg}으로 예측 불가"
        logging.warning(f"{ticker}: {msg} - Prophet 예측 건너뜁니다.") 

    logging.info(f"--- {ticker} 주식 분석 완료 (FMP 기반) ---")
    return output_results


# --- 메인 실행 부분 (테스트용) ---
if __name__ == "__main__":
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    # FMP API 키가 .env 또는 환경변수에 설정되어 있어야 합니다.
    # fmp_api.py 내부에서 FMP_API_KEY를 로드하는 로직이 필요합니다.
    if not fmp_api.FMP_API_KEY: # fmp_api 모듈에 API 키가 로드되었는지 확인하는 변수/함수 필요 (예시)
        print("오류: FMP_API_KEY가 설정되지 않았습니다. .env 파일을 확인하거나 환경변수를 설정해주세요.")
        exit()
        
    print("Testing with FULL FMP API integration...")
    target_ticker = "AAPL"; cps_test = 0.1
    test_avg_price = 165.0 
    # test_avg_price = None 

    print(f"Attempting analysis for: {target_ticker} (using FMP for all data), Avg Price for Viz: {test_avg_price}")
    
    try:
        test_results = analyze_stock(
            ticker=target_ticker, 
            analysis_period_years=1, 
            forecast_days=30, 
            num_trend_periods=4, 
            changepoint_prior_scale=cps_test,
            avg_price=test_avg_price 
        )
    except Exception as e:
        print(f"analyze_stock 실행 중 최상위 오류 발생: {e}\n{traceback.format_exc()}")
        test_results = {"error": f"분석 중 심각한 오류: {e}"}
        
    
    print(f"\n--- 테스트 실행 결과 요약 ({target_ticker}, Changepoint Prior: {cps_test}) ---")
    if test_results and isinstance(test_results, dict) and "error" not in test_results:
        for key, value in test_results.items():
            if 'fig' in key and value is not None and isinstance(value, go.Figure):  # Plotly Figure 객체인지 확인
                print(f"- {key.replace('_',' ').title()}: Plotly Figure 생성됨")
                # 로컬에서 차트 확인 (Streamlit 앱이 아닐 경우)
                # if key == 'forecast_fig' or key == 'stock_chart_fig':
                #     value.show() 
            elif key == 'fundamentals' and isinstance(value, dict): print("- Fundamentals (FMP):"); [print(f"    - {k}: {str(v)[:120]}") for k, v in value.items()] # str(v)로 안전하게 변환
            elif '_trend' in key and isinstance(value, list):
                print(f"- {key.replace('_',' ').title()} ({len(value)} 항목) [FMP]:")
                if value:
                     [print(f"    - {str(item)[:120]}") for item in value[:3]] # str(item)으로 안전하게 변환
                     if len(value) > 3: print("      ...")
                else: print("    - 데이터 없음")
            elif key == 'prophet_forecast':
                status = value
                if isinstance(value, list) and value:
                     status = f"{len(value)}일 예측 생성됨 (첫 날: {value[0]['ds']} 예측: {value[0]['yhat']:.2f})"
                elif isinstance(value, list) and not value:
                     status = "예측 결과 없음 (빈 리스트)"
                print(f"- Prophet Forecast (FMP Macro): {status}")
            elif key == 'news_sentiment':
                status = "뉴스 분석 실패/오류"
                if isinstance(value, list) and value:
                     # 첫 번째 뉴스 요약 또는 감성 점수 등 표시
                     status = str(value[0])[:200] + ('...' if len(str(value[0])) > 200 else '')

                print(f"- News Sentiment (FMP+TextBlob): {status}")
            elif key == 'cv_plot_path':
                 print(f"- Cv Plot Path: {value if value else '생성 안 됨/실패/건너뜀'}") 
            elif key == 'mape':
                 print(f"- MAPE: {value if value else 'N/A'}")
            else:
                 print(f"- {key.replace('_',' ').title()}: {str(value)[:200]}") # 기타 값들도 str로 변환 및 길이 제한
    elif test_results and "error" in test_results: print(f"분석 중 오류 발생: {test_results['error']}")
    else: print("테스트 분석 실패 (결과 없음 또는 알 수 없는 오류).")
    print("\n--- 테스트 실행 종료 ---")