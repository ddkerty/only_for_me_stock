# stock_analysis.py (FMP Macro Data & FRED 제거 최종 적용 버전 - 수정됨 with Plotly enhancements)

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
from fmp_api import get_price_data as get_stock_data


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

# --- (기존 함수들은 변경 없이 유지됩니다. get_fear_greed_index, get_stock_data, ...) ---
# ... (모든 함수들 get_fear_greed_index 부터 get_current_ratio_trend 까지 동일하게 유지) ...
# (plot_stock_chart, get_news_sentiment 함수들도 동일)

# Example functions (get_fear_greed_index, get_stock_data, get_macro_data_fmp, format_market_cap,
# get_fundamental_data, get_operating_margin_trend, get_roe_trend, get_debt_to_equity_trend,
# get_current_ratio_trend, plot_stock_chart, get_news_sentiment are assumed to be the same as in the original file)
# For brevity, I will skip them here and jump to the modified functions.
# 오류 수정된 get_stock_data 사용 예시 (stock_analysis.py 내 적용용)

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
        raw_data = get_price_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if raw_data is None or not isinstance(raw_data, list) or len(raw_data) == 0:
            logging.error(f"{ticker}: get_price_data() 결과 없음 또는 형식 오류")
            return pd.DataFrame()

        df = pd.DataFrame(raw_data)
        required_cols = ['date', 'close']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"{ticker}: 필수 컬럼 누락: {required_cols}")
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df

    except Exception as e:
        logging.error(f"{ticker}: 가격 데이터 로드 실패: {e}")
        return pd.DataFrame()


# --- Prophet Forecasting (FMP Macro Data 적용) ---
def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, changepoint_prior_scale=0.05, avg_price=None): # MODIFIED: Added avg_price
    """Prophet 예측 (FMP Stock Data + FMP Macro Data + TA Regressors) - Custom Plotly Chart"""
    logging.info(f"{ticker}: Prophet 예측 시작 (FMP Data, cp_prior={changepoint_prior_scale}, avg_p_viz={avg_price})...")
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d")

    # 1. 초기 주가 데이터 로딩 (FMP 사용)
    df_stock_initial = safe_load_fmp_price_data(ticker, start_date=start_date, end_date=end_date)
    if df_stock_initial is None or df_stock_initial.empty:
        logging.error(f"{ticker}: Prophet 예측 위한 FMP 주가 데이터 로드 실패.")
        return None, None, None, None # 4 Nones 반환
    try: # 데이터 처리
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
        logging.error(f"{ticker}: 주가 데이터 처리 중 오류: {proc_err}\n{traceback.format_exc()}"); return None, None, None, None

    # 2. FMP 기반 매크로 데이터 로딩 및 병합 (get_macro_data_fmp 호출)
    macro_data_dict = get_macro_data_fmp()
    macro_cols = ["VIX", "US10Y", "US13W", "DXY"]
    df_merged = df_stock_processed.copy()
    if macro_data_dict:
        logging.info(f"{ticker}: FMP 매크로 데이터 로딩 성공: {macro_data_dict}")
        for col in macro_cols:
            value = macro_data_dict.get(col)
            df_merged[col] = pd.to_numeric(value, errors='coerce') if value is not None else np.nan
            if pd.isna(df_merged[col]).all() and value is not None :
                 logging.warning(f"{ticker}: FMP 매크로 데이터 '{col}' 값 ({value}) 숫자 변환 실패.")
            elif value is None:
                 logging.warning(f"{ticker}: FMP 매크로 데이터 '{col}' 누락/None.")
        df_merged[macro_cols] = df_merged[macro_cols].ffill().bfill()
        logging.info(f"{ticker}: FMP 매크로 변수 병합 및 NaN 처리 완료.")
    else:
        logging.warning(f"{ticker}: FMP 매크로 데이터 없음. 주가 데이터만 사용.")
        for col in macro_cols: df_merged[col] = np.nan

    # 3. 기술적 지표 계산 (pandas_ta 사용 - 소문자 컬럼명 주의)
    logging.info(f"{ticker}: 기술적 지표 계산 시작...")
    tech_indicators_to_add = []
    base_ta_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df_merged.columns for c in base_ta_cols):
        logging.warning(f"{ticker}: TA 계산 위한 기본 컬럼 부족 ({[c for c in base_ta_cols if c not in df_merged.columns]}). 건너뜁니다.")
    else:
        df_merged_ta = df_merged.rename(columns={c: c.lower() for c in base_ta_cols})
        try:
            if 'close' in df_merged_ta.columns:
                 df_merged_ta.ta.rsi(close='close', length=14, append=True, col_names=('RSI_14',))
                 df_merged_ta.ta.macd(close='close', fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9','MACDs_12_26_9'))
                 logging.info(f"{ticker}: RSI, MACD 계산 완료.")
                 df_merged['RSI_14'] = df_merged_ta.get('RSI_14')
                 df_merged['MACDs_12_26_9'] = df_merged_ta.get('MACDs_12_26_9')
                 tech_indicators_candidates = ['RSI_14', 'MACDs_12_26_9']
                 for ti in tech_indicators_candidates:
                     if ti in df_merged.columns:
                         if df_merged[ti].isnull().any():
                             df_merged[ti] = df_merged[ti].ffill().bfill()
                             logging.info(f"{ticker}: '{ti}' NaN ffill/bfill 처리.")
                         if df_merged[ti].isnull().any():
                             logging.warning(f"{ticker}: '{ti}' 처리 후에도 NaN 존재 -> 제외.")
                         else:
                             tech_indicators_to_add.append(ti)
                     else:
                         logging.warning(f"{ticker}: 기술 지표 '{ti}' 생성 안 됨.")
            else:
                 logging.warning(f"{ticker}: TA 계산 위한 'close' 컬럼 없음(소문자 변환 후). 건너뜁니다.")
        except Exception as ta_err:
            logging.error(f"{ticker}: TA 계산 오류: {ta_err}\n{traceback.format_exc()}")
            tech_indicators_to_add = []

    # 4. 최종 데이터 검증 및 Prophet 준비
    if df_merged.empty or len(df_merged) < 30 or 'Date' not in df_merged.columns or 'Close' not in df_merged.columns:
        logging.error(f"Prophet 실패: 최종 데이터 부족/형식오류 ({len(df_merged)} 행).")
        return None, None, None, None
    logging.info(f"Prophet 학습 데이터 준비 완료 (Shape: {df_merged.shape})")

    # --- Prophet 모델링 ---
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"}); df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=changepoint_prior_scale)
    all_regressors = []
    macro_cols_available = [col for col in macro_cols if col in df_prophet.columns and df_prophet[col].notna().any()]
    if macro_cols_available:
        for col in macro_cols_available:
            if df_prophet[col].isnull().any():
                 df_prophet[col] = df_prophet[col].fillna(0); logging.warning(f"{ticker}: Macro '{col}' NaN 발견됨 -> 0으로 채움")
            m.add_regressor(col)
        all_regressors.extend(macro_cols_available); logging.info(f"{ticker}: FMP 매크로 Regressors: {macro_cols_available}")
    if tech_indicators_to_add:
        for col in tech_indicators_to_add:
            if col in df_prophet.columns:
                if df_prophet[col].isnull().any():
                    df_prophet[col] = df_prophet[col].fillna(0); logging.warning(f"{ticker}: TA '{col}' NaN 발견됨 -> 0으로 채움")
                m.add_regressor(col)
                all_regressors.append(col)
            else:
                 logging.warning(f"{ticker}: TA Regressor '{col}'가 df_prophet에 없음.")
        logging.info(f"{ticker}: TA Regressors: {tech_indicators_to_add}")

    # 5. 학습, 예측, CV 실행
    forecast_dict, fig_fcst, cv_path, mape = None, None, None, None
    try:
        logging.info(f"{ticker}: Prophet 학습 시작 (Regressors: {all_regressors})...")
        final_regressors = [reg for reg in all_regressors if reg in df_prophet.columns]
        cols_for_fit = ['ds', 'y'] + final_regressors
        missing_fit_cols = [col for col in cols_for_fit if col not in df_prophet.columns]
        if missing_fit_cols:
             logging.error(f"Prophet fit 실패: 학습 컬럼 부족: {missing_fit_cols}")
             return None, None, None, None

        prophet_train_df = df_prophet[cols_for_fit].dropna()
        if len(prophet_train_df) < 30:
             logging.error(f"Prophet fit 실패: 유효 데이터 부족 (NaN 제거 후 {len(prophet_train_df)} 행).")
             return None, None, None, None

        m.fit(prophet_train_df); logging.info(f"{ticker}: Prophet 학습 완료.")

        # Cross Validation Block
        try:
            data_len_days = (prophet_train_df['ds'].max() - prophet_train_df['ds'].min()).days
            initial_cv_days = max(60, int(data_len_days * 0.5))
            period_cv_days = max(15, int(initial_cv_days * 0.2))
            horizon_cv_days = forecast_days
            initial_cv, period_cv, horizon_cv = f'{initial_cv_days} days', f'{period_cv_days} days', f'{horizon_cv_days} days'

            if data_len_days >= initial_cv_days + horizon_cv_days:
                logging.info(f"Prophet CV 시작 (initial='{initial_cv}', period='{period_cv}', horizon='{horizon_cv}')...")
                df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel=None)
                logging.info("CV 완료.")
                df_p = performance_metrics(df_cv)
                mape = df_p["mape"].mean() * 100 if "mape" in df_p else None
                if mape is not None: logging.info(f"Prophet CV 평균 MAPE: {mape:.2f}%")
                else: logging.warning("CV 후 MAPE 계산 불가.")

                fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
                plt.title(f'{ticker} CV MAPE (Params: cp={changepoint_prior_scale})')
                os.makedirs(FORECAST_FOLDER, exist_ok=True)
                cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_cp{changepoint_prior_scale}.png")
                fig_cv.savefig(cv_path); plt.close(fig_cv)
                logging.info(f"CV MAPE 차트 저장 완료: {cv_path}")
            else:
                logging.warning(f"{ticker}: 데이터 기간 부족 ({data_len_days}일 < {initial_cv_days + horizon_cv_days}일)하여 CV를 건너뜁니다.")
                cv_path = None
        except Exception as cv_e:
             logging.error(f"Prophet CV 중 오류 발생: {cv_e}\n{traceback.format_exc()}")
             cv_path = None; mape = None

        # Future Prediction
        logging.info("미래 예측 시작...")
        future = m.make_future_dataframe(periods=forecast_days)
        if final_regressors:
            if not prophet_train_df.empty:
                 last_known_regressors = prophet_train_df[final_regressors].iloc[-1]
                 for col in final_regressors:
                     future[col] = last_known_regressors[col]
                 if future[final_regressors].isnull().any().any():
                     logging.warning(f"미래 예측 Df에 NaN Regressor 존재 -> 0으로 채움.")
                     future[final_regressors] = future[final_regressors].fillna(0)
            else:
                 logging.error("학습 데이터가 비어있어 미래 Regressor 값을 설정할 수 없습니다.")
                 for col in final_regressors:
                      future[col] = 0
                 logging.warning("미래 Regressor 값을 0으로 채웁니다.")

        predict_cols = ['ds'] + final_regressors
        missing_predict_cols = [col for col in predict_cols if col not in future.columns]
        if missing_predict_cols:
             logging.error(f"Predict 실패: 예측 데이터 컬럼 부족: {missing_predict_cols}")
             return None, None, None, mape

        forecast = m.predict(future[predict_cols]); logging.info("미래 예측 완료.")

        # ========== MODIFICATION START: Custom Prophet Plotly Chart ==========
        # The original `plot_plotly(m, forecast)` is replaced by this:
        
        # df_fcst_viz is the 'forecast' DataFrame containing ds, yhat, yhat_lower, yhat_upper
        df_fcst_viz = forecast 
        
        fig_fcst = go.Figure()

        # Upper prediction band (invisible line, used for fill area)
        fig_fcst.add_trace(go.Scatter(
            x=df_fcst_viz['ds'], 
            y=df_fcst_viz['yhat_upper'],
            line=dict(width=0), # Make line invisible
            showlegend=False,
            name='상단 예측 범위' # Name for hover
        ))

        # Lower prediction band (filled to the upper band)
        fig_fcst.add_trace(go.Scatter(
            x=df_fcst_viz['ds'], 
            y=df_fcst_viz['yhat_lower'],
            fill='tonexty', # Fill area to the previous trace (yhat_upper)
            fillcolor='rgba(0,100,80,0.1)', # Light fill color
            line=dict(width=0), # Make line invisible
            showlegend=False,
            name='하단 예측 범위' # Name for hover
        ))
        
        # Actual forecast line
        fig_fcst.add_trace(go.Scatter(
            x=df_fcst_viz['ds'], 
            y=df_fcst_viz['yhat'],
            line=dict(color='blue'), # Or 'rgb(0,100,80)' to match fill
            name='예측값'
        ))

        # Add historical actuals to the plot for context (optional, but good)
        # Ensure 'y' (actual close prices) is from the training data period
        actual_data_trace = prophet_train_df[prophet_train_df['ds'] <= df_fcst_viz['ds'].max()] # Show actuals up to last prediction date
        fig_fcst.add_trace(go.Scatter(
            x=actual_data_trace['ds'],
            y=actual_data_trace['y'],
            mode='markers',
            marker=dict(color='black', size=3),
            name='실제 종가'
        ))


        # Add average price horizontal line if provided
        if avg_price is not None and avg_price > 0:
            fig_fcst.add_hline(
                y=avg_price,
                line_dash="dot",
                line_color="red",
                annotation_text=f"평단가: ${avg_price:.2f}",
                annotation_position="bottom right" # Changed to bottom right to avoid overlap with title
            )

        fig_fcst.update_layout(
            title=f'{ticker} 가격 예측 (Prophet | cp={changepoint_prior_scale})',
            xaxis_title="날짜",
            yaxis_title="가격 ($)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20) # Adjusted top margin for title
        )
        logging.info(f"Prophet 예측 커스텀 Plotly Figure 생성 완료.")
        # ========== MODIFICATION END ==========

        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records')
        for rec in forecast_dict:
             rec['ds'] = rec['ds'].strftime('%Y-%m-%d')

        return forecast_dict, fig_fcst, cv_path, mape

    except Exception as e:
        logging.error(f"Prophet 학습/예측 오류: {e}\n{traceback.format_exc()}")
        return None, None, cv_path, mape


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

    df_stock_full = safe_load_fmp_price_data(ticker, start_date=start_date_str, end_date=end_date_str)
    stock_data_valid = not df_stock_full.empty and 'Close' in df_stock_full.columns

    current_price_str = "N/A"; data_points_count = 0
    if stock_data_valid:
        last_close = df_stock_full['Close'].dropna().iloc[-1] if not df_stock_full['Close'].dropna().empty else None
        current_price_str = f"{last_close:.2f}" if last_close is not None else "N/A"
        data_points_count = len(df_stock_full)
        output_results['current_price'] = current_price_str
        output_results['data_points'] = data_points_count
    else:
        output_results['current_price'] = "N/A"; output_results['data_points'] = 0
        logging.warning(f"{ticker}: 분석 기간 내 유효한 주가 정보 없음 (FMP).")

    output_results['analysis_period_start'] = start_date_str
    output_results['analysis_period_end'] = end_date_str
    output_results['stock_chart_fig'] = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str) or None
    output_results['fundamentals'] = get_fundamental_data(ticker) or {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods) or []
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods) or []
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend(ticker, num_periods=num_trend_periods) or []
    output_results['current_ratio_trend'] = get_current_ratio_trend(ticker, num_periods=num_trend_periods) or []
    output_results['news_sentiment'] = get_news_sentiment(ticker) or ["뉴스 분석 실패"]
    fg_value, fg_class = get_fear_greed_index()
    output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    output_results['prophet_forecast'] = "예측 불가"
    output_results['forecast_fig'] = None
    output_results['cv_plot_path'] = None
    output_results['mape'] = None
    output_results['warn_high_mape'] = False

    if stock_data_valid and data_points_count > 30:
        forecast_result = run_prophet_forecast(
            ticker, start_date=start_date_str, end_date=end_date_str,
            forecast_days=forecast_days,
            changepoint_prior_scale=changepoint_prior_scale,
            avg_price=avg_price # MODIFIED: Pass avg_price
        )
        if forecast_result and isinstance(forecast_result, tuple) and len(forecast_result) == 4:
            fc_list, fc_fig, cv_path, mape_val = forecast_result
            output_results['prophet_forecast'] = fc_list if fc_list else "예측 결과 없음"
            output_results['forecast_fig'] = fc_fig # This is now the custom figure
            output_results['cv_plot_path'] = cv_path
            if isinstance(mape_val, (int, float)):
                 output_results['mape'] = f"{mape_val:.2f}%"
                 output_results['warn_high_mape'] = mape_val > 20
            elif cv_path is None and mape_val is None:
                 output_results['mape'] = "CV 실패/건너뜀" # Corrected Korean
            else:
                 output_results['mape'] = "MAPE 계산 불가"
        else:
            output_results['prophet_forecast'] = "예측 실행 오류/결과 없음"
            logging.error(f"{ticker}: run_prophet_forecast 함수 비정상 반환: {forecast_result}")
    else:
        msg = f"데이터 부족({data_points_count})" if data_points_count <= 30 else "유효한 주가 정보 없음"
        output_results['prophet_forecast'] = f"{msg}으로 예측 불가"
        logging.warning(f"{ticker}: {msg} - Prophet 예측 건너뜁니다.") # Corrected Korean

    logging.info(f"--- {ticker} 주식 분석 완료 (FMP 기반) ---")
    return output_results


# --- 메인 실행 부분 (테스트용) ---
if __name__ == "__main__":
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    print("Testing with FULL FMP API integration...")
    target_ticker = "AAPL"; cps_test = 0.1
    # MODIFIED: Example average price for testing the new chart feature
    test_avg_price = 165.0 
    # test_avg_price = None # Test without average price line

    print(f"Attempting analysis for: {target_ticker} (using FMP for all data except F&G), Avg Price for Viz: {test_avg_price}")
    
    test_results = analyze_stock(
        ticker=target_ticker, 
        analysis_period_years=1, 
        forecast_days=30, # Increased for better visualization
        num_trend_periods=4, 
        changepoint_prior_scale=cps_test,
        avg_price=test_avg_price # MODIFIED: Pass the test average price
    )
    
    print(f"\n--- 테스트 실행 결과 요약 ({target_ticker}, Changepoint Prior: {cps_test}) ---")
    if test_results and isinstance(test_results, dict) and "error" not in test_results:
        for key, value in test_results.items():
            if 'fig' in key and value is not None: 
                print(f"- {key.replace('_',' ').title()}: Plotly Figure 생성됨")
                # 간단히 차트를 로컬에서 보려면:
                # if key == 'forecast_fig':
                #     value.show() # This will open the chart in a browser if run locally
            elif key == 'fundamentals' and isinstance(value, dict): print("- Fundamentals (FMP):"); [print(f"    - {k}: {v}"[:120]) for k, v in value.items()]
            elif '_trend' in key and isinstance(value, list):
                print(f"- {key.replace('_',' ').title()} ({len(value)} 분기) [FMP]:")
                if value:
                     [print(f"    - {item}") for item in value[:3]]
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
                     status = value[0] 
                print(f"- News Sentiment (FMP+TextBlob): {status}")
            elif key == 'cv_plot_path':
                 print(f"- Cv Plot Path: {value if value else '생성 안 됨/실패/건너뜀'}") # Corrected Korean
            elif key == 'mape':
                 print(f"- MAPE: {value if value else 'N/A'}")
            else:
                 print(f"- {key.replace('_',' ').title()}: {value}")
    elif test_results and "error" in test_results: print(f"분석 중 오류 발생: {test_results['error']}")
    else: print("테스트 분석 실패 (결과 없음 또는 알 수 없는 오류).")
    print("\n--- 테스트 실행 종료 ---")