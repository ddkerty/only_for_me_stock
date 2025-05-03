# -*- coding: utf-8 -*-
# Combined app.py V2.2 (SyntaxError Fix, FRED Removed, Tech Call Fixed - Corrected)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import traceback
import plotly.graph_objects as go
import numpy as np
import logging
import requests # FMP API 호출에 필요

# --- FMP API 및 분석 모듈 임포트 ---
try:
    import fmp_api # FMP API 래퍼 모듈
    import stock_analysis as sa # 종합 분석 로직
except ImportError as e:
    st.error(f"필수 API 또는 분석 모듈 로딩 실패: {e}. 'fmp_api.py'와 'stock_analysis.py' 파일이 있는지 확인하세요.")
    st.stop()

# --- 기술 분석 관련 함수 임포트 ---
try:
    # short_term_analysis.py 에 함수들이 올바르게 정의되어 있는지 확인 필요
    from short_term_analysis import interpret_fibonacci, calculate_rsi, calculate_macd
    from technical_interpret import interpret_technical_signals
except ImportError as e:
    st.error(f"기술 분석 모듈 로딩 실패: {e}. 'short_term_analysis.py'와 'technical_interpret.py' 파일이 있는지 확인하세요.")
    # 필요시 st.stop()

# --- 기본 경로 설정 및 로깅 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()

# --- 기술 분석 지표 계산 함수들 (기존 코드 유지) ---
def calculate_vwap(df):
    """VWAP 계산 (Volume 필요)"""
    df = df.copy(); required_cols = ['High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: raise ValueError(f"VWAP 계산 실패: 컬럼 부족 ({missing_cols})")
    if 'Volume' not in df.columns or df['Volume'].isnull().all() or (df['Volume'] == 0).all():
        df['VWAP'] = np.nan; logging.warning(f"Ticker {df.attrs.get('ticker', '')}: VWAP 계산 불가 (거래량 데이터 부족 또는 0)"); return df
    df['Volume'] = df['Volume'].fillna(0); df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']; df['cumulative_volume'] = df['Volume'].cumsum()
    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
    df['VWAP'] = np.where(df['cumulative_volume'] > 0, df['cumulative_tp_volume'] / df['cumulative_volume'], np.nan)
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """볼린저 밴드 계산 (Close 필요)"""
    df = df.copy(); required_col = 'Close'
    if required_col not in df.columns or df[required_col].isnull().all():
        st.warning(f"BB 계산 실패: '{required_col}' 컬럼 없음 또는 데이터 없음."); df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan; return df
    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        st.warning(f"BB 계산 위한 유효 데이터({len(valid_close)}개)가 기간({window}개)보다 부족."); df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan
    else:
        df['MA20'] = df[required_col].rolling(window=window, min_periods=window).mean(); df['STD20'] = df[required_col].rolling(window=window, min_periods=window).std()
        df['Upper'] = df['MA20'] + num_std * df['STD20']; df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

# --- 차트 생성 함수 (기존 코드 유지) ---
def plot_technical_chart(df, ticker):
    """기술적 분석 지표 통합 차트 생성 (VWAP, Bollinger Band, Fibonacci, RSI, MACD 포함)"""
    fig = go.Figure(); required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols) or df[required_candle_cols].isnull().all(axis=None):
        st.error(f"캔들차트 필요 컬럼({required_candle_cols}) 없음/데이터 없음."); return fig
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))
    if 'VWAP' in df.columns and df['VWAP'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns: st.caption("VWAP 데이터 없음/표시 불가.")
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)'))
    elif 'Upper' in df.columns: st.caption("볼린저 밴드 데이터 없음/표시 불가.")
    valid_price_df = df.dropna(subset=['High', 'Low'])
    if not valid_price_df.empty:
        min_price = valid_price_df['Low'].min(); max_price = valid_price_df['High'].max(); diff = max_price - min_price
        if diff > 0:
            levels = {'0.0 (High)': max_price, '0.236': max_price - 0.236 * diff, '0.382': max_price - 0.382 * diff, '0.5': max_price - 0.5 * diff, '0.618': max_price - 0.618 * diff, '1.0 (Low)': min_price}
            fib_colors = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels.items(): fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: ${v:.2f}", line_color=fib_colors.get(k, 'navy'), annotation_position="bottom right", annotation_font_size=10)
        else: st.caption("기간 내 가격 변동 없어 피보나치 미표시.")
    else: st.caption("피보나치 레벨 계산 불가.")
    if 'RSI' in df.columns and df['RSI'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='purple', width=1), yaxis='y2'))
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines', line=dict(color='teal'), yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', mode='lines', line=dict(color='orange'), yaxis='y3'))
        if 'MACD_hist' in df.columns: colors = ['lightgreen' if v >= 0 else 'lightcoral' for v in df['MACD_hist']]; fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color=colors, yaxis='y3'))
    fig.update_layout(title=f"{ticker} - 기술적 분석 통합 차트", xaxis_title="날짜 / 시간", yaxis=dict(domain=[0.4, 1], title="가격 ($)"), yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right', showgrid=False), yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right', showgrid=False), xaxis_rangeslider_visible=False, legend_title_text="지표", hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
    return fig

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="종합 주식 분석 FMP 버전", layout="wide", initial_sidebar_state="expanded")

# --- FMP API 키 로드 및 확인 (FRED 키 제거) ---
fmp_key_loaded = False
sidebar_status = st.sidebar.empty()
final_status_message_displayed = False

secrets_available = hasattr(st, 'secrets')
if secrets_available:
    try:
        fmp_secret_key = st.secrets.get("FMP_API_KEY")
        if fmp_secret_key:
            fmp_api.FMP_API_KEY = fmp_secret_key; fmp_key_loaded = True
            # Key loaded from secrets, clear initial message if shown
            sidebar_status.empty()
        else:
            sidebar_status.warning("Secrets에 FMP API 키가 없습니다.")
    except Exception as e:
        sidebar_status.error(f"Secrets 로드 오류: {e}"); final_status_message_displayed = True

# Try .env only if not loaded from secrets
if not fmp_key_loaded:
    sidebar_status.info("Secrets에 키 없음. .env 파일 확인 중...")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            fmp_env_key = os.getenv("FMP_API_KEY")
            if fmp_env_key:
                fmp_api.FMP_API_KEY = fmp_env_key; fmp_key_loaded = True
                sidebar_status.success("API 키 로드 완료 (.env)")
                final_status_message_displayed = True
            else:
                sidebar_status.error(".env 파일 내 FMP API 키 누락 또는 로드 실패."); final_status_message_displayed = True
        else:
             # If .env doesn't exist AND we didn't load from secrets, show final error
            if not secrets_available:
                sidebar_status.error(".env 파일 없음, Secrets에도 FMP 키 없음."); final_status_message_displayed = True
            # If .env doesn't exist but we ALREADY loaded from secrets, clear the "checking .env" message
            elif fmp_key_loaded:
                sidebar_status.empty()

    except Exception as e:
        sidebar_status.error(f".env 로드 오류: {e}"); final_status_message_displayed = True

# --- 종합 분석 가능 여부 결정 및 최종 메시지 ---
comprehensive_analysis_possible = fmp_key_loaded
if not comprehensive_analysis_possible and not final_status_message_displayed:
    st.sidebar.error("FMP API 키 로드 실패! '종합 분석' 기능이 제한됩니다.")
elif comprehensive_analysis_possible and not final_status_message_displayed:
    # If key loaded but no other message shown, show success
    sidebar_status.success("API 키 로드 완료.")


# --- 사이드바 설정 ---
with st.sidebar:
    # ... (후원계좌, 링크 등 유지) ...
    st.title("📊 주식 분석 도구 (FMP API)")
    st.markdown("---")
    page = st.radio("분석 유형 선택", ["📊 종합 분석", "📈 기술 분석"], captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"], key="page_selector")
    st.markdown("---")
    if page == "📊 종합 분석":
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input("종목 티커", "AAPL", key="main_ticker", help="해외(예: AAPL) 또는 국내(예: 005930 - FMP는 .KS 제외)", disabled=not comprehensive_analysis_possible)
        analysis_years = st.select_slider("분석 기간 (년)", [1, 2, 3, 5, 7, 10], 2, key="analysis_years", disabled=not comprehensive_analysis_possible)
        st.caption(f"과거 {analysis_years}년 데이터 분석")
        forecast_days = st.number_input("예측 기간 (일)", 7, 90, 30, 7, key="forecast_days", disabled=not comprehensive_analysis_possible)
        st.caption(f"향후 {forecast_days}일 예측")
        num_trend_periods_input = st.number_input("재무 추세 분기 수", 2, 12, 4, 1, key="num_trend_periods", disabled=not comprehensive_analysis_possible)
        st.caption(f"최근 {num_trend_periods_input}개 분기 재무 추세 계산")
        st.divider()
        st.subheader("⚙️ 예측 세부 설정 (선택)")
        changepoint_prior_input = st.slider("추세 변화 민감도 (Prophet)", 0.001, 0.5, 0.05, 0.01, "%.3f", help="클수록 과거 추세 변화에 민감 (기본값: 0.05)", key="changepoint_prior", disabled=not comprehensive_analysis_possible)
        st.caption(f"현재 민감도: {changepoint_prior_input:.3f}")
        st.divider()
        st.subheader("💰 보유 정보 입력 (선택)")
        avg_price = st.number_input("평단가", 0.0, format="%.2f", key="avg_price", disabled=not comprehensive_analysis_possible)
        quantity = st.number_input("보유 수량", 0, step=1, key="quantity", disabled=not comprehensive_analysis_possible)
        st.caption("평단가 입력 시 리스크 트래커 분석 활성화"); st.divider()
    elif page == "📈 기술 분석":
        st.header("⚙️ 기술 분석 설정")
        # Ticker input needed here as well for technical analysis
        ticker_tech_input = st.text_input("종목 티커", "AAPL", key="tech_ticker_input", help="해외(예: AAPL) 또는 국내(예: 005930 - FMP는 .KS 제외)")
        bb_window = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window")
        bb_std = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std", format="%.1f")
        st.caption(f"현재 설정: {bb_window}일 기간, {bb_std:.1f} 표준편차")
        st.divider()
        # Add date and interval selectors for Tech Analysis here
        today_tech = datetime.now().date()
        default_start_date_tech = today_tech - relativedelta(months=3)
        min_date_allowed_tech = today_tech - relativedelta(years=5)
        start_date_tech_input = st.date_input("시작일", default_start_date_tech, key="tech_start_input", min_value=min_date_allowed_tech, max_value=today_tech - timedelta(days=1))
        end_date_tech_input = st.date_input("종료일", today_tech, key="tech_end_input", min_value=start_date_tech_input + timedelta(days=1), max_value=today_tech)
        available_intervals_tech = {"일봉": "1day", "1시간": "1hour", "15분": "15min"}
        interval_help_fmp_tech = """데이터 간격 선택 (FMP 기준):\n- FMP 플랜 따라 지원 간격 다름.\n- 분봉 데이터 유료 플랜 필요 가능성.\n* 조회 기간 제한도 FMP 플랜 따라 다름."""
        interval_display_tech_input = st.selectbox("데이터 간격", list(available_intervals_tech.keys()), key="tech_interval_display_input", help=interval_help_fmp_tech)


# --- 캐시된 종합 분석 함수 (fred_key 제거) ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, years, days, num_trend_periods, changepoint_prior_scale): # fred_key 제거
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    try:
        # analyze_stock 호출 시 fred_key 인자 제거
        analysis_results = sa.analyze_stock(
            ticker,
            analysis_period_years=years,
            forecast_days=days,
            num_trend_periods=num_trend_periods,
            changepoint_prior_scale=changepoint_prior_scale
        )
        return analysis_results
    except Exception as e:
        logging.error(f"analyze_stock 함수 실행 중 오류 발생 (ticker: {ticker}): {e}\n{traceback.format_exc()}")
        return {"error": f"종합 분석 중 오류 발생: {e}"}

# --- 메인 화면 로직 ---
# ============== 📊 종합 분석 탭 ==============
if page == "📊 종합 분석":
    st.title("📊 종합 분석 결과 (FMP API 기반)")
    st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공.")
    st.markdown("---")
    analyze_button_main_disabled = not comprehensive_analysis_possible
    if analyze_button_main_disabled: st.error("FMP API 키 로드 실패. 종합 분석 불가.")
    analyze_button_main = st.button("🚀 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_button", disabled=analyze_button_main_disabled)
    results_placeholder = st.container()
    if analyze_button_main:
        ticker = st.session_state.get('main_ticker', "AAPL")
        years = st.session_state.get('analysis_years', 2)
        days = st.session_state.get('forecast_days', 30)
        periods = st.session_state.get('num_trend_periods', 4)
        cp_prior = st.session_state.get('changepoint_prior', 0.05)
        avg_p = st.session_state.get('avg_price', 0.0)
        qty = st.session_state.get('quantity', 0)

        if not ticker: results_placeholder.warning("종목 티커 입력 필요.")
        else:
            ticker_proc = ticker.strip().upper()
            if '.KS' in ticker_proc or '.KQ' in ticker_proc:
                original_ticker = ticker_proc
                ticker_proc = ticker_proc.split('.')[0]
                results_placeholder.info(f"국내 티커 감지: {original_ticker} -> {ticker_proc} (FMP용)")

            with st.spinner(f"{ticker_proc} 종합 분석 중..."):
                try:
                    # --- run_cached_analysis 호출 (FRED_API_KEY 인자 제거) ---
                    results = run_cached_analysis(ticker_proc, years, days, periods, cp_prior)
                    # --- 결과 표시 로직 ---
                    if results and isinstance(results, dict):
                        if "error" in results:
                            results_placeholder.error(f"분석 실패: {results['error']}")
                        else:
                            results_placeholder.empty() # Clear spinner message area
                            if results.get("warn_high_mape"):
                                m = results.get("mape", "N/A")
                                mape_value_str = m if isinstance(m, str) else (f"{m:.1f}%" if isinstance(m, (int, float)) else "N/A")
                                st.warning(f"🔴 모델 정확도 낮음 (MAPE {mape_value_str}). 예측 신뢰도 주의!")

                            with results_placeholder: # Use the container to display results
                                st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")
                                st.subheader("요약 정보")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                                col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                                col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                                st.subheader("📊 기업 기본 정보")
                                fundamentals = results.get('fundamentals')
                                if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A":
                                    colf1, colf2, colf3 = st.columns(3)
                                    with colf1: st.metric("시가총액", fundamentals.get("시가총액", "N/A")); st.metric("PER", fundamentals.get("PER", "N/A"))
                                    with colf2: st.metric("EPS", fundamentals.get("EPS", "N/A")); st.metric("Beta", fundamentals.get("베타", "N/A"))
                                    with colf3: st.metric("배당", fundamentals.get("배당수익률", "N/A")); st.metric("업종", fundamentals.get("업종", "N/A"))
                                    industry = fundamentals.get("산업", "N/A")
                                    summary = fundamentals.get("요약", "N/A")
                                    if industry != "N/A": st.markdown(f"**산업:** {industry}")
                                    # --- SyntaxError Fix: Indent the block below ---
                                    if summary != "N/A":
                                        with st.expander("회사 요약 보기"):
                                            st.write(summary)
                                    # ----------------------------------------------
                                    st.caption("Data Source: Financial Modeling Prep")
                                else:
                                    st.warning("기업 기본 정보 로드 실패.")

                                st.divider()
                                st.subheader(f"📈 주요 재무 추세 (최근 {periods} 분기)")
                                tab_titles = ["영업이익률(%)", "ROE(%)", "부채비율", "유동비율"]
                                tabs = st.tabs(tab_titles)
                                trend_data_map = {
                                    "영업이익률(%)": ('operating_margin_trend', 'Op Margin (%)', "{:.2f}%"),
                                    "ROE(%)": ('roe_trend', 'ROE (%)', "{:.2f}%"),
                                    "부채비율": ('debt_to_equity_trend', 'D/E Ratio', "{:.2f}"),
                                    "유동비율": ('current_ratio_trend', 'Current Ratio', "{:.2f}")
                                }
                                for i, title in enumerate(tab_titles):
                                    with tabs[i]:
                                        data_key, col_name, style_format = trend_data_map[title]
                                        trend_data = results.get(data_key)
                                        if trend_data and isinstance(trend_data, list) and len(trend_data) > 0:
                                            try:
                                                df_trend = pd.DataFrame(trend_data)
                                                df_trend['Date'] = pd.to_datetime(df_trend['Date'])
                                                df_trend.set_index('Date', inplace=True)
                                                if col_name in df_trend.columns:
                                                    st.line_chart(df_trend[[col_name]])
                                                    with st.expander("데이터 보기"):
                                                        st.dataframe(df_trend[[col_name]].style.format({col_name: style_format}), use_container_width=True)
                                                else: st.error(f"'{col_name}' 컬럼 없음.")
                                            except Exception as e: st.error(f"{title} 표시 오류: {e}")
                                        else: st.info(f"{title} 추세 데이터 없음.")

                                st.divider()
                                st.subheader("기술적 분석 차트 (종합)")
                                stock_chart_fig = results.get('stock_chart_fig')
                                if stock_chart_fig:
                                    st.plotly_chart(stock_chart_fig, use_container_width=True)
                                else:
                                    st.warning("주가 차트 생성 실패 (종합).")
                                st.divider()
                                st.subheader("시장 심리 분석")
                                col_news, col_fng = st.columns([2, 1])
                                with col_news:
                                    st.markdown("**📰 뉴스 감정 분석 (FMP + TextBlob)**")
                                    news_sentiment = results.get('news_sentiment', ["정보 없음."])
                                    if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                                        st.info(news_sentiment[0]) # Display summary line
                                        if len(news_sentiment) > 1:
                                            with st.expander("뉴스 목록 보기"):
                                                # Use st.write for each line for better formatting
                                                for line in news_sentiment[1:]:
                                                     st.write(f"- {line}")
                                    else: st.write(str(news_sentiment)) # Handle non-list cases
                                with col_fng:
                                    st.markdown("**😨 공포-탐욕 지수**")
                                    fng_index = results.get('fear_greed_index', "N/A")
                                    if isinstance(fng_index, dict):
                                        st.metric("현재 지수", fng_index.get('value', 'N/A'), fng_index.get('classification', ''))
                                    else: st.write(fng_index)

                                st.divider()
                                st.subheader("Prophet 주가 예측")
                                forecast_fig = results.get('forecast_fig')
                                forecast_data_list = results.get('prophet_forecast')
                                if forecast_fig: st.plotly_chart(forecast_fig, use_container_width=True)
                                elif isinstance(forecast_data_list, str): st.info(forecast_data_list); # Show message like "Prediction failed"
                                else: st.warning("예측 차트 생성 실패.")

                                if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                    st.markdown("**📊 예측 데이터 (최근 10일)**")
                                    try:
                                        df_fcst = pd.DataFrame(forecast_data_list)
                                        df_fcst['ds'] = pd.to_datetime(df_fcst['ds'])
                                        df_fcst_display = df_fcst.sort_values("ds").iloc[-10:].copy()
                                        df_fcst_display['ds'] = df_fcst_display['ds'].dt.strftime('%Y-%m-%d')
                                        format_dict_fcst = {col: "{:.2f}" for col in ['yhat', 'yhat_lower', 'yhat_upper'] if col in df_fcst_display.columns}
                                        # Ensure columns exist before trying to format
                                        display_cols_fcst = ['ds'] + [col for col in format_dict_fcst if col in df_fcst_display]
                                        st.dataframe(df_fcst_display[display_cols_fcst].style.format(format_dict_fcst), use_container_width=True)
                                    except Exception as e: st.error(f"예측 데이터 표시 오류: {e}")

                                cv_plot_path = results.get('cv_plot_path')
                                if cv_plot_path and os.path.exists(cv_plot_path):
                                    st.markdown("**📉 교차 검증 결과 (MAPE)**")
                                    try: st.image(cv_plot_path, caption="MAPE (낮을수록 정확)")
                                    except Exception as img_e: st.warning(f"CV 이미지 로드 실패: {img_e}")
                                elif cv_plot_path is None and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                    # Only show if forecast data exists but CV plot doesn't
                                    st.caption("교차 검증(CV) 결과 없음.")

                                st.divider()
                                st.subheader("🚨 리스크 트래커 (예측 기반)")
                                risk_days, max_loss_pct, max_loss_amt = 0, 0.0, 0.0 # Initialize as floats
                                valid_pred = False # Initialize valid_pred

                                if avg_p > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                    try:
                                        df_pred = pd.DataFrame(forecast_data_list)
                                        required_fcst_cols = ['ds', 'yhat_lower']
                                        if all(col in df_pred.columns for col in required_fcst_cols):
                                            df_pred['ds'] = pd.to_datetime(df_pred['ds'], errors='coerce')
                                            df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], errors='coerce')
                                            df_pred.dropna(subset=['ds', 'yhat_lower'], inplace=True)
                                            if not df_pred.empty: valid_pred = True # Set valid_pred to True if data remains

                                        if valid_pred: # Check valid_pred before proceeding
                                            df_pred['평단가'] = avg_p
                                            df_pred['리스크 여부'] = df_pred['yhat_lower'] < avg_p
                                            # Ensure avg_p is not zero before division
                                            if avg_p != 0:
                                                 df_pred['예상 손실률'] = np.where(df_pred['리스크 여부'], ((df_pred['yhat_lower'] - avg_p) / avg_p) * 100, 0.0).fillna(0.0)
                                            else:
                                                 df_pred['예상 손실률'] = 0.0 # Avoid division by zero

                                            if qty > 0:
                                                df_pred['예상 손실액'] = np.where(df_pred['리스크 여부'], (df_pred['yhat_lower'] - avg_p) * qty, 0.0).fillna(0.0)
                                            else: df_pred['예상 손실액'] = 0.0

                                            risk_days = df_pred['리스크 여부'].sum()
                                            if risk_days > 0:
                                                valid_loss_pct = df_pred.loc[df_pred['리스크 여부'], '예상 손실률'].dropna()
                                                max_loss_pct = valid_loss_pct.min() if not valid_loss_pct.empty else 0.0
                                                if qty > 0:
                                                     valid_loss_amt = df_pred.loc[df_pred['리스크 여부'], '예상 손실액'].dropna()
                                                     max_loss_amt = valid_loss_amt.min() if not valid_loss_amt.empty else 0.0
                                                else: max_loss_amt = 0.0
                                            else: max_loss_pct, max_loss_amt = 0.0, 0.0

                                            st.markdown("##### 리스크 요약")
                                            col_r1, col_r2, col_r3 = st.columns(3)
                                            col_r1.metric("⚠️ < 평단가 일수", f"{risk_days}일 / {days}일")
                                            col_r2.metric("📉 Max 손실률", f"{max_loss_pct:.2f}%")
                                            if qty > 0: col_r3.metric("💸 Max 손실액", f"${max_loss_amt:,.2f}")
                                            else: col_r3.metric("💸 Max 손실액", "N/A (수량 0)")


                                            if risk_days > 0: st.warning(f"{days}일 예측 중 **{risk_days}일** 평단가(${avg_p:.2f}) 하회 가능성.")
                                            else: st.success(f"{days}일간 평단가(${avg_p:.2f}) 하회 가능성 낮음.")

                                            st.markdown("##### 평단가 vs 예측 구간 비교")
                                            fig_risk = go.Figure()
                                            plot_cols_risk = {'yhat_lower': 'rgba(0,100,80,0.2)', 'yhat_upper': 'rgba(0,100,80,0.2)', 'yhat': 'rgba(0,100,80,0.6)'}
                                            df_plot_risk = df_pred[['ds'] + list(plot_cols_risk.keys())].copy()
                                            # Convert plot columns to numeric safely
                                            for col in plot_cols_risk:
                                                 if col in df_plot_risk.columns:
                                                      df_plot_risk[col] = pd.to_numeric(df_plot_risk[col], errors='coerce')
                                            df_plot_risk.dropna(subset=['ds'] + list(plot_cols_risk.keys()), how='any', inplace=True)

                                            if not df_plot_risk.empty:
                                                if 'yhat_upper' in df_plot_risk.columns: fig_risk.add_trace(go.Scatter(x=df_plot_risk['ds'], y=df_plot_risk['yhat_upper'], mode='lines', line_color=plot_cols_risk['yhat_upper'], name='Upper'))
                                                if 'yhat_lower' in df_plot_risk.columns: fig_risk.add_trace(go.Scatter(x=df_plot_risk['ds'], y=df_plot_risk['yhat_lower'], mode='lines', line_color=plot_cols_risk['yhat_lower'], name='Lower', fill='tonexty' if 'yhat_upper' in df_plot_risk.columns else None, fillcolor='rgba(0,100,80,0.1)'))
                                                if 'yhat' in df_plot_risk.columns: fig_risk.add_trace(go.Scatter(x=df_plot_risk['ds'], y=df_plot_risk['yhat'], mode='lines', line=dict(dash='dash', color=plot_cols_risk['yhat']), name='Forecast'))
                                                fig_risk.add_hline(y=avg_p, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_p:.2f}", annotation_position="bottom right")
                                                df_risk_periods = df_pred[df_pred['리스크 여부']]
                                                if not df_risk_periods.empty: fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day'))
                                                fig_risk.update_layout(hovermode="x unified")
                                                st.plotly_chart(fig_risk, use_container_width=True)

                                                if risk_days > 0:
                                                    with st.expander(f"리스크 예측일 상세 데이터 ({risk_days}일)"):
                                                        df_risk_days_display = df_pred[df_pred['리스크 여부']].copy()
                                                        df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d')
                                                        cols_show = ['ds', 'yhat_lower', '평단가', '예상 손실률']
                                                        formatters = {"yhat_lower":"{:.2f}", "평단가":"{:.2f}", "예상 손실률":"{:.2f}%"}
                                                        if qty > 0 and '예상 손실액' in df_risk_days_display.columns:
                                                            cols_show.append('예상 손실액')
                                                            formatters["예상 손실액"] = "${:,.2f}"
                                                        # Ensure columns exist before displaying
                                                        cols_show_final = [col for col in cols_show if col in df_risk_days_display.columns]
                                                        st.dataframe(df_risk_days_display[cols_show_final].style.format(formatters), use_container_width=True)
                                            else: st.info("차트 표시 필요 데이터 부족.")
                                        else: st.info("리스크 분석 위한 유효 데이터 없음.") # Show if valid_pred is False
                                    except Exception as risk_calc_err:
                                         st.error(f"리스크 트래커 계산/표시 오류: {risk_calc_err}")
                                         logging.error(f"Risk tracker error: {traceback.format_exc()}")
                                elif avg_p <= 0: st.info("⬅️ 사이드바 '평단가' 입력 시 리스크 분석 확인 가능.")
                                else: st.warning("예측 데이터 부족/오류로 리스크 분석 불가.")

                                st.divider()
                                st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                                summary_points = []
                                if isinstance(forecast_data_list, list) and forecast_data_list:
                                    try:
                                        df_pred_summary = pd.DataFrame(forecast_data_list)
                                        # Check if necessary columns exist and have data
                                        if all(col in df_pred_summary.columns for col in ['yhat', 'yhat_lower', 'yhat_upper']) and not df_pred_summary['yhat'].isnull().all():
                                            start_pred = df_pred_summary["yhat"].iloc[0]; end_pred = df_pred_summary["yhat"].iloc[-1]
                                            if pd.notna(start_pred) and pd.notna(end_pred):
                                                trend_obs = ("상승" if end_pred > start_pred * 1.02 else "하락" if end_pred < start_pred * 0.98 else "횡보")
                                            else: trend_obs = "판단 불가"

                                            lower = df_pred_summary["yhat_lower"].min() if df_pred_summary['yhat_lower'].notna().any() else 'N/A'
                                            upper = df_pred_summary["yhat_upper"].max() if df_pred_summary['yhat_upper'].notna().any() else 'N/A'
                                            lower_str = f"${lower:.2f}" if isinstance(lower, (int, float)) else lower
                                            upper_str = f"${upper:.2f}" if isinstance(upper, (int, float)) else upper
                                            summary_points.append(f"- **예측:** 향후 {days}일간 **{trend_obs}** 추세 ({lower_str} ~ {upper_str})")
                                        else: summary_points.append("- 예측: 예측 결과 컬럼 부족 또는 데이터 없음")
                                    except Exception as e: summary_points.append(f"- 예측: 요약 생성 오류: {e}")
                                else: summary_points.append("- 예측: 예측 데이터 부족/오류로 요약 불가")

                                # News Sentiment Summary
                                if isinstance(news_sentiment, list) and len(news_sentiment) > 0 and ":" in news_sentiment[0]:
                                    try:
                                        score_part = news_sentiment[0].split(":")[-1].strip()
                                        avg_score = float(score_part)
                                        sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"
                                        summary_points.append(f"- **뉴스:** 평균 감성 {avg_score:.2f}, **{sentiment_desc}** 분위기.")
                                    except Exception as e: summary_points.append(f"- 뉴스: 요약 오류 ({e})") # Include error details
                                elif isinstance(news_sentiment, list): summary_points.append(f"- 뉴스: {news_sentiment[0]}") # Show summary line if parsing failed
                                else: summary_points.append("- 뉴스: 감성 분석 정보 없음/오류.")

                                # Fear & Greed Summary
                                if isinstance(fng_index, dict):
                                     summary_points.append(f"- **시장 심리:** 공포-탐욕 {fng_index.get('value', 'N/A')} (**{fng_index.get('classification', 'N/A')}**).")
                                else: summary_points.append("- 시장 심리: 공포-탐욕 지수 정보 없음/오류.")

                                # Fundamentals Summary
                                if fundamentals and isinstance(fundamentals, dict):
                                    per = fundamentals.get("PER", "N/A"); sector = fundamentals.get("업종", "N/A")
                                    parts = [f"PER {per}"] if per != "N/A" else []
                                    parts.extend([f"업종 '{sector}'"] if sector != "N/A" else [])
                                    if parts: summary_points.append(f"- **기본 정보:** {', '.join(parts)}.")
                                    else: summary_points.append("- 기본 정보: 주요 지표(PER, 업종) 없음.")
                                else: summary_points.append("- 기본 정보: 로드 실패/정보 없음.")

                                # Financial Trend Summary
                                trend_parts = []
                                trend_labels = {'operating_margin_trend': '영업익률','roe_trend': 'ROE','debt_to_equity_trend': '부채비율','current_ratio_trend': '유동비율'}
                                trend_suffix = {'operating_margin_trend': '%','roe_trend': '%','debt_to_equity_trend': '','current_ratio_trend': ''}
                                trend_value_keys = {'operating_margin_trend': 'Op Margin (%)','roe_trend': 'ROE (%)','debt_to_equity_trend': 'D/E Ratio','current_ratio_trend': 'Current Ratio'}
                                try:
                                    for key in trend_labels:
                                        trend_list = results.get(key)
                                        if trend_list and isinstance(trend_list, list):
                                            last_item = trend_list[-1] # Get the most recent period
                                            value_key = trend_value_keys[key]
                                            value = last_item.get(value_key)
                                            if isinstance(value, (int, float)):
                                                 trend_parts.append(f"{trend_labels[key]} {value:.2f}{trend_suffix[key]}")
                                            elif value is not None: # Handle cases where value might be non-numeric string
                                                 trend_parts.append(f"{trend_labels[key]}: {value}")
                                            # else: # Optionally add message for missing value
                                            #      trend_parts.append(f"{trend_labels[key]} 정보 부족")
                                    if trend_parts: summary_points.append(f"- **최근 재무:** {', '.join(trend_parts)}.")
                                    # else: summary_points.append("- 최근 재무: 정보 없음.") # Optionally add if no trends found
                                except Exception as e: summary_points.append(f"- 최근 재무: 요약 오류: {e}")

                                # Risk Tracker Summary
                                if avg_p > 0 and valid_pred: # Use valid_pred flag
                                    if risk_days > 0:
                                        summary_points.append(f"- **리스크:** {days}일 중 **{risk_days}일** 평단가(${avg_p:.2f}) 하회 가능성 (Max 손실률: **{max_loss_pct:.2f}%**).")
                                    else: summary_points.append(f"- **리스크:** 예측 기간 내 평단가(${avg_p:.2f}) 하회 가능성 낮음.")
                                elif avg_p > 0: # If avg_price given but prediction failed
                                     summary_points.append(f"- 리스크: 평단가(${avg_p:.2f}) 입력됨, 예측 데이터 부족/오류로 분석 불가.")
                                # else: # No avg_price given
                                #      summary_points.append("- 리스크: 평단가 미입력으로 분석 생략.")


                                if summary_points:
                                     st.markdown("\n".join(summary_points))
                                     st.caption("⚠️ **주의:** 자동 생성된 요약이며 투자 결정의 근거가 될 수 없습니다.")
                                else: st.write("분석 요약 생성 불가.")

                    elif results is None: results_placeholder.error("분석 결과 처리 중 오류 발생 (결과 없음).")
                    else: results_placeholder.error("분석 결과 처리 중 오류 발생 (결과 형식 오류).")
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logging.error(f"종합 분석 메인 로직 실행 오류: {e}\n{error_traceback}")
                    results_placeholder.error(f"앱 실행 중 오류 발생: {e}")
    else: # 종합 분석 버튼 클릭 전
        if comprehensive_analysis_possible:
             results_placeholder.info("⬅️ 사이드바에서 설정을 확인하고 '종합 분석 시작!' 버튼을 클릭하세요.")
        else:
             results_placeholder.warning("API 키 로드 실패로 종합 분석을 진행할 수 없습니다.")

# ============== 📈 기술 분석 탭 ==============
elif page == "📈 기술 분석":
    st.title("📈 기술적 분석 (VWAP + Bollinger + Fibonacci) - FMP API")
    st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준을 함께 시각화하고 자동 해석을 제공합니다.")
    st.markdown("---")
    # Get settings from sidebar using updated keys
    ticker_tech = st.session_state.get('tech_ticker_input', "AAPL")
    start_date = st.session_state.get('tech_start_input', datetime.now().date() - relativedelta(months=3))
    end_date = st.session_state.get('tech_end_input', datetime.now().date())
    interval_display = st.session_state.get('tech_interval_display_input', "일봉")
    interval = available_intervals_tech[interval_display] # Get internal value
    bb_window_val = st.session_state.get('bb_window', 20)
    bb_std_val = st.session_state.get('bb_std', 2.0)

    analyze_button_tech = st.button("📊 기술적 분석 실행", key="tech_analyze", use_container_width=True, type="primary")

    if analyze_button_tech:
        if not ticker_tech: st.warning("종목 티커를 입력해주세요.")
        else:
            ticker_processed_tech = ticker_tech.strip().upper()
            if '.KS' in ticker_processed_tech or '.KQ' in ticker_processed_tech:
                original_ticker_tech = ticker_processed_tech
                ticker_processed_tech = ticker_processed_tech.split('.')[0]
                st.info(f"국내 티커 감지: {original_ticker_tech} -> {ticker_processed_tech} (FMP용)")

            df_tech = pd.DataFrame() # Initialize empty dataframe
            st.session_state['error_shown_tech'] = False # Reset error flag

            st.write(f"**{ticker_processed_tech}** ({interval_display}, BB:{bb_window_val}일/{bb_std_val:.1f}σ) 분석 중 (FMP API 사용)...")
            with st.spinner(f"{ticker_processed_tech} 데이터 로딩 및 처리 중 (FMP)..."):
                try:
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    fmp_data = None; rename_map = {}

                    if interval == "1day":
                        # --- Correct function call for daily data ---
                        fmp_data = fmp_api.get_price_data(ticker=ticker_processed_tech, start_date=start_date_str, end_date=end_date_str)
                        rename_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                    else: # Intraday intervals
                        # Use from_date and to_date if that's what get_intraday_data expects
                        fmp_data = fmp_api.get_intraday_data(ticker=ticker_processed_tech, interval=interval, from_date=start_date_str, to_date=end_date_str)
                        # Assume same rename map, adjust if FMP intraday response is different
                        rename_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}

                    if fmp_data and isinstance(fmp_data, list):
                        df_tech = pd.DataFrame(fmp_data)
                        if not df_tech.empty:
                            df_tech = df_tech.rename(columns=rename_map)
                            date_col_name = rename_map.get('date', 'Date') # Use the mapped name

                            if date_col_name in df_tech.columns:
                                df_tech[date_col_name] = pd.to_datetime(df_tech[date_col_name], errors='coerce')
                                df_tech = df_tech.set_index(date_col_name).sort_index()
                                # Convert OHLCV to numeric, coercing errors
                                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                    if col in df_tech.columns:
                                         df_tech[col] = pd.to_numeric(df_tech[col], errors='coerce')
                                # Drop rows where essential price data is missing after conversion
                                df_tech.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                            else:
                                st.error(f"FMP 응답 날짜 컬럼 '{date_col_name}' 없음.")
                                df_tech = pd.DataFrame() # Reset to empty
                        # else: # No need for message if dataframe becomes empty after processing, handled below
                        #    st.warning(f"FMP 데이터 처리 후 결과 없음 ('{ticker_processed_tech}', {interval_display}).")
                    elif fmp_data is None:
                         st.error(f"FMP 데이터 로딩 오류 (API 결과 None).")
                         df_tech = pd.DataFrame()
                    else: # Empty list returned by API
                         st.warning(f"FMP 데이터 '{ticker_processed_tech}' ({interval_display}) 가져오기 실패 (API 결과 빈 리스트).")
                         df_tech = pd.DataFrame()

                    # --- Process data if df_tech is not empty ---
                    if df_tech.empty:
                        if not st.session_state.get('error_shown_tech', False):
                            st.error(f"❌ 데이터 조회/처리 실패.")
                            st.session_state['error_shown_tech'] = True
                    else:
                        st.session_state['error_shown_tech'] = False # Reset error flag on success
                        logging.info(f"FMP 데이터 처리 완료 ({ticker_processed_tech}, {interval_display}). {len(df_tech)} 행.")
                        st.caption(f"조회 기간 (FMP): {df_tech.index.min()} ~ {df_tech.index.max()}")

                        # Check for essential columns needed for calculations
                        required_cols_tech = ['Open', 'High', 'Low', 'Close'] # Volume is needed for VWAP but checked in function
                        missing_cols_tech = [col for col in required_cols_tech if col not in df_tech.columns]
                        if missing_cols_tech:
                            st.error(f"❌ 기술적 분석 위한 필수 컬럼 누락: {missing_cols_tech}.")
                            st.dataframe(df_tech.head()) # Show head to help debug
                        else:
                            df_calculated = df_tech.copy()
                            df_calculated.attrs['ticker'] = ticker_processed_tech # Store ticker for potential use in functions

                            # Calculate indicators safely
                            try: df_calculated = calculate_vwap(df_calculated)
                            except Exception as e: st.warning(f"VWAP 계산 오류: {e}", icon="⚠️")
                            try: df_calculated = calculate_bollinger_bands(df_calculated, window=bb_window_val, num_std=bb_std_val)
                            except Exception as e: st.warning(f"BB 계산 오류: {e}", icon="⚠️")
                            try: df_calculated = calculate_rsi(df_calculated) # Assumes this exists in short_term_analysis
                            except NameError: st.error("오류: 'calculate_rsi' 함수를 찾을 수 없습니다. 'short_term_analysis.py'를 확인하세요.")
                            except Exception as e: st.warning(f"RSI 계산 오류: {e}", icon="⚠️")
                            try: df_calculated = calculate_macd(df_calculated) # Assumes this exists in short_term_analysis
                            except NameError: st.error("오류: 'calculate_macd' 함수를 찾을 수 없습니다. 'short_term_analysis.py'를 확인하세요.")
                            except Exception as e: st.warning(f"MACD 계산 오류: {e}", icon="⚠️")

                            # Plotting
                            st.subheader(f"📌 {ticker_processed_tech} 기술적 분석 통합 차트 ({interval_display})")
                            chart_tech = plot_technical_chart(df_calculated, ticker_processed_tech)
                            if chart_tech and chart_tech.data: st.plotly_chart(chart_tech, use_container_width=True)
                            else: st.warning("차트 생성 실패/표시 데이터 없음.")

                            # Display recent data table
                            st.subheader("📄 최근 데이터 (계산된 지표 포함)")
                            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
                            display_cols_exist = [col for col in display_cols if col in df_calculated.columns]
                            format_dict = {col: "{:.2f}" for col in display_cols_exist if col != 'Volume'}
                            if 'Volume' in display_cols_exist: format_dict['Volume'] = "{:,.0f}"
                            st.dataframe(df_calculated[display_cols_exist].tail(10).style.format(format_dict), use_container_width=True)

                            # Signal Interpretation
                            st.divider()
                            st.subheader("🧠 기술적 시그널 해석 (참고용)")
                            if not df_calculated.empty:
                                latest_row = df_calculated.iloc[-1].copy()
                                signal_messages = []
                                # prev_close_fib = None # Fibonacci interpretation commented out
                                # if len(df_calculated) >= 2: prev_close_fib = df_calculated['Close'].iloc[-2]

                                try:
                                     # Ensure the interpretation function exists
                                    if 'interpret_technical_signals' in globals():
                                        signal_messages.extend(interpret_technical_signals(latest_row, df_context=df_calculated))
                                    else:
                                        st.error("오류: 'interpret_technical_signals' 함수를 찾을 수 없습니다. 'technical_interpret.py'를 확인하세요.")
                                except Exception as e: st.warning(f"기본 기술적 시그널 해석 오류: {e}", icon="⚠️")

                                # Fibonacci interpretation (keep commented unless needed and function exists)
                                # try:
                                #    if 'interpret_fibonacci' in globals():
                                #        fib_msg = interpret_fibonacci(df_calculated, close_value=latest_row.get("Close"), prev_close=prev_close_fib)
                                #        if fib_msg: signal_messages.append(fib_msg)
                                #    else:
                                #        st.warning("피보나치 해석 함수 'interpret_fibonacci' 없음.", icon="⚠️")
                                # except Exception as e_fib: st.warning(f"피보나치 시그널 해석 오류: {e_fib}", icon="⚠️")

                                if signal_messages:
                                     for msg in signal_messages: st.info(msg) # Use st.info for better visibility
                                else: st.info("현재 특별히 감지된 기술적 시그널은 없습니다.")
                                st.caption("⚠️ **주의:** 자동 해석은 보조 지표이며 투자 결정은 반드시 종합적인 판단 하에 신중하게 내리시기 바랍니다.")
                            else: st.warning("해석할 데이터가 부족합니다.")

                except requests.exceptions.RequestException as req_err:
                    st.error(f"FMP API 요청 실패: {req_err}")
                    logging.error(f"FMP API request error (Tech Tab): {req_err}")
                except EnvironmentError as env_err: # Catch API key errors specifically if fmp_api raises them
                     st.error(f"FMP API 키 설정 오류: {env_err}")
                     logging.error(f"FMP API key error (Tech Tab): {env_err}")
                except Exception as e:
                    st.error(f"기술적 분석 처리 중 오류 발생: {type(e).__name__} - {e}")
                    logging.error(f"Tech analysis tab error: {traceback.format_exc()}")
    else: # 기술 분석 버튼 클릭 전
        st.info("종목 티커, 기간, 데이터 간격 등을 설정한 후 '기술적 분석 실행' 버튼을 클릭하세요.")


# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 (FMP API) | 정보 제공 목적 (투자 조언 아님)")
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")