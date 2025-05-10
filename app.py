# -*- coding: utf-8 -*-
# Combined app.py V2.0 (FMP API Integration Applied including Technical Analysis Tab)

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
# import yfinance as yf # 기술 분석 탭 FMP 전환으로 제거

# --- FMP API 및 분석 모듈 임포트 ---
try:
    import fmp_api # FMP API 래퍼 모듈
    import stock_analysis as sa # 종합 분석 로직
except ImportError as e:
    st.error(f"필수 API 또는 분석 모듈 로딩 실패: {e}. 'fmp_api.py'와 'stock_analysis.py' 파일이 있는지 확인하세요.")
    st.stop()

# --- 기술적 분석 관련 함수 임포트 ---
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
# calculate_vwap, calculate_bollinger_bands 함수 정의 (이전과 동일)
def calculate_vwap(df):
    """VWAP 계산 (Volume 필요)"""
    df = df.copy()
    required_cols = ['High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: raise ValueError(f"VWAP 계산 실패: 컬럼 부족 ({missing_cols})")
    if 'Volume' not in df.columns or df['Volume'].isnull().all() or (df['Volume'] == 0).all():
        df['VWAP'] = np.nan
        logging.warning(f"Ticker {df.attrs.get('ticker', '')}: VWAP 계산 불가 (거래량 데이터 부족 또는 0)")
        return df
    df['Volume'] = df['Volume'].fillna(0)
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
    df['VWAP'] = np.where(df['cumulative_volume'] > 0, df['cumulative_tp_volume'] / df['cumulative_volume'], np.nan)
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """볼린저 밴드 계산 (Close 필요)"""
    df = df.copy()
    required_col = 'Close'
    if required_col not in df.columns or df[required_col].isnull().all():
        st.warning(f"BB 계산 실패: '{required_col}' 컬럼 없음 또는 데이터 없음.")
        df['MA20'] = np.nan
        df['Upper'] = np.nan
        df['Lower'] = np.nan
        return df
    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        st.warning(f"BB 계산 위한 유효 데이터({len(valid_close)}개)가 기간({window}개)보다 부족.")
        df['MA20'] = np.nan
        df['Upper'] = np.nan
        df['Lower'] = np.nan
    else:
        df['MA20'] = df[required_col].rolling(window=window, min_periods=window).mean()
        df['STD20'] = df[required_col].rolling(window=window, min_periods=window).std()
        df['Upper'] = df['MA20'] + num_std * df['STD20']
        df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

# --- 차트 생성 함수 (기존 코드 유지) ---
def plot_technical_chart(df, ticker):
    """기술적 분석 지표 통합 차트 생성 (VWAP, Bollinger Band, Fibonacci, RSI, MACD 포함)"""
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols) or df[required_candle_cols].isnull().all(axis=None):
        st.error(f"캔들차트 필요 컬럼({required_candle_cols}) 없음/데이터 없음.")
        return fig # 빈 Figure 반환

    # (1) 캔들 차트
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))

    # (2) VWAP
    if 'VWAP' in df.columns and df['VWAP'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines',
                                 name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns:
        st.caption("VWAP 데이터 없음/표시 불가.")

    # (3) Bollinger Bands
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20',
                                     line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper',
                                 line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower',
                                 line=dict(color='grey', width=1), fill='tonexty',
                                 fillcolor='rgba(180,180,180,0.1)'))
    elif 'Upper' in df.columns:
        st.caption("볼린저 밴드 데이터 없음/표시 불가.")

    # (4) Fibonacci Levels
    valid_price_df = df.dropna(subset=['High', 'Low'])
    if not valid_price_df.empty:
        min_price = valid_price_df['Low'].min()
        max_price = valid_price_df['High'].max()
        diff = max_price - min_price
        if diff > 0:
            levels = {'0.0 (High)': max_price, '0.236': max_price - 0.236 * diff,
                      '0.382': max_price - 0.382 * diff, '0.5': max_price - 0.5 * diff,
                      '0.618': max_price - 0.618 * diff, '1.0 (Low)': min_price}
            fib_colors = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold',
                          '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels.items():
                fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: ${v:.2f}",
                              line_color=fib_colors.get(k, 'navy'), annotation_position="bottom right",
                              annotation_font_size=10)
        else:
            st.caption("기간 내 가격 변동 없어 피보나치 미표시.")
    else:
        st.caption("피보나치 레벨 계산 불가.")

    # (5) RSI
    if 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)',
                                 line=dict(color='purple', width=1), yaxis='y2'))

    # (6) MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines',
                                 line=dict(color='teal'), yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', mode='lines',
                                 line=dict(color='orange'), yaxis='y3'))
        if 'MACD_hist' in df.columns:
            colors = ['lightgreen' if v >= 0 else 'lightcoral' for v in df['MACD_hist']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram',
                                 marker_color=colors, yaxis='y3'))

    # 레이아웃
    fig.update_layout(
        title=f"{ticker} - 기술적 분석 통합 차트",
        xaxis_title="날짜 / 시간",
        yaxis=dict(domain=[0.4, 1], title="가격 ($)"),
        yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right', showgrid=False),
        yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        legend_title_text="지표",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="종합 주식 분석 FMP 버전", layout="wide", initial_sidebar_state="expanded")

# --- API 키 로드 및 확인 (이전 답변의 수정된 로직 유지) ---
FRED_API_KEY = None
fmp_key_loaded = False
fred_key_loaded = False # FRED 키 필요 여부에 따라 관리

sidebar_status = st.sidebar.empty() # 임시 메시지용 placeholder
final_status_message_displayed = False # 최종 상태 메시지 표시 여부 플래그

# secrets 로드 시도
secrets_available = hasattr(st, 'secrets')
if secrets_available:
    try:
        fmp_secret_key = st.secrets.get("FMP_API_KEY")
        if fmp_secret_key:
            fmp_api.FMP_API_KEY = fmp_secret_key
            fmp_key_loaded = True

        FRED_API_KEY = st.secrets.get("FRED_API_KEY") # FRED 키 로드 (stock_analysis.py 가 아직 사용)
        if FRED_API_KEY: fred_key_loaded = True

        if not fmp_key_loaded:
            sidebar_status.warning("Secrets에 FMP API 키가 없습니다.")

    except Exception as e:
        sidebar_status.error(f"Secrets 로드 오류: {e}")
        final_status_message_displayed = True

# .env 로드 시도 (FMP 키가 secrets에서 로드 안 됐거나 로딩 실패 시)
if not fmp_key_loaded:
    sidebar_status.info(".env 파일 확인 중...")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            fmp_env_key = os.getenv("FMP_API_KEY")
            if fmp_env_key:
                fmp_api.FMP_API_KEY = fmp_env_key
                fmp_key_loaded = True

            fred_env_key = os.getenv("FRED_API_KEY")
            if fred_env_key and not fred_key_loaded:
                FRED_API_KEY = fred_env_key
                fred_key_loaded = True

            if fmp_key_loaded:
                sidebar_status.success("API 키 로드 완료 (.env 또는 secrets)")
                final_status_message_displayed = True
            else:
                sidebar_status.error(".env 파일 내 FMP API 키 누락 또는 로드 실패.")
                final_status_message_displayed = True
        else:
             if not secrets_available or not fmp_key_loaded:
                 sidebar_status.error(".env 파일 없음, Secrets에도 FMP 키 없음.")
                 final_status_message_displayed = True
             elif fmp_key_loaded:
                  sidebar_status.empty()

    except Exception as e:
        sidebar_status.error(f".env 로드 오류: {e}")
        final_status_message_displayed = True

# --- 종합 분석 가능 여부 결정 ---
comprehensive_analysis_possible = fmp_key_loaded

# --- 최종 상태 메시지 정리 ---
if not comprehensive_analysis_possible and not final_status_message_displayed:
    st.sidebar.error("FMP API 키 로드 실패! '종합 분석' 기능이 제한됩니다.")
    final_status_message_displayed = True
elif comprehensive_analysis_possible and not final_status_message_displayed:
    sidebar_status.success("API 키 로드 완료.")
    final_status_message_displayed = True
elif comprehensive_analysis_possible and final_status_message_displayed:
    pass # 이미 메시지 표시됨

# FRED 키 관련 경고 (선택 사항)
# if comprehensive_analysis_possible and not fred_key_loaded:
#    st.sidebar.warning("FRED API 키는 로드되지 않았습니다 (필요 시 확인).")


# --- 사이드바 설정 ---
with st.sidebar:
    # ... (후원계좌, 링크 등 유지) ...
    st.title("📊 주식 분석 도구 (FMP API)")
    st.markdown("---")

    page = st.radio("분석 유형 선택", ["📊 종합 분석", "📈 기술 분석"],
                    captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"],
                    key="page_selector")
    st.markdown("---")

    # 각 페이지별 설정 UI (기존과 동일)
    if page == "📊 종합 분석":
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input("종목 티커", "AAPL", key="main_ticker",
                                     help="해외(예: AAPL) 또는 국내(예: 005930 - FMP는 .KS 제외)",
                                     disabled=not comprehensive_analysis_possible)
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
        st.caption("평단가 입력 시 리스크 트래커 분석 활성화")
        st.divider()

    elif page == "📈 기술 분석":
        st.header("⚙️ 기술 분석 설정")
        bb_window = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window")
        bb_std = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std", format="%.1f")
        st.caption(f"현재 설정: {bb_window}일 기간, {bb_std:.1f} 표준편차")
        st.divider()

# --- 캐시된 종합 분석 함수 (기존과 동일, news_key 인자 제거됨) ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker,  years, days, num_trend_periods, changepoint_prior_scale):
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    try:
        analysis_results = sa.analyze_stock(
            ticker,
            #fred_key=fred_key, # FRED 키 전달 (stock_analysis.py 가 아직 사용)
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

        if not ticker:
            results_placeholder.warning("종목 티커 입력 필요.")
        else:
            ticker_proc = ticker.strip().upper()
            if '.KS' in ticker_proc or '.KQ' in ticker_proc:
                 original_ticker = ticker_proc
                 ticker_proc = ticker_proc.split('.')[0]
                 results_placeholder.info(f"국내 티커 감지: {original_ticker} -> {ticker_proc} (FMP용)")

            with st.spinner(f"{ticker_proc} 종합 분석 중..."):
                try:
                    # --- run_cached_analysis 호출 (news_key 없이, fred_key는 전달) ---
                    results = run_cached_analysis(
                        ticker_proc,
                        #FRED_API_KEY, # FRED 키 전달
                        years, days, periods, cp_prior
                    )

                    # --- 결과 표시 로직 (stock_analysis.py에서 반환된 결과 사용) ---
                    if results and isinstance(results, dict):
                        if "error" in results:
                            results_placeholder.error(f"분석 실패: {results['error']}")
                        else:
                            results_placeholder.empty() # 이전 메시지 지우기
                            if results.get("warn_high_mape"):
                                m = results.get("mape", "N/A")
                                mape_value_str = m if isinstance(m, str) else (f"{m:.1f}%" if isinstance(m, (int, float)) else "N/A")
                                st.warning(f"🔴 모델 정확도 낮음 (MAPE {mape_value_str}). 예측 신뢰도 주의!")

                            with results_placeholder:
                                st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")
                                # 1. 요약 정보
                                st.subheader("요약 정보")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                                col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                                col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                                # 2. 기본적 분석
                                st.subheader("📊 기업 기본 정보")
                                fundamentals = results.get('fundamentals')
                                if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A":
                                    colf1, colf2, colf3 = st.columns(3)
                                    with colf1: st.metric("시가총액", fundamentals.get("시가총액", "N/A")); st.metric("PER", fundamentals.get("PER", "N/A"))
                                    with colf2: st.metric("EPS", fundamentals.get("EPS", "N/A")); st.metric("Beta", fundamentals.get("베타", "N/A"))
                                    with colf3: st.metric("배당", fundamentals.get("배당수익률", "N/A")); st.metric("업종", fundamentals.get("업종", "N/A")) # 배당금액 표시
                                    industry = fundamentals.get("산업", "N/A"); summary = fundamentals.get("요약", "N/A")
                                    if industry != "N/A": st.markdown(f"**산업:** {industry}")
                                    if summary != "N/A":
                                        with st.expander("회사 요약 보기"): st.write(summary)
                                    st.caption("Data Source: Financial Modeling Prep")
                                else: st.warning("기업 기본 정보 로드 실패.")
                                st.divider()

                                # 3. 주요 재무 추세
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
                                                    with st.expander("데이터 보기"): st.dataframe(df_trend[[col_name]].style.format({col_name: style_format}), use_container_width=True)
                                                else: st.error(f"'{col_name}' 컬럼 없음.")
                                            except Exception as e: st.error(f"{title} 표시 오류: {e}")
                                        else: st.info(f"{title} 추세 데이터 없음.")
                                st.divider()

                                # 4. 기술적 분석 차트 (종합)
                                st.subheader("기술적 분석 차트 (종합)")
                                stock_chart_fig = results.get('stock_chart_fig')
                                if stock_chart_fig: st.plotly_chart(stock_chart_fig, use_container_width=True)
                                else: st.warning("주가 차트 생성 실패 (종합).")
                                st.divider()

                                # 5. 시장 심리 분석
                                st.subheader("시장 심리 분석")
                                col_news, col_fng = st.columns([2, 1])
                                with col_news:
                                    st.markdown("**📰 뉴스 감정 분석 (FMP + TextBlob)**")
                                    news_sentiment = results.get('news_sentiment', ["정보 없음."])
                                    if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                                        st.info(news_sentiment[0]) # 헤더 (평균 감성 등)
                                        if len(news_sentiment) > 1:
                                            with st.expander("뉴스 목록 보기"):
                                                for line in news_sentiment[1:]: st.write(f"- {line}") # 개별 뉴스
                                    else: st.write(str(news_sentiment))
                                with col_fng:
                                    st.markdown("**😨 공포-탐욕 지수**")
                                    fng_index = results.get('fear_greed_index', "N/A")
                                    if isinstance(fng_index, dict): st.metric("현재 지수", fng_index.get('value', 'N/A'), fng_index.get('classification', ''))
                                    else: st.write(fng_index)
                                st.divider()

                                # 6. Prophet 주가 예측
                                st.subheader("Prophet 주가 예측")
                                forecast_fig = results.get('forecast_fig')
                                forecast_data_list = results.get('prophet_forecast')
                                if forecast_fig: st.plotly_chart(forecast_fig, use_container_width=True)
                                elif isinstance(forecast_data_list, str): st.info(forecast_data_list)
                                else: st.warning("예측 차트 생성 실패.")

                                if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                    st.markdown("**📊 예측 데이터 (최근 10일)**")
                                    try:
                                        df_fcst = pd.DataFrame(forecast_data_list); df_fcst['ds'] = pd.to_datetime(df_fcst['ds'])
                                        df_fcst_display = df_fcst.sort_values("ds").iloc[-10:].copy(); df_fcst_display['ds'] = df_fcst_display['ds'].dt.strftime('%Y-%m-%d')
                                        format_dict_fcst = {col: "{:.2f}" for col in ['yhat', 'yhat_lower', 'yhat_upper'] if col in df_fcst_display.columns}
                                        st.dataframe(df_fcst_display[['ds'] + list(format_dict_fcst.keys())].style.format(format_dict_fcst), use_container_width=True)
                                    except Exception as e: st.error(f"예측 데이터 표시 오류: {e}")

                                cv_plot_path = results.get('cv_plot_path')
                                if cv_plot_path and os.path.exists(cv_plot_path):
                                    st.markdown("**📉 교차 검증 결과 (MAPE)**")
                                    try: st.image(cv_plot_path, caption="MAPE (낮을수록 정확)")
                                    except Exception as img_e: st.warning(f"CV 이미지 로드 실패: {img_e}")
                                elif cv_plot_path is None and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0: st.caption("교차 검증(CV) 결과 없음.")
                                st.divider()

                                # 7. 리스크 트래커
                                st.subheader("🚨 리스크 트래커 (예측 기반)")
                                # ... (리스크 트래커 표시 로직 - 이전 답변과 동일) ...
                                # (이 부분은 stock_analysis.py의 결과에 의존하므로 app.py 코드는 크게 변경 없음)
                                risk_days, max_loss_pct, max_loss_amt = 0, 0, 0
                                if avg_p > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                                    # ... (리스크 계산 로직) ...
                                    try:
                                        df_pred = pd.DataFrame(forecast_data_list)
                                        required_fcst_cols = ['ds', 'yhat_lower']; valid_pred = False
                                        if all(col in df_pred.columns for col in required_fcst_cols):
                                            df_pred['ds'] = pd.to_datetime(df_pred['ds'], errors='coerce')
                                            df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], errors='coerce')
                                            df_pred.dropna(subset=['ds', 'yhat_lower'], inplace=True)
                                            if not df_pred.empty: valid_pred = True

                                        if valid_pred:
                                            df_pred['평단가'] = avg_p; df_pred['리스크 여부'] = df_pred['yhat_lower'] < avg_p
                                            df_pred['예상 손실률'] = np.where((df_pred['리스크 여부']) & (avg_p != 0), ((df_pred['yhat_lower'] - avg_p) / avg_p) * 100, 0).fillna(0)
                                            if qty > 0: df_pred['예상 손실액'] = np.where(df_pred['리스크 여부'], (df_pred['yhat_lower'] - avg_p) * qty, 0).fillna(0)
                                            else: df_pred['예상 손실액'] = 0

                                            risk_days = df_pred['리스크 여부'].sum()
                                            if risk_days > 0:
                                                valid_loss_pct = df_pred.loc[df_pred['리스크 여부'], '예상 손실률'].dropna()
                                                max_loss_pct = valid_loss_pct.min() if not valid_loss_pct.empty else 0
                                                if qty > 0:
                                                    valid_loss_amt = df_pred.loc[df_pred['리스크 여부'], '예상 손실액'].dropna()
                                                    max_loss_amt = valid_loss_amt.min() if not valid_loss_amt.empty else 0
                                                else: max_loss_amt = 0
                                            else: max_loss_pct, max_loss_amt = 0, 0

                                            st.markdown("##### 리스크 요약")
                                            col_r1, col_r2, col_r3 = st.columns(3)
                                            col_r1.metric("⚠️ < 평단가 일수", f"{risk_days}일 / {days}일"); col_r2.metric("📉 Max 손실률", f"{max_loss_pct:.2f}%")
                                            if qty > 0: col_r3.metric("💸 Max 손실액", f"${max_loss_amt:,.2f}")

                                            if risk_days > 0: st.warning(f"{days}일 예측 중 **{risk_days}일** 평단가(${avg_p:.2f}) 하회 가능성.")
                                            else: st.success(f"{days}일간 평단가(${avg_p:.2f}) 하회 가능성 낮음.")

                                            st.markdown("##### 평단가 vs 예측 구간 비교"); fig_risk = go.Figure()
                                            plot_cols_risk = {'yhat_lower': 'rgba(0,100,80,0.2)', 'yhat_upper': 'rgba(0,100,80,0.2)', 'yhat': 'rgba(0,100,80,0.6)'}
                                            df_plot_risk = df_pred[['ds'] + list(plot_cols_risk.keys())].copy()
                                            for col in plot_cols_risk: df_plot_risk[col] = pd.to_numeric(df_plot_risk[col], errors='coerce')
                                            df_plot_risk.dropna(subset=['ds'] + list(plot_cols_risk.keys()), how='any', inplace=True)

                                            if not df_plot_risk.empty:
                                                # ... (fig_risk 차트 그리기 로직 유지) ...
                                                if 'yhat_upper' in df_plot_risk.columns: fig_risk.add_trace(go.Scatter(x=df_plot_risk['ds'], y=df_plot_risk['yhat_upper'], mode='lines', line_color=plot_cols_risk['yhat_upper'], name='Upper'))
                                                if 'yhat_lower' in df_plot_risk.columns: fig_risk.add_trace(go.Scatter(x=df_plot_risk['ds'], y=df_plot_risk['yhat_lower'], mode='lines', line_color=plot_cols_risk['yhat_lower'], name='Lower', fill='tonexty' if 'yhat_upper' in df_plot_risk.columns else None, fillcolor='rgba(0,100,80,0.1)'))
                                                if 'yhat' in df_plot_risk.columns: fig_risk.add_trace(go.Scatter(x=df_plot_risk['ds'], y=df_plot_risk['yhat'], mode='lines', line=dict(dash='dash', color=plot_cols_risk['yhat']), name='Forecast'))
                                                fig_risk.add_hline(y=avg_p, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_p:.2f}", annotation_position="bottom right")
                                                df_risk_periods = df_pred[df_pred['리스크 여부']];
                                                if not df_risk_periods.empty: fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day'))
                                                fig_risk.update_layout(hovermode="x unified"); st.plotly_chart(fig_risk, use_container_width=True)
                                                if risk_days > 0:
                                                    with st.expander(f"리스크 예측일 상세 데이터 ({risk_days}일)"):
                                                        df_risk_days_display = df_pred[df_pred['리스크 여부']].copy(); df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d')
                                                        cols_show = ['ds', 'yhat_lower', '평단가', '예상 손실률']; formatters = {"yhat_lower":"{:.2f}", "평단가":"{:.2f}", "예상 손실률":"{:.2f}%"}
                                                        if qty > 0 and '예상 손실액' in df_risk_days_display.columns: cols_show.append('예상 손실액'); formatters["예상 손실액"] = "${:,.2f}"
                                                        st.dataframe(df_risk_days_display[cols_show].style.format(formatters), use_container_width=True)
                                            else: st.info("차트 표시에 필요한 유효한 예측 데이터가 부족합니다.")
                                        else: st.info("리스크 분석 위한 유효한 데이터가 없습니다.") # df_pred가 유효하지 않은 경우
                                    except Exception as risk_calc_err:
                                        st.error(f"리스크 트래커 계산/표시 중 오류 발생: {risk_calc_err}")
                                        logging.error(f"Risk tracker error: {traceback.format_exc()}")
                                elif avg_p <= 0: st.info("⬅️ 사이드바에서 '평단가' 입력 시 리스크 분석 결과를 확인할 수 있습니다.")
                                else: st.warning("예측 데이터가 유효하지 않아 리스크 분석을 수행할 수 없습니다.")
                                st.divider()

                                # 8. 자동 분석 결과 요약
                                st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                                # ... (요약 생성 로직 - 이전 답변과 동일) ...
                                summary_points = []
                                if isinstance(forecast_data_list, list) and forecast_data_list:
                                    try:
                                        df_pred_summary = pd.DataFrame(forecast_data_list) # 예측 데이터 다시 사용
                                        if all(col in df_pred_summary.columns for col in ['yhat', 'yhat_lower', 'yhat_upper']):
                                             start_pred = df_pred_summary["yhat"].iloc[0]; end_pred = df_pred_summary["yhat"].iloc[-1]
                                             if pd.notna(start_pred) and pd.notna(end_pred): trend_obs = ("상승" if end_pred > start_pred * 1.02 else "하락" if end_pred < start_pred * 0.98 else "횡보")
                                             else: trend_obs = "판단 불가"
                                             lower = df_pred_summary["yhat_lower"].min() if df_pred_summary['yhat_lower'].notna().any() else 'N/A'
                                             upper = df_pred_summary["yhat_upper"].max() if df_pred_summary['yhat_upper'].notna().any() else 'N/A'
                                             lower_str = f"${lower:.2f}" if isinstance(lower, (int, float)) else lower; upper_str = f"${upper:.2f}" if isinstance(upper, (int, float)) else upper
                                             summary_points.append(f"- **예측:** 향후 {days}일간 **{trend_obs}** 추세 ({lower_str} ~ {upper_str})")
                                        else: summary_points.append("- 예측: 예측 결과 컬럼 부족")
                                    except Exception as e: summary_points.append(f"- 예측: 요약 생성 오류: {e}")
                                else: summary_points.append("- 예측: 예측 데이터 부족/오류로 요약 불가")

                                if isinstance(news_sentiment, list) and len(news_sentiment) > 0 and ":" in news_sentiment[0]:
                                    try:
                                        score_part = news_sentiment[0].split(":")[-1].strip(); avg_score = float(score_part)
                                        sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"
                                        summary_points.append(f"- **뉴스:** 평균 감성 {avg_score:.2f}, **{sentiment_desc}** 분위기.")
                                    except Exception as e: summary_points.append(f"- 뉴스: 요약 오류: {e}")
                                elif isinstance(news_sentiment, list): summary_points.append(f"- 뉴스: {news_sentiment[0]}")
                                else: summary_points.append("- 뉴스: 감성 분석 정보 없음/오류.")

                                if isinstance(fng_index, dict): summary_points.append(f"- **시장 심리:** 공포-탐욕 {fng_index.get('value', 'N/A')} (**{fng_index.get('classification', 'N/A')}**).")
                                else: summary_points.append("- 시장 심리: 공포-탐욕 지수 정보 없음/오류.")

                                if fundamentals and isinstance(fundamentals, dict):
                                    per = fundamentals.get("PER", "N/A"); sector = fundamentals.get("업종", "N/A")
                                    parts = [f"PER {per}"] if per != "N/A" else []; parts.extend([f"업종 '{sector}'"] if sector != "N/A" else [])
                                    if parts: summary_points.append(f"- **기본 정보:** {', '.join(parts)}.")
                                    else: summary_points.append("- 기본 정보: 주요 지표(PER, 업종) 없음.")
                                else: summary_points.append("- 기본 정보: 로드 실패/정보 없음.")

                                trend_parts = []; trend_labels = {'operating_margin_trend': '영업익률','roe_trend': 'ROE','debt_to_equity_trend': '부채비율','current_ratio_trend': '유동비율'}; trend_suffix = {'operating_margin_trend': '%','roe_trend': '%','debt_to_equity_trend': '','current_ratio_trend': ''}; trend_value_keys = {'operating_margin_trend': 'Op Margin (%)','roe_trend': 'ROE (%)','debt_to_equity_trend': 'D/E Ratio','current_ratio_trend': 'Current Ratio'}
                                try:
                                    for key in trend_labels:
                                        trend_list = results.get(key);
                                        if trend_list and isinstance(trend_list, list):
                                            last_item = trend_list[-1]; value_key = trend_value_keys[key]; value = last_item.get(value_key)
                                            if isinstance(value, (int, float)): trend_parts.append(f"{trend_labels[key]} {value:.2f}{trend_suffix[key]}")
                                            elif value is not None: trend_parts.append(f"{trend_labels[key]}: {value}")
                                            else: trend_parts.append(f"{trend_labels[key]} 정보 부족")
                                    if trend_parts: summary_points.append(f"- **최근 재무:** {', '.join(trend_parts)}.")
                                except Exception as e: summary_points.append(f"- 최근 재무: 요약 오류: {e}")

                                if avg_p > 0 and valid_pred: # 리스크 요약 조건 강화
                                    if risk_days > 0: summary_points.append(f"- **리스크:** {days}일 중 **{risk_days}일** 평단가(${avg_p:.2f}) 하회 가능성 (Max 손실률: **{max_loss_pct:.2f}%**).")
                                    else: summary_points.append(f"- **리스크:** 예측 기간 내 평단가(${avg_p:.2f}) 하회 가능성 낮음.")
                                elif avg_p > 0: summary_points.append(f"- 리스크: 평단가(${avg_p:.2f}) 입력됨, 예측 데이터 부족/오류로 분석 불가.")

                                if summary_points:
                                    st.markdown("\n".join(summary_points))
                                    st.caption("⚠️ **주의:** 자동 생성된 요약이며 투자 결정의 근거가 될 수 없습니다.")
                                else: st.write("분석 요약을 생성할 수 없습니다.")

                            # --- 결과 표시 영역 끝 ---
                    elif results is None: results_placeholder.error("분석 결과 처리 중 오류 발생 (결과 없음).")
                    else: results_placeholder.error("분석 결과 처리 중 오류 발생 (결과 형식 오류).")

                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logging.error(f"종합 분석 메인 로직 실행 오류: {e}\n{error_traceback}")
                    results_placeholder.error(f"앱 실행 중 오류 발생: {e}")

    else: # 종합 분석 버튼 클릭 전
        if comprehensive_analysis_possible: results_placeholder.info("⬅️ 사이드바에서 설정을 확인하고 '종합 분석 시작!' 버튼을 클릭하세요.")
        else: results_placeholder.warning("API 키 로드 실패로 종합 분석을 진행할 수 없습니다.")


# ============== 📈 기술 분석 탭 ==============
elif page == "📈 기술 분석":
    st.title("📈 기술적 분석 (VWAP + Bollinger + Fibonacci) - FMP API") # API 출처 명시
    st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준을 함께 시각화하고 자동 해석을 제공합니다.")
    st.markdown("---")
    ticker_tech = st.text_input("종목 티커", "AAPL", key="tech_ticker", help="해외(예: AAPL) 또는 국내(예: 005930 - FMP는 .KS 제외)")

    today = datetime.now().date()
    default_start_date = today - relativedelta(months=3)
    min_date_allowed = today - relativedelta(years=5) # FMP 제한 따라 조정 가능

    col1, col2, col3 = st.columns(3)
    with col1: start_date = st.date_input("시작일", default_start_date, key="tech_start", min_value=min_date_allowed, max_value=today - timedelta(days=1))
    with col2: end_date = st.date_input("종료일", today, key="tech_end", min_value=start_date + timedelta(days=1), max_value=today)
    with col3:
        # FMP interval 옵션 (FMP 문서 및 사용자 플랜 확인 필요)
        available_intervals = {"일봉": "1day", "1시간": "1hour", "15분": "15min"} # 예시: 사용 가능 간격
        interval_help_fmp = """데이터 간격 선택 (FMP 기준):\n- FMP 플랜에 따라 지원 간격 다름.\n- 분봉 데이터는 유료 플랜 필요 가능성.\n* 조회 기간 제한도 FMP 플랜 따라 다름."""
        interval_display = st.selectbox("데이터 간격", list(available_intervals.keys()), key="tech_interval_display", help=interval_help_fmp)
        interval = available_intervals[interval_display]

    bb_window_val = st.session_state.get('bb_window', 20)
    bb_std_val = st.session_state.get('bb_std', 2.0)
    analyze_button_tech = st.button("📊 기술적 분석 실행", key="tech_analyze", use_container_width=True, type="primary")

    if analyze_button_tech:
        if not ticker_tech: st.warning("종목 티커를 입력해주세요.")
        else:
            ticker_processed_tech = ticker_tech.strip().upper()
            if '.KS' in ticker_processed_tech or '.KQ' in ticker_processed_tech:
                 original_ticker_tech = ticker_processed_tech; ticker_processed_tech = ticker_processed_tech.split('.')[0]
                 st.info(f"국내 티커 감지: {original_ticker_tech} -> {ticker_processed_tech} (FMP용)")

            df_tech = pd.DataFrame() # 초기화
            st.session_state['error_shown_tech'] = False # 오류 메시지 중복 방지 플래그
            st.write(f"**{ticker_processed_tech}** ({interval_display}, BB:{bb_window_val}일/{bb_std_val:.1f}σ) 분석 중 (FMP API 사용)...")

            with st.spinner(f"{ticker_processed_tech} 데이터 로딩 및 처리 중 (FMP)..."):
                try:
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    fmp_data = None; rename_map = {}

                    # --- FMP API 호출 구현 ---
                    if interval == "1day":
                        fmp_data = fmp_api.get_price_data(ticker=ticker_processed_tech, start_date=start_date_str, end_date=end_date_str)
                        rename_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                    else:
                        # fmp_api.py에 get_intraday_data 구현 필요
                        fmp_data = fmp_api.get_intraday_data(ticker=ticker_processed_tech, interval=interval, from_date=start_date_str, to_date=end_date_str)
                        # FMP 분봉 응답 컬럼 확인 후 rename_map 조정 필요
                        rename_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'} # 예시

                    # --- FMP 데이터 처리 (공통 로직) ---
                    if fmp_data and isinstance(fmp_data, list):
                        df_tech = pd.DataFrame(fmp_data)
                        if not df_tech.empty:
                            df_tech = df_tech.rename(columns=rename_map)
                            date_col_name = rename_map.get('date', 'Date') # Get the target 'Date' column name

                            if date_col_name in df_tech.columns:
                                df_tech[date_col_name] = pd.to_datetime(df_tech[date_col_name], errors='coerce') # Robust datetime conversion
                                df_tech = df_tech.set_index(date_col_name).sort_index()
                                # 필수 컬럼 숫자 변환
                                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                    if col in df_tech.columns:
                                        df_tech[col] = pd.to_numeric(df_tech[col], errors='coerce')
                                df_tech.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # 가격 NaN 행 제거
                            else:
                                st.error(f"FMP 응답에 날짜 컬럼 '{date_col_name}' 이 없습니다.")
                                df_tech = pd.DataFrame() # 오류 시 비우기
                        else: # 빈 리스트 반환 시
                             st.warning(f"FMP에서 '{ticker_processed_tech}' ({interval_display}) 데이터를 찾을 수 없습니다.")
                             df_tech = pd.DataFrame()
                    elif fmp_data is None: # API 호출 함수 내부 오류 등
                         st.error(f"FMP 데이터 로딩 중 오류 발생 (API 결과 None).")
                         df_tech = pd.DataFrame()
                    else: # 빈 리스트 반환 시
                         st.warning(f"FMP에서 '{ticker_processed_tech}' ({interval_display}) 데이터를 가져오지 못했습니다 (API 결과 빈 리스트).")
                         df_tech = pd.DataFrame()

                    # --- 데이터 후처리 및 분석 ---
                    if df_tech.empty:
                        if not st.session_state.get('error_shown_tech'):
                             st.error(f"❌ 데이터를 조회/처리하지 못했습니다.")
                             st.session_state['error_shown_tech'] = True
                    else:
                        st.session_state['error_shown_tech'] = False
                        logging.info(f"FMP 데이터 처리 완료 ({ticker_processed_tech}, {interval_display}). 행 수: {len(df_tech)}, 컬럼: {df_tech.columns.tolist()}")
                        st.caption(f"조회된 데이터 기간 (FMP): {df_tech.index.min()} ~ {df_tech.index.max()}")

                        required_cols_tech = ['Open', 'High', 'Low', 'Close', 'Volume']
                        missing_cols_tech = [col for col in required_cols_tech if col not in df_tech.columns]
                        if missing_cols_tech:
                            st.error(f"❌ FMP 데이터 처리 후 필수 컬럼 누락: {missing_cols_tech}.")
                            st.dataframe(df_tech.head())
                        else:
                            # --- 지표 계산 ---
                            df_calculated = df_tech.copy(); df_calculated.attrs['ticker'] = ticker_processed_tech
                            try: df_calculated = calculate_vwap(df_calculated)
                            except ValueError as ve: st.warning(f"VWAP 계산 불가: {ve}")
                            except Exception as e: st.warning(f"VWAP 계산 오류: {e}", icon="⚠️")

                            try: df_calculated = calculate_bollinger_bands(df_calculated, window=bb_window_val, num_std=bb_std_val)
                            except ValueError as ve: st.warning(f"볼린저 밴드 계산 불가: {ve}")
                            except Exception as e: st.warning(f"BB 계산 오류: {e}", icon="⚠️")

                            try: df_calculated = calculate_rsi(df_calculated) # short_term_analysis 임포트 가정
                            except Exception as e: st.warning(f"RSI 계산 불가: {e}", icon="⚠️")

                            try: df_calculated = calculate_macd(df_calculated) # short_term_analysis 임포트 가정
                            except Exception as e: st.warning(f"MACD 계산 불가: {e}", icon="⚠️")

                            # --- 차트 표시 ---
                            st.subheader(f"📌 {ticker_processed_tech} 기술적 분석 통합 차트 ({interval_display})")
                            chart_tech = plot_technical_chart(df_calculated, ticker_processed_tech)
                            if chart_tech and chart_tech.data: st.plotly_chart(chart_tech, use_container_width=True)
                            else: st.warning("차트 생성 실패/표시할 데이터 없음.")

                            # --- 최근 데이터 표시 ---
                            st.subheader("📄 최근 데이터 (계산된 지표 포함)")
                            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
                            display_cols_exist = [col for col in display_cols if col in df_calculated.columns]
                            format_dict = {col: "{:.2f}" for col in display_cols_exist if col != 'Volume'}; format_dict['Volume'] = "{:,.0f}"
                            st.dataframe(df_calculated[display_cols_exist].tail(10).style.format(format_dict), use_container_width=True)

                            # --- 자동 해석 ---
                            st.divider(); st.subheader("🧠 기술적 시그널 해석 (참고용)")
                            if not df_calculated.empty:
                                latest_row = df_calculated.iloc[-1].copy()
                                signal_messages = []; prev_close_fib = None
                                if len(df_calculated) >= 2: prev_close_fib = df_calculated['Close'].iloc[-2]

                                try: signal_messages.extend(interpret_technical_signals(latest_row, df_context=df_calculated)) # df_context 전달
                                except Exception as e: st.warning(f"기본 기술적 시그널 해석 오류: {e}", icon="⚠️")

                                # 피보나치 해석은 interpret_technical_signals 내부에서 처리하도록 변경 권장 또는 아래 유지
                                # try:
                                #     fib_msg = interpret_fibonacci(df_calculated, close_value=latest_row["Close"], prev_close=prev_close_fib)
                                #     if fib_msg: signal_messages.append(fib_msg)
                                # except Exception as e_fib: st.warning(f"피보나치 시그널 해석 오류: {e_fib}", icon="⚠️")

                                if signal_messages:
                                    for msg in signal_messages: st.info(msg)
                                else: st.info("현재 특별히 감지된 기술적 시그널은 없습니다.")
                                st.caption("⚠️ **주의:** 자동 해석은 보조 지표이며 투자 결정은 반드시 종합적인 판단 하에 신중하게 내리시기 바랍니다.")
                            else: st.warning("해석할 데이터가 부족합니다.")

                # --- FMP API 관련 및 기타 오류 처리 ---
                except requests.exceptions.RequestException as req_err:
                    st.error(f"FMP API 요청 실패: {req_err}"); logging.error(f"FMP API request error (Tech Tab): {req_err}")
                except EnvironmentError as env_err: # fmp_api._request 에서 발생
                    st.error(f"FMP API 키 설정 오류: {env_err}"); logging.error(f"FMP API key error (Tech Tab): {env_err}")
                except Exception as e:
                    st.error(f"기술적 분석 처리 중 오류 발생: {type(e).__name__} - {e}"); logging.error(f"Tech analysis tab error: {traceback.format_exc()}")
                    # if 'df_tech' in locals() and not df_tech.empty: st.dataframe(df_tech.head()) # 디버깅용

    else: # 기술 분석 버튼 클릭 전
        st.info("종목 티커, 기간, 데이터 간격 등을 설정한 후 '기술적 분석 실행' 버튼을 클릭하세요.")


# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 (FMP API) | 정보 제공 목적 (투자 조언 아님)")
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")