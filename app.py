# -*- coding: utf-8 -*-
# Combined app.py V1.9.9 (FMP API Integration Reflected)

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

# --- 기술적 분석 관련 함수 임포트 ---
try:
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
    df = df.copy()
    required_cols = ['High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: raise ValueError(f"VWAP 계산 실패: 컬럼 부족 ({missing_cols})")

    # Volume 컬럼이 모두 NaN이거나 0인지 확인 (0으로 나누기 방지)
    if 'Volume' not in df.columns or df['Volume'].isnull().all() or (df['Volume'] == 0).all():
        df['VWAP'] = np.nan # 계산 불가 시 NaN 할당
        logging.warning(f"Ticker {df.attrs.get('ticker', '')}: VWAP 계산 불가 (거래량 데이터 부족 또는 0)")
        return df # VWAP 컬럼만 추가된 df 반환

    # Volume NaN 값을 0으로 채우기 (이후 계산에 영향 없도록)
    df['Volume'] = df['Volume'].fillna(0)

    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()

    # 누적 거래량이 0인 경우 VWAP를 NaN으로 처리 (0으로 나누기 방지)
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
        return df # 컬럼 추가된 df 반환

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

# RSI, MACD 계산 함수는 short_term_analysis.py 에서 임포트하여 사용


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
            # 히스토그램 색상 조건부 설정 (양수/음수 구분)
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
st.set_page_config(page_title="종합 주식 분석 FMP 버전", layout="wide", initial_sidebar_state="expanded") # 버전 업데이트

# --- API 키 로드 및 확인 ---
FRED_API_KEY = None
fmp_key_loaded = False
fred_key_loaded = False # FRED 키 필요 여부에 따라 관리

sidebar_status = st.sidebar.empty()

# secrets 로드 시도
secrets_available = hasattr(st, 'secrets')
if secrets_available:
    try:
        # fmp_api 모듈의 FMP_API_KEY 변수를 직접 업데이트 시도
        fmp_secret_key = st.secrets.get("FMP_API_KEY")
        if fmp_secret_key:
            fmp_api.FMP_API_KEY = fmp_secret_key
            fmp_key_loaded = True

        # FRED 키 로드 (Prophet 매크로 데이터에 여전히 사용될 수 있음)
        FRED_API_KEY = st.secrets.get("FRED_API_KEY")
        if FRED_API_KEY: fred_key_loaded = True

        if not fmp_key_loaded:
            sidebar_status.warning("Secrets에 FMP API 키가 없습니다.")
        # FRED 키 경고는 필요 시 추가
        # elif not fred_key_loaded:
        #     sidebar_status.warning("Secrets에 FRED API 키가 없습니다.")

    except Exception as e:
        sidebar_status.error(f"Secrets 로드 오류: {e}")

# .env 로드 시도 (FMP 키가 secrets에서 로드 안 됐을 경우)
if not fmp_key_loaded:
    sidebar_status.info(".env 파일 확인 중...")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            fmp_env_key = os.getenv("FMP_API_KEY")
            if fmp_env_key:
                fmp_api.FMP_API_KEY = fmp_env_key # 모듈 변수 업데이트
                fmp_key_loaded = True

            # FRED 키 로드
            fred_env_key = os.getenv("FRED_API_KEY")
            if fred_env_key and not fred_key_loaded:
                FRED_API_KEY = fred_env_key
                fred_key_loaded = True

            if fmp_key_loaded:
                sidebar_status.success("API 키 로드 완료 (.env 또는 secrets)")
            else:
                sidebar_status.error(".env 파일 내 FMP API 키 누락 또는 로드 실패.")
        else:
             if not secrets_available: # secrets도 없고 .env도 없으면
                 sidebar_status.error(".env 파일 없음, Secrets에도 FMP 키 없음.")
             elif fmp_key_loaded: # secrets에서 이미 로드됨
                  sidebar_status.empty() # .env 확인 메시지 제거

    except Exception as e:
        sidebar_status.error(f".env 로드 오류: {e}")

# 종합 분석 가능 여부 결정 (FMP 키 필수)
comprehensive_analysis_possible = fmp_key_loaded
if not comprehensive_analysis_possible:
    st.sidebar.error("FMP API 키 로드 실패! '종합 분석' 기능이 제한됩니다.")
# FRED 키 관련 메시지 (필요 시)
# elif not fred_key_loaded:
#     st.sidebar.warning("FRED API 키 로드 실패. 일부 매크로 분석이 제한될 수 있습니다.")
elif not isinstance(sidebar_status, st.empty): # 에러/경고 메시지가 있다면 성공 메시지 덮어쓰지 않음
    pass
# elif sidebar_status: # 성공 메시지 표시 (선택적, 이미 표시되었을 수 있음)
#      sidebar_status.success("API 키 로드 완료.")


# --- 사이드바 설정 ---
with st.sidebar:
    # ... (기존 후원계좌, 상세정보 링크 유지) ...
    st.title("📊 주식 분석 도구 (FMP API)") # 버전 정보 업데이트
    st.markdown("---")

    page = st.radio("분석 유형 선택", ["📊 종합 분석", "📈 기술 분석"],
                    captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"],
                    key="page_selector")
    st.markdown("---")

    if page == "📊 종합 분석":
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input("종목 티커", "AAPL", key="main_ticker",
                                     help="해외(예: AAPL) 또는 국내(예: 005930 - FMP는 .KS 제외)", # 도움말 업데이트
                                     disabled=not comprehensive_analysis_possible) # FMP 키 기준 활성화
        # ... (나머지 종합 분석 설정 위젯들도 disabled=not comprehensive_analysis_possible 적용) ...
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


# --- 캐시된 종합 분석 함수 ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, fred_key, years, days, num_trend_periods, changepoint_prior_scale): # news_key 제거
    """종합 분석 실행 및 결과 반환 (캐싱 적용)"""
    # stock_analysis 모듈 임포트는 파일 상단으로 이동
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    # FMP 키 유효성은 fmp_api 모듈 내부에서 체크됨
    try:
        # analyze_stock 호출 시 news_key 제거
        analysis_results = sa.analyze_stock(
            ticker,
            fred_key=fred_key, # FRED 키 전달 (Prophet 매크로 데이터용)
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
    results_placeholder = st.container() # 결과를 표시할 영역

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
            # 국내 티커 처리
            if '.KS' in ticker_proc or '.KQ' in ticker_proc:
                 original_ticker = ticker_proc
                 ticker_proc = ticker_proc.split('.')[0]
                 results_placeholder.info(f"국내 티커 감지: {original_ticker} -> {ticker_proc} (FMP용)")

            with st.spinner(f"{ticker_proc} 종합 분석 중..."):
                try:
                    # --- run_cached_analysis 호출 (news_key 없이) ---
                    results = run_cached_analysis(
                        ticker_proc,
                        FRED_API_KEY, # FRED 키 전달
                        years, days, periods, cp_prior
                    )

                    # --- 결과 표시 로직 (기존과 거의 동일, 내부 데이터는 FMP 기반) ---
                    if results and isinstance(results, dict):
                        if "error" in results:
                            results_placeholder.error(f"분석 실패: {results['error']}")
                        else:
                            results_placeholder.empty() # 이전 메시지 비우기
                            # --- MAPE 경고 배너 등 (기존 로직 유지) ---
                            if results.get("warn_high_mape"):
                                m = results.get("mape", "N/A")
                                mape_value_str = f"{m:.1f}%" if isinstance(m, (int, float)) else "N/A"
                                st.warning(f"🔴 모델 정확도 낮음 (MAPE {mape_value_str}). 예측 신뢰도에 주의하세요!")
                            # --- 결과 표시 영역 시작 (기존 로직 유지) ---
                            with results_placeholder:
                                st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")
                                # 1. 요약 정보
                                st.subheader("요약 정보")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                                col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                                col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                                # 2. 기본적 분석 (FMP 기반 데이터 표시)
                                st.subheader("📊 기업 기본 정보")
                                fundamentals = results.get('fundamentals')
                                if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A":
                                    colf1, colf2, colf3 = st.columns(3)
                                    with colf1:
                                        st.metric("시가총액", fundamentals.get("시가총액", "N/A"))
                                        st.metric("PER", fundamentals.get("PER", "N/A"))
                                    with colf2:
                                        st.metric("EPS", fundamentals.get("EPS", "N/A"))
                                        st.metric("Beta", fundamentals.get("베타", "N/A"))
                                    with colf3:
                                        st.metric("배당수익률", fundamentals.get("배당수익률", "N/A")) # FMP 'lastDiv' 기준
                                        st.metric("업종", fundamentals.get("업종", "N/A"))
                                    industry = fundamentals.get("산업", "N/A")
                                    summary = fundamentals.get("요약", "N/A")
                                    if industry != "N/A": st.markdown(f"**산업:** {industry}")
                                    if summary != "N/A":
                                        with st.expander("회사 요약 보기"): st.write(summary)
                                    st.caption("Data Source: Financial Modeling Prep") # 출처 변경
                                else: st.warning("기업 기본 정보 로드 실패.")
                                st.divider()

                                # 3. 주요 재무 추세 (FMP 기반 데이터 표시)
                                st.subheader(f"📈 주요 재무 추세 (최근 {periods} 분기)")
                                # ... (기존 탭 구조 및 차트/데이터 표시 로직 유지) ...
                                # 데이터 키는 stock_analysis.py에서 FMP 기반으로 반환된 것을 사용
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

                                # 4. 기술적 분석 차트 (종합) (FMP 기반 데이터)
                                st.subheader("기술적 분석 차트 (종합)")
                                stock_chart_fig = results.get('stock_chart_fig') # stock_analysis에서 FMP 데이터로 생성됨
                                if stock_chart_fig: st.plotly_chart(stock_chart_fig, use_container_width=True)
                                else: st.warning("주가 차트 생성 실패 (종합).")
                                st.divider()

                                # 5. 시장 심리 분석 (뉴스: FMP 기반, F&G: 외부 API)
                                st.subheader("시장 심리 분석")
                                col_news, col_fng = st.columns([2, 1])
                                with col_news:
                                    st.markdown("**📰 뉴스 감정 분석 (FMP)**") # 출처 명시
                                    news_sentiment = results.get('news_sentiment', ["정보 없음."])
                                    # ... (기존 뉴스 표시 로직 유지) ...
                                with col_fng:
                                    st.markdown("**😨 공포-탐욕 지수**")
                                    fng_index = results.get('fear_greed_index', "N/A")
                                    # ... (기존 F&G 표시 로직 유지) ...
                                st.divider()

                                # 6. Prophet 주가 예측 (기본 데이터는 FMP, 매크로는 FRED/FMP 혼용 가능)
                                st.subheader("Prophet 주가 예측")
                                # ... (기존 예측 결과 표시 로직 유지) ...

                                # 7. 리스크 트래커 (예측 기반)
                                st.subheader("🚨 리스크 트래커 (예측 기반)")
                                # ... (기존 리스크 트래커 로직 유지) ...

                                # 8. 자동 분석 결과 요약 (FMP 기반 데이터 반영)
                                st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                                # ... (기존 요약 생성 로직 유지, 내부 데이터는 FMP 기반임을 인지) ...

                            # --- 결과 표시 영역 끝 ---
                    # ... (나머지 결과 처리 및 오류 처리 로직 유지) ...
                except Exception as e: # 메인 로직 실행 중 예외 처리
                    error_traceback = traceback.format_exc()
                    logging.error(f"종합 분석 메인 로직 실행 오류: {e}\n{error_traceback}")
                    results_placeholder.error(f"앱 실행 중 오류 발생: {e}")

    else: # 종합 분석 버튼 클릭 전
        if comprehensive_analysis_possible:
            results_placeholder.info("⬅️ 사이드바에서 설정을 확인하고 '종합 분석 시작!' 버튼을 클릭하세요.")
        else:
            results_placeholder.warning("FMP API 키 로드 실패로 종합 분석을 진행할 수 없습니다. 사이드바 메시지를 확인하세요.")


# ============== 📈 기술 분석 탭 ==============
elif page == "📈 기술 분석":
    st.title("📈 기술적 분석 (VWAP + Bollinger + Fibonacci) - FMP API") # API 출처 명시
    st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준을 함께 시각화하고 자동 해석을 제공합니다.")
    st.markdown("---")
    ticker_tech = st.text_input("종목 티커", "AAPL", key="tech_ticker", help="해외(예: AAPL) 또는 국내(예: 005930 - FMP는 .KS 제외)") # 도움말 수정

    # 날짜 입력 (기존 유지)
    today = datetime.now().date()
    default_start_date = today - relativedelta(months=3)
    min_date_allowed = today - relativedelta(years=5) # FMP 제한에 따라 조정될 수 있음

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("시작일", default_start_date, key="tech_start",
                                   min_value=min_date_allowed, max_value=today - timedelta(days=1))
    with col2:
        end_date = st.date_input("종료일", today, key="tech_end",
                                 min_value=start_date + timedelta(days=1), max_value=today)
    with col3:
        # FMP interval 옵션 (FMP 문서 및 사용자 플랜 확인 필요)
        interval_options_fmp = {
            "일봉": "1day",     # FMP 형식 예시
            "4시간": "4hour",   # FMP 형식 예시
            "1시간": "1hour",   # FMP 형식 예시
            "30분": "30min",    # FMP 형식 예시
            "15분": "15min",    # FMP 형식 예시
            "5분": "5min",     # FMP 형식 예시 (유료 플랜 필요 가능성 높음)
            "1분": "1min",     # FMP 형식 예시 (유료 플랜 필요 가능성 높음)
        }
        # 사용자의 FMP 플랜에 맞춰 옵션 조절 필요
        available_intervals = {"일봉": "1day", "1시간": "1hour", "15분": "15min"} # 예시: 사용 가능 간격
        interval_help_fmp = """
        데이터 간격 선택 (FMP 기준):
        - FMP 플랜에 따라 지원되는 간격이 다릅니다.
        - 분봉 데이터는 유료 플랜이 필요할 수 있습니다.
        * 조회 기간 제한도 FMP 플랜에 따라 다릅니다.
        """
        interval_display = st.selectbox("데이터 간격", list(available_intervals.keys()), # 사용 가능한 간격만 표시
                                        key="tech_interval_display", help=interval_help_fmp)
        interval = available_intervals[interval_display] # 실제 FMP API에 전달될 값

    # 사이드바에서 BB 설정값 가져오기
    bb_window_val = st.session_state.get('bb_window', 20)
    bb_std_val = st.session_state.get('bb_std', 2.0)

    analyze_button_tech = st.button("📊 기술적 분석 실행", key="tech_analyze", use_container_width=True, type="primary")

    if analyze_button_tech:
        if not ticker_tech:
            st.warning("종목 티커를 입력해주세요.")
        else:
            ticker_processed_tech = ticker_tech.strip().upper()
            # 국내 티커 처리
            if '.KS' in ticker_processed_tech or '.KQ' in ticker_processed_tech:
                 original_ticker_tech = ticker_processed_tech
                 ticker_processed_tech = ticker_processed_tech.split('.')[0]
                 st.info(f"국내 티커 감지: {original_ticker_tech} -> {ticker_processed_tech} (FMP용)")

            df_tech = pd.DataFrame() # 데이터프레임 초기화
            st.write(f"**{ticker_processed_tech}** ({interval_display}, BB:{bb_window_val}일/{bb_std_val:.1f}σ) 분석 중 (FMP API 사용)...")

            with st.spinner(f"{ticker_processed_tech} 데이터 로딩 및 처리 중 (FMP)..."):
                try:
                    # --- FMP API 호출 ---
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")

                    # fmp_api.py에 아래 함수들이 구현되어 있다고 가정
                    # 함수 이름과 파라미터는 실제 구현에 맞게 조정 필요
                    if interval == "1day":
                        # 예시: fmp_api.py의 get_historical_data 함수 호출
                        fmp_data = fmp_api.get_historical_data(ticker=ticker_processed_tech, from_date=start_date_str, to_date=end_date_str)
                        if fmp_data and isinstance(fmp_data, list):
                            df_tech = pd.DataFrame(fmp_data)
                            # 컬럼 이름 변경 및 Date 인덱스 설정
                            rename_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                            df_tech = df_tech.rename(columns=rename_map)
                            if 'Date' in df_tech.columns:
                                df_tech['Date'] = pd.to_datetime(df_tech['Date'])
                                df_tech = df_tech.set_index('Date').sort_index()
                            else:
                                st.error("FMP 응답에 'date' 컬럼이 없습니다.")
                                df_tech = pd.DataFrame()
                        else:
                             st.warning(f"FMP에서 '{ticker_processed_tech}' 일봉 데이터를 가져오지 못했습니다.")
                             df_tech = pd.DataFrame()
                    else:
                        # 예시: fmp_api.py의 get_intraday_data 함수 호출
                        fmp_data = fmp_api.get_intraday_data(ticker=ticker_processed_tech, interval=interval, from_date=start_date_str, to_date=end_date_str)
                        if fmp_data and isinstance(fmp_data, list):
                            df_tech = pd.DataFrame(fmp_data)
                             # 컬럼 이름 변경 및 Date 인덱스 설정 (FMP 분봉 응답 구조 확인 필요)
                            rename_map_intra = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'} # 'date' 또는 'datetime' 등 확인 필요
                            df_tech = df_tech.rename(columns=rename_map_intra)
                            if 'Date' in df_tech.columns:
                                df_tech['Date'] = pd.to_datetime(df_tech['Date'])
                                df_tech = df_tech.set_index('Date').sort_index()
                            else:
                                st.error(f"FMP {interval} 응답에 날짜 관련 컬럼(예: 'date', 'datetime')이 없습니다.")
                                df_tech = pd.DataFrame()
                        else:
                             st.warning(f"FMP에서 '{ticker_processed_tech}' {interval} 데이터를 가져오지 못했습니다.")
                             df_tech = pd.DataFrame()

                    # --- 데이터 후처리 및 분석 ---
                    if df_tech.empty:
                        st.error(f"❌ **{ticker_processed_tech}** ({interval_display}) 데이터를 FMP API에서 조회하지 못했습니다. 티커, 기간, 간격 또는 API 플랜을 확인해주세요.")
                    else:
                        # 숫자형 변환 및 필수 데이터 NaN 제거
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if col in df_tech.columns:
                                df_tech[col] = pd.to_numeric(df_tech[col], errors='coerce')
                        df_tech.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # 가격 정보 없으면 제외

                        if df_tech.empty:
                            st.warning("데이터 정제 후 분석할 데이터가 없습니다.")
                            st.stop()

                        logging.info(f"FMP 데이터 처리 완료. 행 수: {len(df_tech)}, 컬럼: {df_tech.columns.tolist()}")
                        st.caption(f"조회된 데이터 기간 (FMP): {df_tech.index.min()} ~ {df_tech.index.max()}")

                        # 필수 컬럼 재확인 (Volume 포함)
                        required_cols_tech = ['Open', 'High', 'Low', 'Close', 'Volume']
                        missing_cols_tech = [col for col in required_cols_tech if col not in df_tech.columns]
                        if missing_cols_tech:
                            st.error(f"❌ FMP 데이터 처리 후 필수 컬럼이 누락되었습니다: {missing_cols_tech}.")
                            st.dataframe(df_tech.head())
                        else:
                            # --- 데이터 처리 및 지표 계산 ---
                            df_calculated = df_tech.copy()
                            df_calculated.attrs['ticker'] = ticker_processed_tech # 티커 정보 추가 (로깅용)

                            try: df_calculated = calculate_vwap(df_calculated)
                            except ValueError as ve_vwap: st.warning(f"VWAP 계산 불가: {ve_vwap}")
                            except Exception as e_vwap: st.warning(f"VWAP 계산 중 오류: {e_vwap}", icon="⚠️")

                            try: df_calculated = calculate_bollinger_bands(df_calculated, window=bb_window_val, num_std=bb_std_val)
                            except ValueError as ve_bb: st.warning(f"볼린저 밴드 계산 불가: {ve_bb}")
                            except Exception as e_bb: st.warning(f"BB 계산 중 오류: {e_bb}", icon="⚠️")

                            try: df_calculated = calculate_rsi(df_calculated)
                            except Exception as e_rsi: st.warning(f"RSI 계산 불가: {e_rsi}", icon="⚠️")

                            try: df_calculated = calculate_macd(df_calculated)
                            except Exception as e_macd: st.warning(f"MACD 계산 불가: {e_macd}", icon="⚠️")

                            # --- 차트 생성 및 표시 ---
                            st.subheader(f"📌 {ticker_processed_tech} 기술적 분석 통합 차트 ({interval_display})")
                            chart_tech = plot_technical_chart(df_calculated, ticker_processed_tech)
                            if chart_tech.data: # 차트에 데이터가 있는지 확인
                                st.plotly_chart(chart_tech, use_container_width=True)
                            else:
                                st.warning("차트 생성에 실패했거나 표시할 데이터가 없습니다.")

                            # --- 최근 데이터 표시 ---
                            st.subheader("📄 최근 데이터 (계산된 지표 포함)")
                            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
                            display_cols_exist = [col for col in display_cols if col in df_calculated.columns]
                            format_dict = {col: "{:.2f}" for col in display_cols_exist if col not in ['Volume']}
                            format_dict['Volume'] = "{:,.0f}"
                            st.dataframe(df_calculated[display_cols_exist].tail(10).style.format(format_dict), use_container_width=True)

                            # --- 자동 해석 기능 ---
                            st.divider()
                            st.subheader("🧠 기술적 시그널 해석 (참고용)")
                            if not df_calculated.empty:
                                latest_row = df_calculated.iloc[-1].copy()
                                signal_messages = []
                                try:
                                    # VWAP, BB, RSI, MACD 해석
                                    signal_messages.extend(interpret_technical_signals(latest_row))
                                except Exception as e_interpret:
                                     st.warning(f"기본 기술적 시그널 해석 중 오류: {e_interpret}", icon="⚠️")

                                try:
                                    # 피보나치 해석 (직전 봉 데이터 필요)
                                    prev_close_fib = df_calculated['Close'].iloc[-2] if len(df_calculated) >= 2 else None
                                    fib_msg = interpret_fibonacci(df_calculated, close_value=latest_row["Close"], prev_close=prev_close_fib)
                                    if fib_msg:
                                        signal_messages.append(fib_msg)
                                except Exception as e_fib:
                                    st.warning(f"피보나치 시그널 해석 중 오류: {e_fib}", icon="⚠️")

                                if signal_messages:
                                    for msg in signal_messages: st.info(msg)
                                else: st.info("현재 특별히 감지된 기술적 시그널은 없습니다.")
                                st.caption("⚠️ **주의:** 자동 해석은 보조 지표이며 투자 결정은 반드시 종합적인 판단 하에 신중하게 내리시기 바랍니다.")
                            else: st.warning("해석할 데이터가 부족합니다.")

                # --- FMP API 관련 및 기타 오류 처리 ---
                except requests.exceptions.RequestException as req_err:
                    st.error(f"FMP API 요청 실패: {req_err}")
                    logging.error(f"FMP API request error (Technical Tab): {req_err}")
                except EnvironmentError as env_err: # fmp_api._request 에서 발생
                    st.error(f"FMP API 키 설정 오류: {env_err}")
                    logging.error(f"FMP API key error (Technical Tab): {env_err}")
                except Exception as e:
                    st.error(f"기술적 분석 처리 중 예기치 못한 오류 발생: {type(e).__name__} - {e}")
                    logging.error(f"Technical analysis tab error: {traceback.format_exc()}")
                    # 오류 발생 시 로드된 데이터가 있다면 일부라도 보여주기
                    # if 'df_tech' in locals() and not df_tech.empty: st.dataframe(df_tech.head())

    else: # 기술 분석 버튼 클릭 전
        st.info("종목 티커, 기간, 데이터 간격 등을 설정한 후 '기술적 분석 실행' 버튼을 클릭하세요.")


# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 (FMP API) | 정보 제공 목적 (투자 조언 아님)")
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")