# -*- coding: utf-8 -*-
# app.py (pykrx 적용 버전)

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
import asyncio # 비동기 함수 호출을 위해 추가 (get_ticker_from_input)

# --- pykrx 임포트 ---
try:
    from pykrx import stock
except ImportError:
    st.error("pykrx 라이브러리를 찾을 수 없습니다. 'pip install pykrx'로 설치해주세요.")
    st.stop()

# --- FMP API 및 분석 모듈 임포트 ---
try:
    import fmp_api # FMP API 래퍼 모듈
    import stock_analysis as sa # 종합 분석 로직
except ImportError as e:
    st.error(f"필수 API 또는 분석 모듈 로딩 실패: {e}. 'fmp_api.py'와 'stock_analysis.py' 파일이 있는지 확인하세요.")
    st.stop()

# --- 기술 분석 관련 함수 임포트 ---
try:
    from short_term_analysis import interpret_fibonacci, calculate_rsi, calculate_macd
    from technical_interpret import interpret_technical_signals
except ImportError as e:
    st.error(f"기술 분석 모듈 로딩 실패: {e}. 'short_term_analysis.py'와 'technical_interpret.py' 파일이 있는지 확인하세요.")
    # 필요시 st.stop()

# --- DART API 모듈 임포트 (한국 기업 기본 정보용) ---
try:
    import dart as dart_api # dart.py 파일을 dart_api 라는 이름으로 임포트
except ImportError:
    st.warning("dart.py 파일을 찾을 수 없습니다. '한국 기업 기본 정보' 기능이 제한될 수 있습니다.")
    dart_api = None # dart_api 객체를 None으로 설정하여 오류 방지

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
    # 0으로 나누기 경고 방지 및 거래량 0 경우 처리
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    if (df['Volume'] == 0).all():
        df['VWAP'] = np.nan; logging.warning(f"Ticker {df.attrs.get('ticker', '')}: VWAP 계산 불가 (모든 거래량 0)"); return df

    df['typical_price'] = (pd.to_numeric(df['High'], errors='coerce') + pd.to_numeric(df['Low'], errors='coerce') + pd.to_numeric(df['Close'], errors='coerce')) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
    # 0으로 나누기 방지
    df['VWAP'] = np.where(df['cumulative_volume'] > 0, df['cumulative_tp_volume'] / df['cumulative_volume'], np.nan)
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """볼린저 밴드 계산 (Close 필요)"""
    df = df.copy(); required_col = 'Close'
    if required_col not in df.columns:
        st.warning(f"BB 계산 실패: '{required_col}' 컬럼 없음."); df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan; return df

    df[required_col] = pd.to_numeric(df[required_col], errors='coerce') # 숫자로 변환
    if df[required_col].isnull().all():
        st.warning(f"BB 계산 실패: '{required_col}' 데이터 없음."); df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan; return df

    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        st.warning(f"BB 계산 위한 유효 데이터({len(valid_close)}개)가 기간({window}개)보다 부족."); df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan; return df
    else:
        df['MA20'] = df[required_col].rolling(window=window, min_periods=window).mean()
        df['STD20'] = df[required_col].rolling(window=window, min_periods=window).std()
        df['Upper'] = df['MA20'] + num_std * df['STD20']
        df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

# --- 차트 생성 함수 (미국 주식용 - 기존 코드 유지) ---
def plot_technical_chart(df, ticker):
    """미국 주식용 기술적 분석 지표 통합 차트 생성"""
    fig = go.Figure(); required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols) or df[required_candle_cols].isnull().all(axis=None):
        st.error(f"캔들차트 필요 컬럼({required_candle_cols}) 없음/데이터 없음."); return fig
    # OHLC 데이터 숫자로 변환 (오류 발생 시 NaN 처리)
    for col in required_candle_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_candle_cols, inplace=True) # NaN 있는 행 제거
    if df.empty:
        st.error("유효한 OHLC 데이터가 없습니다."); return fig

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))
    # ... (이하 기존 plot_technical_chart 로직 유지 - VWAP, BB, Fib, RSI, MACD 추가) ...
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

    fig.update_layout(title=f"{ticker} - 기술적 분석 통합 차트 (미국 주식)", xaxis_title="날짜 / 시간", yaxis=dict(domain=[0.4, 1], title="가격 ($)"), yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right', showgrid=False), yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right', showgrid=False), xaxis_rangeslider_visible=False, legend_title_text="지표", hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
    return fig

# --- 한국 주식용 차트 생성 함수 (pykrx 데이터 기반) ---
def plot_korean_technical_chart(df, ticker_code, company_name):
    """한국 주식용 기술적 분석 지표 통합 차트 생성 (추세선 포함)"""
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols):
        logging.error(f"한국 주식 캔들차트 필요 컬럼({required_candle_cols}) 없음/데이터 없음.")
        return fig # 빈 Figure 반환
    # OHLC 데이터 숫자로 변환 (오류 발생 시 NaN 처리)
    for col in required_candle_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_candle_cols, inplace=True) # NaN 있는 행 제거
    if df.empty:
        logging.error("한국 주식 유효한 OHLC 데이터가 없습니다.")
        return fig

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{company_name} 캔들"))

    # VWAP
    if 'VWAP' in df.columns and df['VWAP'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))

    # Bollinger Bands
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20 (BB 중심선)', line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)'))

    # Fibonacci Retracement Levels
    valid_price_df_kr = df.dropna(subset=['High', 'Low'])
    if not valid_price_df_kr.empty:
        min_price_kr = valid_price_df_kr['Low'].min()
        max_price_kr = valid_price_df_kr['High'].max()
        diff_kr = max_price_kr - min_price_kr
        if diff_kr > 0:
            levels_kr = {'0.0 (High)': max_price_kr, '0.236': max_price_kr - 0.236 * diff_kr, '0.382': max_price_kr - 0.382 * diff_kr, '0.5': max_price_kr - 0.5 * diff_kr, '0.618': max_price_kr - 0.618 * diff_kr, '1.0 (Low)': min_price_kr}
            fib_colors_kr = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels_kr.items():
                fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: {v:,.0f}원", line_color=fib_colors_kr.get(k, 'navy'), annotation_position="bottom right", annotation_font_size=10)

    # 추세선 (이동평균선)
    if 'SMA5' in df.columns and df['SMA5'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'], mode='lines', name='SMA 5일', line=dict(color='green', width=1)))
    if 'SMA20' in df.columns and df['SMA20'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20일', line=dict(color='red', width=1)))
    if 'SMA60' in df.columns and df['SMA60'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], mode='lines', name='SMA 60일', line=dict(color='purple', width=1)))

    # TODO: 실제 상승/하락 추세선 그리기 (선택적 고급 기능)
    # (이전 답변의 linear regression 예시 코드는 scipy.stats 임포트 필요)
    # from scipy.stats import linregress # 필요시 추가

    # RSI
    if 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='purple', width=1), yaxis='y2'))
    # MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines', line=dict(color='teal'), yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', mode='lines', line=dict(color='orange'), yaxis='y3'))
        if 'MACD_hist' in df.columns:
            colors_kr = ['lightgreen' if v >= 0 else 'lightcoral' for v in df['MACD_hist']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color=colors_kr, yaxis='y3'))

    fig.update_layout(
        title=f"{company_name} ({ticker_code}) 기술적 분석 (한국 주식)",
        xaxis_title="날짜",
        yaxis=dict(domain=[0.4, 1], title="가격 (원)"), # Y축 레이블 변경
        yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right', showgrid=False),
        yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        legend_title_text="지표",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# --- pykrx 티커 조회 헬퍼 함수 ---
@st.cache_data(ttl=3600) # 한 시간 동안 KRX 종목 목록 캐싱
def get_kr_ticker_map():
    """KRX KOSPI, KOSDAQ 종목 목록을 가져와 이름-티커, 티커-이름 맵을 반환합니다."""
    name_to_ticker = {}
    ticker_to_name = {}
    try:
        today_str = datetime.now().strftime("%Y%m%d") # 오늘 날짜 기준으로 조회 시도
        for market in ["KOSPI", "KOSDAQ"]:
            # 특정 날짜 기준이 필요하면 해당 날짜 문자열 사용
            tickers = stock.get_market_ticker_list(date=today_str, market=market)
            for ticker in tickers:
                try:
                    # get_market_ticker_name 에도 날짜 인자 추가 가능 (필요시)
                    name = stock.get_market_ticker_name(ticker)
                    name_to_ticker[name] = ticker
                    ticker_to_name[ticker] = name
                except Exception as e_inner:
                    logging.warning(f"Ticker {ticker} 이름 조회 실패: {e_inner}") # 개별 티커 오류는 로깅만
        logging.info(f"KRX Ticker Map 로드 완료: {len(name_to_ticker)} 종목")
        return name_to_ticker, ticker_to_name
    except Exception as e:
        logging.error(f"KRX 종목 목록 조회 중 오류 발생: {e}")
        st.error(f"KRX 종목 목록을 불러오는 데 실패했습니다: {e}") # 사용자에게도 에러 표시
        return {}, {}

async def get_ticker_from_input(user_input):
    """사용자 입력(회사명 또는 종목코드)으로부터 6자리 종목코드와 회사명을 반환합니다."""
    user_input = user_input.strip()
    name_to_ticker_map, ticker_to_name_map = get_kr_ticker_map() # 캐싱된 맵 사용

    if not name_to_ticker_map and not ticker_to_name_map:
        # get_kr_ticker_map 내부에서 에러 메시지를 표시했을 것이므로 여기서는 None 반환
        return None, user_input # 실패 시 None, 원래 입력값 반환

    # 입력이 6자리 숫자이고, 유효한 티커인지 확인
    if user_input.isdigit() and len(user_input) == 6:
        if user_input in ticker_to_name_map:
            return user_input, ticker_to_name_map[user_input]
        else:
            # 6자리 숫자이지만 유효하지 않은 티커일 경우 (예: 상장 폐지 등)
            st.warning(f"입력하신 종목코드 '{user_input}'는 현재 KRX 목록에 없습니다. 상장 폐지되었거나 잘못된 코드일 수 있습니다.")
            return None, user_input # 실패

    # 입력이 회사명과 정확히 일치하는지 확인
    if user_input in name_to_ticker_map:
        return name_to_ticker_map[user_input], user_input

    # 입력이 회사명의 일부를 포함하는지 확인 (대소문자 무시, 첫 번째 매칭)
    cleaned_input = user_input.lower()
    found_ticker = None
    found_name = user_input # 기본값
    for name, ticker in name_to_ticker_map.items():
        if cleaned_input in name.lower():
            found_ticker = ticker
            found_name = name
            st.info(f"입력 '{user_input}'과(와) 유사한 '{found_name}'(으)로 검색합니다. (티커: {found_ticker})")
            return found_ticker, found_name # 첫 번째 매칭 결과 반환

    # 위 모든 경우에 해당하지 않으면 찾지 못한 것
    st.warning(f"'{user_input}'에 해당하는 종목을 찾지 못했습니다. 정확한 회사명이나 6자리 종목코드를 입력해주세요.")
    return None, user_input # 실패

# --- 한국 주식 데이터 로딩 및 차트 표시 함수 (pykrx 사용) ---
async def display_korean_stock_chart(ticker_input_kr, start_date_kr, end_date_kr, bb_window_val_kr, bb_std_val_kr, results_container):
    """한국 주식 데이터를 pykrx로 가져와 기술적 분석 차트를 표시합니다."""
    with results_container:
        # 1. 사용자 입력으로부터 종목코드 가져오기
        ticker_code, company_name_krx = await get_ticker_from_input(ticker_input_kr)

        if not ticker_code:
            # get_ticker_from_input 함수 내에서 이미 경고/오류 메시지를 표시했을 것임
            return

        # 2. pykrx 용 날짜 형식 변환 (YYYYMMDD)
        start_date_str_krx = start_date_kr.strftime("%Y%m%d")
        end_date_str_krx = end_date_kr.strftime("%Y%m%d")

        st.write(f"**{company_name_krx} ({ticker_code})** 기술적 분석 (BB: {bb_window_val_kr}일/{bb_std_val_kr:.1f}σ)")
        with st.spinner(f"{company_name_krx} ({ticker_code}) 데이터 로딩 및 차트 생성 중... (pykrx)"):
            try:
                # 3. pykrx로 주가 데이터 다운로드
                df_kr = stock.get_market_ohlcv(start_date_str_krx, end_date_str_krx, ticker_code)

                if df_kr.empty:
                    st.error(f"❌ {company_name_krx}({ticker_code})에 대한 데이터를 pykrx로 불러오지 못했습니다. 기간 내 거래가 없거나 잘못된 요청일 수 있습니다.")
                    return

                # 4. 컬럼명 변경 ('시가' -> 'Open' 등) 및 인덱스 타입 확인/변환
                df_kr.rename(columns={'시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'}, inplace=True)
                df_kr.index = pd.to_datetime(df_kr.index) # DatetimeIndex로 변환

                # 5. 기술적 지표 계산
                required_cols_kr = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols_kr = [col for col in required_cols_kr if col not in df_kr.columns]
                if missing_cols_kr:
                    st.error(f"❌ {company_name_krx} 데이터에 필수 컬럼이 부족합니다: {missing_cols_kr}")
                    st.dataframe(df_kr.head())
                    return

                df_calculated_kr = df_kr.copy()
                df_calculated_kr.attrs['ticker'] = ticker_code # Ticker 정보 추가

                # 데이터 타입을 숫자로 변환 (지표 계산 전)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                     df_calculated_kr[col] = pd.to_numeric(df_calculated_kr[col], errors='coerce')
                df_calculated_kr.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # 가격 정보 없는 행 제거

                if df_calculated_kr.empty:
                     st.error("❌ 숫자 변환 후 유효한 가격 데이터가 없습니다.")
                     return

                try: df_calculated_kr = calculate_vwap(df_calculated_kr)
                except Exception as e: st.warning(f"VWAP 계산 오류: {e}", icon="⚠️")
                try: df_calculated_kr = calculate_bollinger_bands(df_calculated_kr, window=bb_window_val_kr, num_std=bb_std_val_kr)
                except Exception as e: st.warning(f"BB 계산 오류: {e}", icon="⚠️")
                try: df_calculated_kr = calculate_rsi(df_calculated_kr)
                except Exception as e: st.warning(f"RSI 계산 오류: {e}", icon="⚠️")
                try: df_calculated_kr = calculate_macd(df_calculated_kr)
                except Exception as e: st.warning(f"MACD 계산 오류: {e}", icon="⚠️")

                # 이동평균선 계산
                df_calculated_kr['SMA5'] = df_calculated_kr['Close'].rolling(window=5).mean()
                df_calculated_kr['SMA20'] = df_calculated_kr['Close'].rolling(window=20).mean()
                df_calculated_kr['SMA60'] = df_calculated_kr['Close'].rolling(window=60).mean()

                # 6. 차트 생성
                fig_kr = plot_korean_technical_chart(df_calculated_kr, ticker_code, company_name_krx)
                if fig_kr and fig_kr.data:
                    st.plotly_chart(fig_kr, use_container_width=True)
                else:
                    st.warning("한국 주식 차트 생성에 실패했거나 표시할 데이터가 없습니다.")

                # 7. 최근 데이터 테이블 표시
                st.subheader("📄 최근 데이터 (계산된 지표 포함)")
                display_cols_kr = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA5', 'SMA20', 'SMA60']
                display_cols_exist_kr = [col for col in display_cols_kr if col in df_calculated_kr.columns]
                format_dict_kr = {col: "{:,.0f}" for col in ['Open', 'High', 'Low', 'Close', 'VWAP', 'MA20', 'Upper', 'Lower', 'SMA5', 'SMA20', 'SMA60'] if col in display_cols_exist_kr}
                if 'Volume' in display_cols_exist_kr: format_dict_kr['Volume'] = "{:,.0f}"
                for col_rsi_macd in ['RSI', 'MACD', 'MACD_signal', 'MACD_hist']:
                    if col_rsi_macd in display_cols_exist_kr:
                        format_dict_kr[col_rsi_macd] = "{:.2f}"
                st.dataframe(df_calculated_kr[display_cols_exist_kr].tail(10).style.format(format_dict_kr), use_container_width=True)

                # 8. 기술적 시그널 해석
                st.divider()
                st.subheader("🧠 기술적 시그널 해석 (참고용)")
                if not df_calculated_kr.empty:
                    latest_row_kr = df_calculated_kr.iloc[-1].copy()
                    signal_messages_kr = []
                    try:
                        if 'interpret_technical_signals' in globals():
                            signal_messages_kr.extend(interpret_technical_signals(latest_row_kr, df_context=df_calculated_kr))
                        else:
                            st.error("오류: 'interpret_technical_signals' 함수를 찾을 수 없습니다.")
                    except Exception as e:
                        st.warning(f"기술적 시그널 해석 중 오류 발생: {e}", icon="⚠️")

                    if signal_messages_kr:
                        for msg in signal_messages_kr: st.info(msg)
                    else:
                        st.info("현재 특별히 감지된 기술적 시그널은 없습니다.")
                    st.caption("⚠️ **주의:** 자동 해석은 보조 지표이며 투자 결정은 반드시 종합적인 판단 하에 신중하게 내리시기 바랍니다.")
                else:
                    st.warning("해석할 데이터가 부족합니다.")

            except Exception as e:
                st.error(f"한국 주식 데이터 처리 중 오류 발생: {type(e).__name__} - {e}")
                st.error(f"Traceback: {traceback.format_exc()}")

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="종합 주식 분석 (FMP & pykrx)", layout="wide", initial_sidebar_state="expanded") # 타이틀 변경

# --- FMP API 키 로드 및 확인 ---
# ... (기존 코드 유지) ...
fmp_key_loaded = False
sidebar_status = st.sidebar.empty()
final_status_message_displayed = False
# ... (이하 FMP 키 로드 로직 전체) ...
secrets_available = hasattr(st, 'secrets')
if secrets_available:
    try:
        fmp_secret_key = st.secrets.get("FMP_API_KEY")
        if fmp_secret_key:
            # fmp_api 모듈이 성공적으로 임포트되었는지 확인 후 키 설정
            if 'fmp_api' in globals() and fmp_api:
                fmp_api.FMP_API_KEY = fmp_secret_key
                fmp_key_loaded = True
                sidebar_status.empty()
            else:
                 sidebar_status.warning("FMP 모듈 로드 실패로 Secrets 키를 설정할 수 없습니다.")
        else:
            sidebar_status.warning("Secrets에 FMP API 키가 없습니다.")
    except Exception as e:
        sidebar_status.error(f"Secrets 로드 오류: {e}"); final_status_message_displayed = True

if not fmp_key_loaded and 'fmp_api' in globals() and fmp_api: # FMP 모듈 로드 확인 추가
    sidebar_status.info("Secrets에 키 없음. .env 파일 확인 중...")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            fmp_env_key = os.getenv("FMP_API_KEY")
            if fmp_env_key:
                fmp_api.FMP_API_KEY = fmp_env_key; fmp_key_loaded = True
                sidebar_status.success("FMP API 키 로드 완료 (.env)")
                final_status_message_displayed = True
            else:
                sidebar_status.error(".env 파일 내 FMP API 키 누락 또는 로드 실패."); final_status_message_displayed = True
        else:
            if not secrets_available:
                sidebar_status.error(".env 파일 없음, Secrets에도 FMP 키 없음."); final_status_message_displayed = True
            elif fmp_key_loaded: # Should not happen if already loaded, but for completeness
                sidebar_status.empty()
    except Exception as e:
        sidebar_status.error(f".env 로드 오류: {e}"); final_status_message_displayed = True
elif not fmp_key_loaded and 'fmp_api' not in globals():
     sidebar_status.error("FMP API 모듈 로드 실패. API 키를 설정할 수 없습니다.")
     final_status_message_displayed = True

comprehensive_analysis_possible = fmp_key_loaded
if not comprehensive_analysis_possible and not final_status_message_displayed:
    st.sidebar.error("FMP API 키 로드 실패! '미국 주식 종합 분석' 기능이 제한됩니다.")
elif comprehensive_analysis_possible and not final_status_message_displayed:
    sidebar_status.success("FMP API 키 로드 완료.")


# --- DART API 키 로드 (선택적, dart.py에서 환경 변수 사용 가정) ---
# dart.py 내부에서 os.environ.get("DART_API_KEY")를 사용하므로 별도 로직 불필요.
# 단, dart_api 모듈 로드 여부 확인
dart_available = False
if 'dart_api' in globals() and dart_api:
    try:
        # dart.py 내부에 API 키 확인 로직이 있다면 호출 (없으면 DART 기능 사용 시 오류 발생)
        # 예: if dart_api.API_KEY: dart_available = True
        # dart.py의 API_KEY 상수를 직접 확인 (import dart as dart_api 가정)
        if dart_api.API_KEY:
             dart_available = True
             st.sidebar.success("DART API 키 확인됨 (환경 변수).")
        else:
             st.sidebar.warning("DART API 키가 환경 변수에 설정되지 않았습니다. '한국 기업 기본 정보' 기능이 제한될 수 있습니다.")
    except AttributeError:
         st.sidebar.warning("dart.py 에서 API_KEY 를 찾을 수 없습니다. '한국 기업 기본 정보' 기능이 제한될 수 있습니다.")
    except Exception as e:
         st.sidebar.warning(f"DART API 키 확인 중 오류: {e}")
else:
    st.sidebar.warning("dart.py 모듈 로드 실패. DART 관련 기능 사용 불가.")


# --- 사이드바 설정 ---
with st.sidebar:
    st.title("📊 주식 분석 도구")
    selected_country = st.radio("국가 선택", ["🇺🇸 미국 주식", "🇰🇷 한국 주식"], key="country_selector")
    st.markdown("---")

    page = None # page 변수 초기화
    if selected_country == "🇺🇸 미국 주식":
        page = st.radio("분석 유형 선택 (미국)", ["📊 종합 분석 (FMP)", "📈 기술 분석 (FMP)"],
                        captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"], key="page_selector_us")
        st.markdown("---")
        if page == "📊 종합 분석 (FMP)":
            st.header("⚙️ 종합 분석 설정 (미국)")
            ticker_input_us = st.text_input("종목 티커 (미국)", "AAPL", key="main_ticker_us", help="예: AAPL", disabled=not comprehensive_analysis_possible)
            analysis_years_us = st.select_slider("분석 기간 (년)", [1, 2, 3, 5, 7, 10], 2, key="analysis_years_us", disabled=not comprehensive_analysis_possible)
            forecast_days_us = st.number_input("예측 기간 (일)", 7, 90, 30, 7, key="forecast_days_us", disabled=not comprehensive_analysis_possible)
            num_trend_periods_us = st.number_input("재무 추세 분기 수", 2, 12, 4, 1, key="num_trend_periods_us", disabled=not comprehensive_analysis_possible)
            st.divider()
            st.subheader("⚙️ 예측 세부 설정 (선택)")
            changepoint_prior_us = st.slider("추세 변화 민감도 (Prophet)", 0.001, 0.5, 0.05, 0.01, "%.3f", key="changepoint_prior_us", disabled=not comprehensive_analysis_possible)
            st.divider()
            st.subheader("💰 보유 정보 입력 (선택)")
            avg_price_us = st.number_input("평단가", 0.0, format="%.2f", key="avg_price_us", disabled=not comprehensive_analysis_possible)
            quantity_us = st.number_input("보유 수량", 0, step=1, key="quantity_us", disabled=not comprehensive_analysis_possible)
            st.divider()

        elif page == "📈 기술 분석 (FMP)":
            st.header("⚙️ 기술 분석 설정 (미국)")
            ticker_tech_us = st.text_input("종목 티커 (미국)", "AAPL", key="tech_ticker_input_us", help="예: AAPL")
            bb_window_us = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window_us")
            bb_std_us = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std_us", format="%.1f")
            st.divider()
            today_us = datetime.now().date()
            default_start_us = today_us - relativedelta(months=3)
            min_date_us = today_us - relativedelta(years=5)
            start_date_us = st.date_input("시작일", default_start_us, key="tech_start_input_us", min_value=min_date_us, max_value=today_us - timedelta(days=1))
            end_date_us = st.date_input("종료일", today_us, key="tech_end_input_us", min_value=start_date_us + timedelta(days=1), max_value=today_us)
            available_intervals_tech_fmp = {"일봉": "1day", "1시간": "1hour", "15분": "15min"}
            interval_display_us = st.selectbox("데이터 간격", list(available_intervals_tech_fmp.keys()), key="tech_interval_display_input_us", help="FMP 플랜 따라 지원 간격 및 기간 다름")
            st.divider()

    elif selected_country == "🇰🇷 한국 주식":
        page = st.radio("분석 유형 선택 (한국)", ["📈 기술 분석 (pykrx)", "📝 기본 정보 (DART)"],
                        captions=["차트 및 추세선", "기업 개요 및 공시정보"], key="page_selector_kr",
                        help="기술 분석은 pykrx, 기본 정보는 DART API를 사용합니다.")
        st.markdown("---")
        if page == "📈 기술 분석 (pykrx)":
            st.header("⚙️ 기술 분석 설정 (한국)")
            ticker_tech_kr = st.text_input("종목명 또는 종목코드 (한국)", "삼성전자", key="tech_ticker_input_kr", help="예: 삼성전자 또는 005930")
            bb_window_kr = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window_kr")
            bb_std_kr = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std_kr", format="%.1f")
            st.divider()
            today_kr = datetime.now().date()
            default_start_kr = today_kr - relativedelta(months=6) # 기본 6개월 조회
            min_date_kr = today_kr - relativedelta(years=10) # pykrx는 비교적 긴 기간 제공
            start_date_kr = st.date_input("시작일 (한국)", default_start_kr, key="tech_start_input_kr", min_value=min_date_kr, max_value=today_kr - timedelta(days=1))
            end_date_kr = st.date_input("종료일 (한국)", today_kr, key="tech_end_input_kr", min_value=start_date_kr + timedelta(days=1), max_value=today_kr)
            st.caption("pykrx는 일별 데이터를 제공합니다.")
            st.divider()

        elif page == "📝 기본 정보 (DART)":
            st.header("⚙️ 기본 정보 설정 (한국)")
            company_kr_info = st.text_input("기업명 또는 종목코드 (한국)", "삼성전자", key="company_info_input_kr", disabled=not dart_available)
            # DART 정보 조회용 날짜 범위 (예: 최근 1년)
            today_dart = datetime.now().date()
            default_start_dart = today_dart - relativedelta(years=1)
            start_date_dart = st.date_input("공시 검색 시작일", default_start_dart, key="dart_start_input", max_value=today_dart - timedelta(days=1), disabled=not dart_available)
            end_date_dart = st.date_input("공시 검색 종료일", today_dart, key="dart_end_input", min_value=start_date_dart + timedelta(days=1), max_value=today_dart, disabled=not dart_available)
            st.divider()


# --- 캐시된 종합 분석 함수 (기존 코드 유지) ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, years, days, num_trend_periods, changepoint_prior_scale):
    # ... (기존 run_cached_analysis 함수 내용) ...
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    try:
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
if page: # page가 None이 아닐 때만 실행 (초기 로딩 시 방지)
    if selected_country == "🇺🇸 미국 주식":
        if page == "📊 종합 분석 (FMP)":
            st.title("🇺🇸 미국 주식 종합 분석 (FMP API 기반)")
            st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공.")
            st.markdown("---")
            analyze_button_main_disabled = not comprehensive_analysis_possible
            if analyze_button_main_disabled: st.error("FMP API 키 로드 실패. 종합 분석 불가.")

            # 사이드바에서 설정된 값 가져오기 (키 일관성 확인!)
            ticker_us = st.session_state.get('main_ticker_us', "AAPL")
            analysis_years_us = st.session_state.get('analysis_years_us', 2)
            forecast_days_us = st.session_state.get('forecast_days_us', 30)
            num_trend_periods_us = st.session_state.get('num_trend_periods_us', 4)
            changepoint_prior_us = st.session_state.get('changepoint_prior_us', 0.05)
            avg_price_us = st.session_state.get('avg_price_us', 0.0)
            quantity_us = st.session_state.get('quantity_us', 0)

            analyze_button_main_us = st.button("🚀 미국 주식 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_us_button", disabled=analyze_button_main_disabled)
            results_placeholder_us = st.container()

            if analyze_button_main_us:
                if not ticker_us: results_placeholder_us.warning("종목 티커 입력 필요.")
                else:
                    ticker_proc_us = ticker_us.strip().upper()
                    with st.spinner(f"{ticker_proc_us} 종합 분석 중..."):
                        try:
                            results = run_cached_analysis(ticker_proc_us, analysis_years_us, forecast_days_us, num_trend_periods_us, changepoint_prior_us)
                            if results and isinstance(results, dict):
                                if "error" in results:
                                    results_placeholder_us.error(f"분석 실패: {results['error']}")
                                else:
                                    results_placeholder_us.empty()
                                    with results_placeholder_us:
                                        # --- 미국 종합 분석 결과 표시 ---
                                        # 이 부분은 제공해주신 기존 코드의 결과 표시 로직을 그대로 가져옵니다.
                                        # (st.header, st.subheader, st.metric, st.columns, st.tabs, st.plotly_chart, st.expander 등)
                                        # 변수명만 위에서 정의한 _us 접미사가 붙은 변수들 (ticker_proc_us, changepoint_prior_us, avg_price_us 등)로 사용해야 합니다.
                                        # 예시 시작:
                                        st.header(f"📈 {ticker_proc_us} 분석 결과 (민감도: {changepoint_prior_us:.3f})")
                                        st.subheader("요약 정보")
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("현재가", f"${results.get('current_price', 'N/A')}") # FMP는 달러
                                        col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                                        col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))
                                        # ... [기존 미국 종합 분석 결과 표시 로직 전체 삽입] ...
                                        # ... (Fundamentals, 재무 추세, 차트, 뉴스, F&G, 예측, 리스크 트래커, 요약 등)
                                        # 리스크 트래커 부분에서 avg_p 대신 avg_price_us, qty 대신 quantity_us 사용
                                        # 예시 끝.

                            elif results is None: results_placeholder_us.error("분석 결과 처리 중 오류 발생 (결과 없음).")
                            else: results_placeholder_us.error("분석 결과 처리 중 오류 발생 (결과 형식 오류).")
                        except Exception as e:
                            error_traceback = traceback.format_exc()
                            logging.error(f"미국 종합 분석 메인 로직 실행 오류: {e}\n{error_traceback}")
                            results_placeholder_us.error(f"앱 실행 중 오류 발생: {e}")
            else:
                if comprehensive_analysis_possible:
                    results_placeholder_us.info("⬅️ 사이드바에서 설정을 확인하고 '미국 주식 종합 분석 시작!' 버튼을 클릭하세요.")
                else:
                    results_placeholder_us.warning("API 키 로드 실패로 종합 분석을 진행할 수 없습니다.")


        elif page == "📈 기술 분석 (FMP)":
            st.title("🇺🇸 미국 주식 기술적 분석 (FMP API)")
            st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준을 함께 시각화하고 자동 해석을 제공합니다.")
            st.markdown("---")

            # 사이드바에서 설정된 값 가져오기 (키 일관성 확인!)
            ticker_tech_us = st.session_state.get('tech_ticker_input_us', "AAPL")
            start_date_us = st.session_state.get('tech_start_input_us', datetime.now().date() - relativedelta(months=3))
            end_date_us = st.session_state.get('tech_end_input_us', datetime.now().date())
            available_intervals_tech_fmp = {"일봉": "1day", "1시간": "1hour", "15분": "15min"}
            interval_display_us = st.session_state.get('tech_interval_display_input_us', "일봉")
            interval_us = available_intervals_tech_fmp.get(interval_display_us, "1day")
            bb_window_us = st.session_state.get('bb_window_us', 20)
            bb_std_us = st.session_state.get('bb_std_us', 2.0)

            analyze_button_tech_us = st.button("📊 미국 주식 기술적 분석 실행", key="tech_analyze_us_button", use_container_width=True, type="primary")
            results_placeholder_tech_us = st.container()

            if analyze_button_tech_us:
                if not ticker_tech_us: results_placeholder_tech_us.warning("종목 티커를 입력해주세요.")
                else:
                    ticker_processed_us = ticker_tech_us.strip().upper()
                    with results_placeholder_tech_us:
                        st.write(f"**{ticker_processed_us}** ({interval_display_us}, BB:{bb_window_us}일/{bb_std_us:.1f}σ) 분석 중 (FMP API 사용)...")
                        with st.spinner(f"{ticker_processed_us} 데이터 로딩 및 처리 중 (FMP)..."):
                            try:
                                # --- 미국 기술 분석 실행 로직 ---
                                start_date_str_us = start_date_us.strftime("%Y-%m-%d")
                                end_date_str_us = end_date_us.strftime("%Y-%m-%d")
                                fmp_data_us = None
                                rename_map_us = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}

                                if interval_us == "1day":
                                    fmp_data_us = fmp_api.get_price_data(ticker=ticker_processed_us, start_date=start_date_str_us, end_date=end_date_str_us)
                                else:
                                    fmp_data_us = fmp_api.get_intraday_data(ticker=ticker_processed_us, interval=interval_us, from_date=start_date_str_us, to_date=end_date_str_us)

                                df_tech_us = pd.DataFrame()
                                if fmp_data_us and isinstance(fmp_data_us, list) and len(fmp_data_us) > 0:
                                    df_tech_us = pd.DataFrame(fmp_data_us)
                                    if not df_tech_us.empty:
                                        df_tech_us = df_tech_us.rename(columns=rename_map_us)
                                        date_col_name_us = rename_map_us.get('date', 'Date')
                                        if date_col_name_us in df_tech_us.columns:
                                            df_tech_us[date_col_name_us] = pd.to_datetime(df_tech_us[date_col_name_us], errors='coerce')
                                            df_tech_us = df_tech_us.set_index(date_col_name_us).sort_index()
                                            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                                if col in df_tech_us.columns:
                                                    df_tech_us[col] = pd.to_numeric(df_tech_us[col], errors='coerce')
                                            df_tech_us.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                                        else:
                                            st.error(f"FMP 응답 날짜 컬럼 '{date_col_name_us}' 없음.")
                                            df_tech_us = pd.DataFrame()
                                elif fmp_data_us is None:
                                    st.error(f"FMP 데이터 로딩 오류 (API 결과 None).")
                                else: # Empty list
                                    st.warning(f"FMP 데이터 '{ticker_processed_us}' ({interval_display_us}) 가져오기 실패 (API 결과 빈 리스트).")

                                if df_tech_us.empty:
                                    if not st.session_state.get('error_shown_tech_us', False):
                                        st.error(f"❌ 데이터 조회/처리 실패.")
                                        st.session_state['error_shown_tech_us'] = True
                                else:
                                    st.session_state['error_shown_tech_us'] = False
                                    logging.info(f"FMP 데이터 처리 완료 ({ticker_processed_us}, {interval_display_us}). {len(df_tech_us)} 행.")
                                    st.caption(f"조회 기간 (FMP): {df_tech_us.index.min()} ~ {df_tech_us.index.max()}")

                                    required_cols_us = ['Open', 'High', 'Low', 'Close']
                                    missing_cols_us = [col for col in required_cols_us if col not in df_tech_us.columns]
                                    if missing_cols_us:
                                        st.error(f"❌ 기술적 분석 위한 필수 컬럼 누락: {missing_cols_us}.")
                                        st.dataframe(df_tech_us.head())
                                    else:
                                        df_calculated_us = df_tech_us.copy()
                                        df_calculated_us.attrs['ticker'] = ticker_processed_us
                                        try: df_calculated_us = calculate_vwap(df_calculated_us)
                                        except Exception as e: st.warning(f"VWAP 계산 오류: {e}", icon="⚠️")
                                        try: df_calculated_us = calculate_bollinger_bands(df_calculated_us, window=bb_window_us, num_std=bb_std_us)
                                        except Exception as e: st.warning(f"BB 계산 오류: {e}", icon="⚠️")
                                        try: df_calculated_us = calculate_rsi(df_calculated_us)
                                        except NameError: st.error("오류: 'calculate_rsi' 함수를 찾을 수 없습니다.")
                                        except Exception as e: st.warning(f"RSI 계산 오류: {e}", icon="⚠️")
                                        try: df_calculated_us = calculate_macd(df_calculated_us)
                                        except NameError: st.error("오류: 'calculate_macd' 함수를 찾을 수 없습니다.")
                                        except Exception as e: st.warning(f"MACD 계산 오류: {e}", icon="⚠️")

                                        st.subheader(f"📌 {ticker_processed_us} 기술적 분석 통합 차트 ({interval_display_us})")
                                        chart_tech_us = plot_technical_chart(df_calculated_us, ticker_processed_us) # 미국용 차트 함수 호출
                                        if chart_tech_us and chart_tech_us.data: st.plotly_chart(chart_tech_us, use_container_width=True)
                                        else: st.warning("차트 생성 실패/표시 데이터 없음.")

                                        st.subheader("📄 최근 데이터 (계산된 지표 포함)")
                                        display_cols_us = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
                                        display_cols_exist_us = [col for col in display_cols_us if col in df_calculated_us.columns]
                                        format_dict_us = {col: "${:.2f}" for col in display_cols_exist_us if col not in ['Volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']}
                                        if 'Volume' in display_cols_exist_us: format_dict_us['Volume'] = "{:,.0f}"
                                        for col_macd in ['RSI', 'MACD', 'MACD_signal', 'MACD_hist']:
                                             if col_macd in display_cols_exist_us: format_dict_us[col_macd] = "{:.2f}"
                                        st.dataframe(df_calculated_us[display_cols_exist_us].tail(10).style.format(format_dict_us), use_container_width=True)

                                        st.divider()
                                        st.subheader("🧠 기술적 시그널 해석 (참고용)")
                                        if not df_calculated_us.empty:
                                            latest_row_us = df_calculated_us.iloc[-1].copy()
                                            signal_messages_us = []
                                            try:
                                                if 'interpret_technical_signals' in globals():
                                                    signal_messages_us.extend(interpret_technical_signals(latest_row_us, df_context=df_calculated_us))
                                                else:
                                                    st.error("오류: 'interpret_technical_signals' 함수를 찾을 수 없습니다.")
                                            except Exception as e: st.warning(f"기본 기술적 시그널 해석 오류: {e}", icon="⚠️")

                                            if signal_messages_us:
                                                for msg in signal_messages_us: st.info(msg)
                                            else: st.info("현재 특별히 감지된 기술적 시그널은 없습니다.")
                                            st.caption("⚠️ **주의:** 자동 해석은 보조 지표이며 투자 결정은 반드시 종합적인 판단 하에 신중하게 내리시기 바랍니다.")
                                        else: st.warning("해석할 데이터가 부족합니다.")
                            except requests.exceptions.RequestException as req_err:
                                st.error(f"FMP API 요청 실패: {req_err}")
                            except EnvironmentError as env_err:
                                st.error(f"FMP API 키 설정 오류: {env_err}")
                            except Exception as e:
                                st.error(f"미국 기술적 분석 처리 중 오류 발생: {type(e).__name__} - {e}")
                                logging.error(f"미국 기술 분석 탭 오류: {traceback.format_exc()}")
            else:
                results_placeholder_tech_us.info("⬅️ 사이드바에서 설정을 확인하고 '미국 주식 기술적 분석 실행' 버튼을 클릭하세요.")

    elif selected_country == "🇰🇷 한국 주식":
        if page == "📈 기술 분석 (pykrx)":
            st.title("🇰🇷 한국 주식 기술적 분석 (pykrx 활용)")
            st.markdown("`pykrx` 라이브러리를 사용하여 한국 주식 차트 및 기술적 지표를 제공합니다.")
            st.markdown("---")
            results_placeholder_kr_tech = st.container()

            ticker_input_kr = st.session_state.get('tech_ticker_input_kr', "삼성전자")
            start_date_kr = st.session_state.get('tech_start_input_kr', datetime.now().date() - relativedelta(months=6))
            end_date_kr = st.session_state.get('tech_end_input_kr', datetime.now().date())
            bb_window_val_kr = st.session_state.get('bb_window_kr', 20)
            bb_std_val_kr = st.session_state.get('bb_std_kr', 2.0)

            analyze_button_tech_kr = st.button("📊 한국 주식 기술적 분석 실행", key="tech_analyze_kr_button", use_container_width=True, type="primary")

            if analyze_button_tech_kr:
                if not ticker_input_kr:
                    results_placeholder_kr_tech.warning("종목명 또는 종목코드를 입력해주세요.")
                else:
                    # 비동기 함수 호출 (Streamlit 환경에서 asyncio 사용)
                    # Streamlit은 기본적으로 동기적으로 작동하므로, 최상위 레벨에서 await 사용 불가.
                    # asyncio.run() 또는 이벤트 루프를 직접 관리해야 함.
                    # 여기서는 간단하게 하기 위해 get_ticker_from_input을 동기 함수로 가정하거나,
                    # Streamlit의 비동기 지원 기능을 활용해야 함 (st.experimental_rerun 등).
                    # 가장 간단한 방법은 get_ticker_from_input에서 await을 제거하고 동기적으로 실행하는 것.
                    # pykrx 함수들은 대부분 동기 함수이므로 문제가 없을 수 있음.
                    # get_ticker_from_input 함수 정의에서 async와 await 제거 필요.
                    # 아래는 동기 함수로 가정하고 호출.
                    # ticker_code, company_name = get_ticker_from_input(ticker_input_kr)
                    # display_korean_stock_chart 함수도 async 제거 필요.

                    # --- Streamlit에서 비동기 함수 호출 (권장 방식) ---
                    # asyncio.run()은 Streamlit 스크립트 내에서 직접 사용 시 문제 발생 가능성 있음.
                    # Streamlit 버전 1.17.0 이상에서는 st.experimental_singleton 또는 st.cache_data 등과
                    # 함께 사용하여 비동기 함수 결과를 캐시하는 방식으로 간접 실행 가능.
                    # 하지만 display 함수처럼 UI를 직접 그리는 함수는 비동기로 만들기 어려움.
                    # 여기서는 display_korean_stock_chart 함수 자체를 호출하는 방식으로 유지하고,
                    # 해당 함수 내부에서 필요한 비동기 작업(향후 추가될 수 있는)을 처리하도록 설계하는 것이 나음.
                    # 현재 pykrx 기반 로직은 동기적이므로 async/await 없이 바로 호출 가능.
                    # display_korean_stock_chart 함수 정의에서도 async 제거 필요.

                    # display_korean_stock_chart 함수 호출 (async 제거했다고 가정)
                    display_korean_stock_chart(
                        ticker_input_kr, start_date_kr, end_date_kr,
                        bb_window_val_kr, bb_std_val_kr, results_placeholder_kr_tech
                    )

            else:
                results_placeholder_kr_tech.info("⬅️ 사이드바에서 설정을 확인하고 '한국 주식 기술적 분석 실행' 버튼을 클릭하세요.")


        elif page == "📝 기본 정보 (DART)":
            st.title("🇰🇷 한국 기업 기본 정보 (DART API)")
            st.markdown("DART API를 활용하여 한국 기업의 공시 정보 등을 조회합니다.")
            st.markdown("---")
            results_placeholder_kr_info = st.container()

            company_kr_info = st.session_state.get('company_info_input_kr', "삼성전자")
            start_date_dart = st.session_state.get('dart_start_input', datetime.now().date() - relativedelta(years=1))
            end_date_dart = st.session_state.get('dart_end_input', datetime.now().date())

            analyze_button_info_kr = st.button("🔍 한국 기업 정보 조회", key="info_analyze_kr_button", use_container_width=True, type="primary", disabled=not dart_available)

            if analyze_button_info_kr:
                if not company_kr_info:
                    results_placeholder_kr_info.warning("기업명 또는 종목코드를 입력해주세요.")
                elif not dart_available:
                     results_placeholder_kr_info.error("DART API 키가 설정되지 않아 조회할 수 없습니다.")
                else:
                    with results_placeholder_kr_info:
                        st.info(f"{company_kr_info} 기업 정보를 DART에서 조회합니다...")
                        with st.spinner("DART 정보 조회 중..."):
                            try:
                                # dart_api 모듈의 함수 호출 (dart.py에 비동기 함수가 있다면 await 필요)
                                # dart.py의 함수들이 비동기(async def)로 정의되어 있다면,
                                # 여기서 await을 사용해야 함. Streamlit 환경에서의 비동기 호출 주의.
                                # 예시: corp_code, matched_name = await dart_api.get_corp_code_by_name(company_kr_info)

                                # dart.py 함수가 동기 함수라고 가정하고 호출 예시:
                                # corp_code, matched_name = dart_api.get_corp_code_by_name(company_kr_info) # await 제거

                                # --- dart.py 비동기 함수 호출 (asyncio 사용 예시 - Streamlit 버전 확인 필요) ---
                                async def run_dart_tasks():
                                    corp_code, matched_name = await dart_api.get_corp_code_by_name(company_kr_info)
                                    if not corp_code:
                                        st.error(f"DART에서 '{company_kr_info}'에 해당하는 기업을 찾을 수 없습니다: {matched_name}")
                                        return

                                    st.success(f"기업 정보 확인: {matched_name} (고유번호: {corp_code})")

                                    start_str = start_date_dart.strftime("%Y%m%d")
                                    end_str = end_date_dart.strftime("%Y%m%d")
                                    disclosures, error_msg = await dart_api.get_disclosure_list(corp_code, start_str, end_str)

                                    if error_msg:
                                        st.error(f"공시 목록 조회 오류: {error_msg}")
                                        return

                                    if not disclosures:
                                        st.warning(f"{start_str}~{end_str} 기간 동안 {matched_name}의 정기 공시가 없습니다.")
                                        return

                                    st.subheader(f"최근 정기 공시 목록 ({len(disclosures)}건)")
                                    # 공시 정보를 DataFrame으로 만들어 표시
                                    df_disclosures = pd.DataFrame(disclosures)
                                    # 필요한 컬럼만 선택 및 이름 변경
                                    df_display = df_disclosures[['rcept_dt', 'report_nm', 'corp_name', 'flr_nm']].rename(
                                        columns={'rcept_dt': '접수일', 'report_nm': '보고서명', 'corp_name': '회사명', 'flr_nm': '제출인'}
                                    )
                                    st.dataframe(df_display, use_container_width=True)

                                    # 추가 정보 표시 (예: 가장 최근 사업보고서의 '사업의 개요' 추출)
                                    latest_business_report = None
                                    for disc in disclosures:
                                         if "사업보고서" in disc.get('report_nm', ''):
                                              latest_business_report = disc
                                              break

                                    if latest_business_report:
                                         rcept_no = latest_business_report.get('rcept_no')
                                         if rcept_no:
                                              st.subheader("최근 사업보고서 개요 (DART)")
                                              with st.spinner("'사업의 개요' 추출 중..."):
                                                   # dart.py 함수가 비동기면 await 사용
                                                   overview_text = await dart_api.extract_business_section_from_dart(rcept_no, '사업의 개요')
                                                   if overview_text.startswith("공시서류 다운로드 실패") or overview_text.startswith("'사업의 개요' 섹션을 찾을 수 없습니다"):
                                                        st.warning(f"사업 개요 추출 실패: {overview_text}")
                                                   elif overview_text.startswith("정보 추출 중 오류 발생"):
                                                         st.error(f"사업 개요 추출 오류: {overview_text}")
                                                   else:
                                                         st.text_area("사업의 개요 내용", overview_text, height=300)
                                    else:
                                         st.info("최근 공시 목록에 사업보고서가 없어 개요를 추출하지 못했습니다.")


                                # Streamlit에서 비동기 함수 실행 (try-except로 감싸기)
                                try:
                                    # get_event_loop()는 Deprecated 될 수 있으므로 get_running_loop() 시도
                                    loop = asyncio.get_running_loop()
                                    loop.run_until_complete(run_dart_tasks())
                                except RuntimeError: # No running event loop
                                    # 새로운 이벤트 루프 생성 및 실행 (Streamlit 환경에서 권장되지는 않음)
                                    # asyncio.run(run_dart_tasks()) # 이게 더 간단할 수 있음
                                    # 또는 nest_asyncio 라이브러리 사용 고려
                                    st.warning("비동기 작업 실행에 문제가 발생했습니다. (Event Loop)")
                                    # 동기적 대안 시도 (dart.py 함수가 동기/비동기 모두 지원하거나, 동기 버전이 있다면)
                                    # corp_code_sync, matched_name_sync = dart_api.get_corp_code_by_name_sync(company_kr_info) ...
                            except AttributeError as attr_err:
                                 st.error(f"DART 모듈 함수 호출 오류: {attr_err}. dart.py 파일의 함수 정의를 확인하세요.")
                            except Exception as dart_e:
                                st.error(f"DART 정보 조회 중 오류 발생: {dart_e}")
                                st.error(f"Traceback: {traceback.format_exc()}")
            else:
                results_placeholder_kr_info.info("⬅️ 사이드바에서 기업명을 입력하고 '한국 기업 정보 조회' 버튼을 클릭하세요.")

else:
    # 앱 초기 로딩 시 또는 페이지 선택 전 표시할 내용 (선택 사항)
    st.info("⬅️ 사이드바에서 국가와 분석 유형을 선택하세요.")


# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 (FMP & pykrx/DART) | 정보 제공 목적") # 제공 API 명시
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")