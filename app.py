# -*- coding: utf-8 -*-
# app.py (pykrx 적용 및 수정 완료 버전)

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
import asyncio # DART 비동기 함수 호출 위함

# --- Streamlit 페이지 설정 (가장 먼저 호출!) ---
st.set_page_config(page_title="종합 주식 분석 (FMP & pykrx/DART)", layout="wide", initial_sidebar_state="expanded")

# --- 기본 경로 설정 및 로깅 ---
# logging 설정은 페이지 설정 다음에 위치해도 괜찮음
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()

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
    # dart.py가 성공적으로 임포트되면, 실제 dart.py 내부의 API_KEY 변수는 사용하지 않고,
    # app.py에서 직접 st.secrets를 확인하여 dart_available을 설정합니다.
except ImportError:
    st.warning("dart.py 파일을 찾을 수 없습니다. '한국 기업 기본 정보' 기능이 제한될 수 있습니다.")
    DART_API = None # dart_api 객체를 None으로 설정하여 오류 방지

# --- 기술 분석 지표 계산 함수들 (수정됨) ---
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
        # st.warning은 Streamlit 요소이므로 함수 내부보다 호출하는 곳에서 사용하는 것이 좋음
        logging.warning(f"BB 계산 실패: '{required_col}' 컬럼 없음.")
        df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan; return df

    df[required_col] = pd.to_numeric(df[required_col], errors='coerce') # 숫자로 변환
    if df[required_col].isnull().all():
        logging.warning(f"BB 계산 실패: '{required_col}' 데이터 없음.")
        df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan; return df

    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        logging.warning(f"BB 계산 위한 유효 데이터({len(valid_close)}개)가 기간({window}개)보다 부족.")
        df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan; return df
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
    # 함수 시작 시점에 데이터 타입 변환 및 NaN 처리
    for col in required_candle_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
             logging.error(f"미국 차트: 필수 컬럼 '{col}' 없음"); return go.Figure() # 빈 차트 반환
    df.dropna(subset=required_candle_cols, inplace=True)
    if df.empty:
        logging.error("미국 차트: 유효한 OHLC 데이터 없음"); return go.Figure()

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))
    # VWAP
    if 'VWAP' in df.columns and df['VWAP'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns: logging.info(f"{ticker}: VWAP 데이터 없음/표시 불가.")
    # Bollinger Bands
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)'))
    elif 'Upper' in df.columns: logging.info(f"{ticker}: 볼린저 밴드 데이터 없음/표시 불가.")
    # Fibonacci
    valid_price_df = df.dropna(subset=['High', 'Low'])
    if not valid_price_df.empty:
        min_price = valid_price_df['Low'].min(); max_price = valid_price_df['High'].max(); diff = max_price - min_price
        if diff > 0:
            levels = {'0.0 (High)': max_price, '0.236': max_price - 0.236 * diff, '0.382': max_price - 0.382 * diff, '0.5': max_price - 0.5 * diff, '0.618': max_price - 0.618 * diff, '1.0 (Low)': min_price}
            fib_colors = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels.items(): fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: ${v:.2f}", line_color=fib_colors.get(k, 'navy'), annotation_position="bottom right", annotation_font_size=10)
        else: logging.info(f"{ticker}: 기간 내 가격 변동 없어 피보나치 미표시.")
    else: logging.info(f"{ticker}: 피보나치 레벨 계산 불가.")
    # RSI
    if 'RSI' in df.columns and df['RSI'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='purple', width=1), yaxis='y2'))
    # MACD
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
    # 함수 시작 시점에 데이터 타입 변환 및 NaN 처리
    for col in required_candle_cols:
         if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
         else:
             logging.error(f"한국 차트 ({company_name}): 필수 컬럼 '{col}' 없음"); return go.Figure()
    df.dropna(subset=required_candle_cols, inplace=True)
    if df.empty:
        logging.error(f"한국 차트 ({company_name}): 유효한 OHLC 데이터 없음"); return go.Figure()

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{company_name} 캔들"))

    # VWAP
    if 'VWAP' in df.columns and df['VWAP'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns: logging.info(f"{company_name}: VWAP 데이터 없음/표시 불가.")

    # Bollinger Bands
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20 (BB 중심)', line=dict(color='blue', width=1, dash='dash'))) # 이름 수정
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)'))
    elif 'Upper' in df.columns: logging.info(f"{company_name}: 볼린저 밴드 데이터 없음/표시 불가.")

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
        else: logging.info(f"{company_name}: 기간 내 가격 변동 없어 피보나치 미표시.")
    else: logging.info(f"{company_name}: 피보나치 레벨 계산 불가.")

    # 추세선 (이동평균선)
    if 'SMA5' in df.columns and df['SMA5'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'], mode='lines', name='SMA 5일', line=dict(color='green', width=1)))
    if 'SMA20' in df.columns and df['SMA20'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20일', line=dict(color='red', width=1)))
    if 'SMA60' in df.columns and df['SMA60'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], mode='lines', name='SMA 60일', line=dict(color='purple', width=1)))

    # TODO: 실제 상승/하락 추세선 그리기 (선택적 고급 기능)

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

# --- pykrx 티커 조회 헬퍼 함수 (동기 버전) ---
@st.cache_data(ttl=3600)
def get_kr_ticker_map():
    """KRX KOSPI, KOSDAQ 종목 목록을 가져와 이름-티커, 티커-이름 맵을 반환합니다."""
    name_to_ticker = {}
    ticker_to_name = {}
    try:
        today_str = datetime.now().strftime("%Y%m%d")
        for market in ["KOSPI", "KOSDAQ", "KONEX"]: # 코넥스도 포함 가능
            tickers = stock.get_market_ticker_list(date=today_str, market=market)
            for ticker in tickers:
                try:
                    name = stock.get_market_ticker_name(ticker)
                    # 가끔 이름이 None으로 반환되는 경우 처리
                    if name:
                        name_to_ticker[name] = ticker
                        ticker_to_name[ticker] = name
                except Exception as e_inner:
                    logging.warning(f"Ticker {ticker} 이름 조회 실패: {e_inner}")
        logging.info(f"KRX Ticker Map 로드 완료: {len(name_to_ticker)} 종목")
        if not name_to_ticker:
             logging.warning("KRX 종목 목록을 가져왔으나 비어있습니다. 날짜나 네트워크 문제일 수 있습니다.")
        return name_to_ticker, ticker_to_name
    except Exception as e:
        logging.error(f"KRX 종목 목록 조회 중 오류 발생: {e}")
        # 앱 실행 중 오류 발생 시 사용자에게 알림 (캐싱 함수 내에서는 st 사용 주의)
        # 이 오류는 앱 로딩 시 발생할 수 있으므로, 호출하는 곳에서 처리하는 것이 더 안전
        # raise Exception(f"KRX 종목 목록 조회 실패: {e}") # 또는 빈 딕셔너리 반환 유지
        return {}, {}

# --- 사용자 입력 처리 헬퍼 함수 (동기 버전) ---
def get_ticker_from_input(user_input):
    """사용자 입력(회사명 또는 종목코드)으로부터 6자리 종목코드와 회사명을 반환합니다."""
    user_input_stripped = user_input.strip()
    name_to_ticker_map, ticker_to_name_map = get_kr_ticker_map() # 캐싱된 맵 사용

    if not name_to_ticker_map and not ticker_to_name_map:
        # get_kr_ticker_map 함수 자체에서 오류 로깅/처리를 하므로 여기서는 결과만 확인
        st.error("KRX 종목 목록을 불러올 수 없습니다. 앱을 새로고침하거나 잠시 후 다시 시도해주세요.")
        return None, user_input_stripped # 실패

    # 입력이 6자리 숫자이고, 유효한 티커인지 확인
    if user_input_stripped.isdigit() and len(user_input_stripped) == 6:
        if user_input_stripped in ticker_to_name_map:
            return user_input_stripped, ticker_to_name_map[user_input_stripped]
        else:
            st.warning(f"입력하신 종목코드 '{user_input_stripped}'는 현재 KRX 목록에 없습니다.")
            return None, user_input_stripped # 실패

    # 입력이 회사명과 정확히 일치하는지 확인
    if user_input_stripped in name_to_ticker_map:
        return name_to_ticker_map[user_input_stripped], user_input_stripped

    # 입력이 회사명의 일부를 포함하는지 확인 (대소문자 무시, 첫 번째 매칭)
    cleaned_input = user_input_stripped.lower()
    matches = []
    for name, ticker in name_to_ticker_map.items():
        if cleaned_input in name.lower():
            matches.append((name, ticker))

    if len(matches) == 1: # 정확히 하나만 매칭될 경우
        found_ticker, found_name = matches[0][1], matches[0][0]
        st.info(f"'{user_input_stripped}' -> '{found_name}'(으)로 검색합니다. (티커: {found_ticker})")
        return found_ticker, found_name
    elif len(matches) > 1: # 여러 개 매칭될 경우
        st.warning(f"'{user_input_stripped}'와(과) 유사한 이름의 종목이 여러 개 있습니다. 더 정확한 회사명을 입력해주세요.")
        st.info(f"검색된 종목 예시: {', '.join([m[0] for m in matches[:5]])}...") # 최대 5개 예시
        return None, user_input_stripped # 실패
    else: # 매칭되는 것이 없을 경우
        st.warning(f"'{user_input_stripped}'에 해당하는 종목을 찾지 못했습니다. 정확한 회사명이나 6자리 종목코드를 입력해주세요.")
        return None, user_input_stripped # 실패

# --- 한국 주식 데이터 로딩 및 차트 표시 함수 (동기 버전) ---
def display_korean_stock_chart(ticker_input_kr, start_date_kr, end_date_kr, bb_window_val_kr, bb_std_val_kr, results_container):
    """한국 주식 데이터를 pykrx로 가져와 기술적 분석 차트를 표시합니다."""
    with results_container:
        # 1. 사용자 입력으로부터 종목코드 가져오기
        ticker_code, company_name_krx = get_ticker_from_input(ticker_input_kr)

        if not ticker_code:
            return # 오류 메시지는 get_ticker_from_input에서 이미 표시됨

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

                # 4. 컬럼명 변경 및 타입 변환
                df_kr.rename(columns={'시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'}, inplace=True)
                df_kr.index = pd.to_datetime(df_kr.index)

                # 5. 기술적 지표 계산
                required_cols_kr = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols_kr = [col for col in required_cols_kr if col not in df_kr.columns]
                if missing_cols_kr:
                    st.error(f"❌ {company_name_krx} 데이터에 필수 컬럼이 부족합니다: {missing_cols_kr}")
                    st.dataframe(df_kr.head())
                    return

                df_calculated_kr = df_kr.copy()
                df_calculated_kr.attrs['ticker'] = ticker_code

                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                     df_calculated_kr[col] = pd.to_numeric(df_calculated_kr[col], errors='coerce')
                df_calculated_kr.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

                if df_calculated_kr.empty:
                     st.error("❌ 숫자 변환 후 유효한 가격 데이터가 없습니다.")
                     return

                # 지표 계산
                try: df_calculated_kr = calculate_vwap(df_calculated_kr)
                except ValueError as ve: st.warning(f"VWAP 계산 실패: {ve}", icon="⚠️")
                except Exception as e: st.warning(f"VWAP 계산 중 오류: {e}", icon="⚠️")

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
                logging.error(f"한국 주식 처리 오류 ({company_name_krx}): {traceback.format_exc()}") # 로깅 강화

# --- FMP API 키 로드 및 확인 ---
fmp_key_loaded = False
sidebar_status = st.sidebar.empty()
final_status_message_displayed = False

secrets_available = hasattr(st, 'secrets')
if secrets_available:
    try:
        fmp_secret_key = st.secrets.get("FMP_API_KEY")
        if fmp_secret_key:
            if 'fmp_api' in globals() and fmp_api:
                fmp_api.FMP_API_KEY = fmp_secret_key
                fmp_key_loaded = True
            else:
                 logging.warning("FMP 모듈 로드 실패로 Secrets 키를 설정할 수 없습니다.")
        else:
            logging.warning("Secrets에 FMP API 키가 없습니다.")
    except Exception as e:
        logging.error(f"Secrets 로드 오류: {e}"); final_status_message_displayed = True

if not fmp_key_loaded and 'fmp_api' in globals() and fmp_api:
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path):
        try:
            load_dotenv(dotenv_path=dotenv_path)
            fmp_env_key = os.getenv("FMP_API_KEY")
            if fmp_env_key:
                fmp_api.FMP_API_KEY = fmp_env_key; fmp_key_loaded = True
                logging.info("FMP API 키 로드 완료 (.env)") # sidebar_status 대신 로깅
                final_status_message_displayed = True # 상태 업데이트
            else:
                 logging.error(".env 파일 내 FMP API 키 누락 또는 로드 실패.")
                 final_status_message_displayed = True
        except Exception as e:
            logging.error(f".env 로드 오류: {e}"); final_status_message_displayed = True
    else:
         # .env 없고 secrets에도 없었으면 최종 에러 상태
         if not secrets_available:
             logging.error(".env 파일 없음, Secrets에도 FMP 키 없음.")
             final_status_message_displayed = True

comprehensive_analysis_possible = fmp_key_loaded

# --- DART API 키 확인 ---
dart_available = False
if dart_api: # dart.py 모듈이 성공적으로 임포트 되었는지 먼저 확인
    if hasattr(st, 'secrets') and "DART_API_KEY" in st.secrets and st.secrets.DART_API_KEY:
        # Streamlit Secrets에 DART_API_KEY가 있고, 그 값이 비어있지 않은 경우
        dart_available = True
        logging.info("DART API 키 확인됨 (Streamlit Secrets).")
        # 이 경우, dart.py 내부의 API_KEY 변수에도 이 값이 반영되도록 할 수 있지만,
        # dart.py 함수들이 API_KEY를 직접 참조하므로, dart.py 내의 API_KEY 로드 로직도 중요합니다.
        # dart.py의 API_KEY 로드 로직이 st.secrets을 포함하도록 이전 답변처럼 수정되어 있어야 합니다.
    elif os.getenv("DART_API_KEY"):
        # 환경 변수에 DART_API_KEY가 설정된 경우 (로컬 .env 등)
        dart_available = True
        logging.info("DART API 키 확인됨 (환경 변수).")
    else:
        logging.warning("DART API 키가 Streamlit Secrets 또는 환경 변수에 설정되지 않았습니다.")
else:
    logging.warning("dart.py 모듈 로드 실패. DART 관련 기능 사용 불가.")

# --- 사이드바 설정 ---
with st.sidebar:
    # --- API 키 상태 표시 ---
    if not comprehensive_analysis_possible:
        st.error("FMP API 키 로드 실패! 미국 종합 분석 불가.")
    else:
        st.success("FMP API 키 로드 완료.")

    if not dart_available:
        st.warning("DART API 키 설정 확인 필요. 한국 기업 정보 조회 제한.")
    else:
        st.success("DART API 키 확인 완료.")

    st.title("📊 주식 분석 도구")
    selected_country = st.radio("국가 선택", ["🇺🇸 미국 주식", "🇰🇷 한국 주식"], key="country_selector")
    st.markdown("---")

    page = None
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
            default_start_kr = today_kr - relativedelta(months=6)
            min_date_kr = today_kr - relativedelta(years=10)
            start_date_kr = st.date_input("시작일 (한국)", default_start_kr, key="tech_start_input_kr", min_value=min_date_kr, max_value=today_kr - timedelta(days=1))
            end_date_kr = st.date_input("종료일 (한국)", today_kr, key="tech_end_input_kr", min_value=start_date_kr + timedelta(days=1), max_value=today_kr)
            st.caption("pykrx는 일별 데이터를 제공합니다.")
            st.divider()

        elif page == "📝 기본 정보 (DART)":
            st.header("⚙️ 기본 정보 설정 (한국)")
            company_kr_info = st.text_input("기업명 또는 종목코드 (한국)", "삼성전자", key="company_info_input_kr", disabled=not dart_available)
            today_dart = datetime.now().date()
            default_start_dart = today_dart - relativedelta(years=1)
            start_date_dart = st.date_input("공시 검색 시작일", default_start_dart, key="dart_start_input", max_value=today_dart - timedelta(days=1), disabled=not dart_available)
            end_date_dart = st.date_input("공시 검색 종료일", today_dart, key="dart_end_input", min_value=start_date_dart + timedelta(days=1), max_value=today_dart, disabled=not dart_available)
            st.divider()

# --- 캐시된 종합 분석 함수 (기존 코드 유지) ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, years, days, num_trend_periods, changepoint_prior_scale):
    # ... (이전 run_cached_analysis 내용) ...
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    try:
        # stock_analysis.py 가 필요
        analysis_results = sa.analyze_stock(
            ticker,
            analysis_period_years=years,
            forecast_days=days,
            num_trend_periods=num_trend_periods,
            changepoint_prior_scale=changepoint_prior_scale
        )
        return analysis_results
    except NameError: # sa 모듈 로드 실패 시
         return {"error": "종합 분석 모듈(stock_analysis.py) 로딩 실패"}
    except Exception as e:
        logging.error(f"analyze_stock 함수 실행 중 오류 발생 (ticker: {ticker}): {e}\n{traceback.format_exc()}")
        return {"error": f"종합 분석 중 오류 발생: {e}"}

# --- 메인 화면 로직 ---
if page: # page가 None이 아닐 때만 실행
    if selected_country == "🇺🇸 미국 주식":
        if page == "📊 종합 분석 (FMP)":
            st.title("🇺🇸 미국 주식 종합 분석 (FMP API 기반)")
            st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공.")
            st.markdown("---")
            analyze_button_main_disabled = not comprehensive_analysis_possible
            if analyze_button_main_disabled: st.error("FMP API 키 로드 실패. 종합 분석 불가.")

            # 사이드바에서 설정된 값 가져오기
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
                    # FMP API는 종종 .KS, .KQ 접미사 없이 사용 가능
                    if '.KS' in ticker_proc_us or '.KQ' in ticker_proc_us:
                         original_ticker_us = ticker_proc_us
                         ticker_proc_us = ticker_proc_us.split('.')[0]
                         results_placeholder_us.info(f"국내 티커 감지: {original_ticker_us} -> {ticker_proc_us} (FMP용)")

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
                                        # (기존 코드의 결과 표시 로직을 여기에 삽입)
                                        # 예시:
                                        if results.get("warn_high_mape"):
                                            m = results.get("mape", "N/A")
                                            mape_value_str = m if isinstance(m, str) else (f"{m:.1f}%" if isinstance(m, (int, float)) else "N/A")
                                            st.warning(f"🔴 모델 정확도 낮음 (MAPE {mape_value_str}). 예측 신뢰도 주의!")
                                        st.header(f"📈 {ticker_proc_us} 분석 결과 (민감도: {changepoint_prior_us:.3f})")
                                        st.subheader("요약 정보")
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                                        col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                                        col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))
                                        # ... [Fundamentals, 재무 추세 탭, 차트, 뉴스, F&G, 예측, 리스크 트래커, 요약 등 전체 로직] ...
                                        # 리스크 트래커 부분에서 avg_price_us, quantity_us 사용
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
            st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준 등을 시각화하고 자동 해석합니다.")
            st.markdown("---")

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
                    # FMP API는 종종 .KS, .KQ 접미사 없이 사용 가능
                    if '.KS' in ticker_processed_us or '.KQ' in ticker_processed_us:
                         original_ticker_us = ticker_processed_us
                         ticker_processed_us = ticker_processed_us.split('.')[0]
                         results_placeholder_tech_us.info(f"국내 티커 감지: {original_ticker_us} -> {ticker_processed_us} (FMP용)")

                    with results_placeholder_tech_us:
                        st.write(f"**{ticker_processed_us}** ({interval_display_us}, BB:{bb_window_us}일/{bb_std_us:.1f}σ) 분석 중 (FMP API 사용)...")
                        with st.spinner(f"{ticker_processed_us} 데이터 로딩 및 처리 중 (FMP)..."):
                            try:
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
                                    # ... [이하 미국 기술 분석 데이터 처리 및 결과 표시 로직 전체 삽입] ...
                                    # (데이터프레임 생성, 컬럼명 변경, 인덱스 설정, 숫자 변환, 지표 계산, 차트 표시, 테이블 표시, 시그널 해석 등)

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
                    # display_korean_stock_chart 함수 호출 (동기 함수)
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
                elif not dart_api:
                     results_placeholder_kr_info.error("DART 모듈 로드 실패로 조회할 수 없습니다.")
                elif not dart_available:
                     results_placeholder_kr_info.error("DART API 키가 설정되지 않아 조회할 수 없습니다.")
                else:
                    with results_placeholder_kr_info:
                        st.info(f"{company_kr_info} 기업 정보를 DART에서 조회합니다...")
                        with st.spinner("DART 정보 조회 중..."):
                            # --- dart.py 비동기 함수 호출 ---
                            # dart.py의 함수가 async def로 정의되어 있으므로 asyncio 사용
                            async def run_dart_tasks():
                                try:
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
                                    df_disclosures = pd.DataFrame(disclosures)
                                    df_display = df_disclosures[['rcept_dt', 'report_nm', 'corp_name', 'flr_nm']].rename(
                                        columns={'rcept_dt': '접수일', 'report_nm': '보고서명', 'corp_name': '회사명', 'flr_nm': '제출인'}
                                    )
                                    st.dataframe(df_display, use_container_width=True)

                                    latest_business_report = next((d for d in disclosures if "사업보고서" in d.get('report_nm', '')), None)

                                    if latest_business_report:
                                         rcept_no = latest_business_report.get('rcept_no')
                                         if rcept_no:
                                              st.subheader("최근 사업보고서 개요 (DART)")
                                              with st.spinner("'사업의 개요' 추출 중..."):
                                                   overview_text = await dart_api.extract_business_section_from_dart(rcept_no, '사업의 개요')
                                                   if "실패" in overview_text or "찾을 수 없습니다" in overview_text or "오류 발생" in overview_text:
                                                         st.warning(f"사업 개요 추출 실패/오류: {overview_text}")
                                                   else:
                                                         st.text_area("사업의 개요 내용", overview_text, height=300)
                                    else:
                                         st.info("최근 공시 목록에 사업보고서가 없어 개요를 추출하지 못했습니다.")

                                except Exception as dart_e:
                                     st.error(f"DART 정보 조회 중 오류 발생: {dart_e}")
                                     logging.error(f"DART 정보 조회 오류: {traceback.format_exc()}")


                            # Streamlit에서 비동기 함수 실행 (asyncio.run 사용 시도, 오류 시 로깅)
                            try:
                                # Streamlit 클라우드 등 일부 환경에서는 nest_asyncio가 필요할 수 있음
                                # import nest_asyncio
                                # nest_asyncio.apply()
                                asyncio.run(run_dart_tasks())
                            except RuntimeError as re:
                                # 이미 이벤트 루프가 실행 중일 때 발생 가능
                                st.warning(f"DART 비동기 작업 실행 중 이벤트 루프 문제 발생: {re}. 결과가 표시되지 않을 수 있습니다.")
                                logging.warning(f"Asyncio RuntimeError: {re}. Trying to run tasks in existing loop if possible.")
                                # 현재 실행중인 루프에서 실행 시도 (더 복잡하고 항상 가능하지는 않음)
                                # try:
                                #    loop = asyncio.get_running_loop()
                                #    loop.create_task(run_dart_tasks()) # 백그라운드 실행 (결과 표시 타이밍 문제 가능)
                                # except RuntimeError:
                                #    st.error("기존 이벤트 루프를 찾을 수 없어 DART 작업을 실행할 수 없습니다.")
                            except Exception as e:
                                 st.error(f"DART 정보 조회 중 예측하지 못한 오류 발생: {e}")
                                 logging.error(f"DART 작업 실행 오류: {traceback.format_exc()}")
            else:
                results_placeholder_kr_info.info("⬅️ 사이드바에서 기업명을 입력하고 '한국 기업 정보 조회' 버튼을 클릭하세요.")

else:
    st.info("⬅️ 사이드바에서 국가와 분석 유형을 선택하세요.")

# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 (FMP & pykrx/DART) | 정보 제공 목적")
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")