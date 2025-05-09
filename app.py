# -*- coding: utf-8 -*-
# app.py (pykrx 적용 및 MCP 제거 가정 버전)

import streamlit as st # Streamlit 임포트가 가장 먼저 오는 것이 좋음
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import traceback
import plotly.graph_objects as go
import numpy as np
import logging # logging 임포트
import requests # FMP API 호출에 필요
import asyncio # DART 비동기 함수 호출 위함

# --- Streamlit 페이지 설정 (가장 먼저 호출!) ---
st.set_page_config(page_title="종합 주식 분석 (FMP & pykrx/DART)", layout="wide", initial_sidebar_state="expanded")

# --- 기본 경로 설정 및 로깅 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()
logging.info(f"BASE_DIR 설정: {BASE_DIR}")

# --- pykrx 임포트 ---
try:
    from pykrx import stock
    logging.info("pykrx 모듈 로드 성공")
except ImportError:
    logging.error("pykrx 라이브러리 임포트 실패. 설치 필요.")
    st.error("pykrx 라이브러리를 찾을 수 없습니다. 'pip install pykrx'로 설치해주세요.")
    st.stop()

# --- FMP API 및 분석 모듈 임포트 ---
try:
    import fmp_api # FMP API 래퍼 모듈
    import stock_analysis as sa # 종합 분석 로직
    logging.info("FMP 및 stock_analysis 모듈 로드 성공")
except ImportError as e:
    logging.error(f"FMP 또는 stock_analysis 모듈 로딩 실패: {e}")
    st.error(f"필수 API 또는 분석 모듈 로딩 실패: {e}. 'fmp_api.py'와 'stock_analysis.py' 파일이 있는지 확인하세요.")
    st.stop()

# --- 기술 분석 관련 함수 임포트 ---
try:
    from short_term_analysis import interpret_fibonacci, calculate_rsi, calculate_macd
    from technical_interpret import interpret_technical_signals
    logging.info("기술 분석 함수 (short_term_analysis, technical_interpret) 로드 성공")
except ImportError as e:
    logging.error(f"기술 분석 함수 로딩 실패: {e}")
    st.error(f"기술 분석 모듈 로딩 실패: {e}. 'short_term_analysis.py'와 'technical_interpret.py' 파일이 있는지 확인하세요.")
    # st.stop() # 앱 전체 중단보다는 경고 후 기능 제한이 나을 수 있음

# --- DART API 모듈 임포트 (한국 기업 기본 정보용) ---
dart_api_module_loaded = False
try:
    import dart as dart_api # dart.py 파일을 dart_api 라는 이름으로 임포트
    dart_api_module_loaded = True
    logging.info("dart.py 모듈 로드 성공")
except ImportError:
    logging.warning("dart.py 파일을 찾을 수 없습니다.")
    # st.warning("dart.py 파일을 찾을 수 없습니다. '한국 기업 기본 정보' 기능이 제한될 수 있습니다.") # 사이드바에서 한번 더 안내 예정
    dart_api = None # dart_api 객체를 None으로 설정하여 오류 방지
except Exception as e:
    logging.error(f"dart.py 모듈 로드 중 예외 발생: {e}")
    # st.warning(f"dart.py 모듈 로드 중 오류 발생: {e}. '한국 기업 기본 정보' 기능이 제한될 수 있습니다.")
    dart_api = None


# --- 기술 분석 지표 계산 함수들 (수정됨) ---
def calculate_vwap(df):
    df = df.copy()
    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"VWAP 계산 실패: 필수 컬럼 부족 ({[col for col in required_cols if col not in df.columns]})")
    for col in required_cols: # 숫자로 변환 시도
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_cols, inplace=True) # NaN 값 있는 행 제거
    if df.empty:
        logging.warning(f"VWAP 계산 불가: 유효한 OHLCV 데이터 없음 (after dropna)")
        df['VWAP'] = np.nan
        return df

    if (df['Volume'] == 0).all():
        df['VWAP'] = np.nan
        logging.warning(f"Ticker {df.attrs.get('ticker', '')}: VWAP 계산 불가 (모든 거래량 0)")
        return df

    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
    df['VWAP'] = np.where(df['cumulative_volume'] > 0, df['cumulative_tp_volume'] / df['cumulative_volume'], np.nan)
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    df = df.copy()
    required_col = 'Close'
    if required_col not in df.columns:
        logging.warning(f"BB 계산 실패: '{required_col}' 컬럼 없음.")
        df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan
        return df

    df[required_col] = pd.to_numeric(df[required_col], errors='coerce')
    if df[required_col].isnull().all():
        logging.warning(f"BB 계산 실패: '{required_col}' 데이터 없음.")
        df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan
        return df

    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        logging.warning(f"BB 계산 위한 유효 데이터({len(valid_close)}개)가 기간({window}개)보다 부족.")
        df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan
    else:
        df['MA20'] = df[required_col].rolling(window=window, min_periods=max(1, window)).mean() # min_periods 수정
        df['STD20'] = df[required_col].rolling(window=window, min_periods=max(1, window)).std() # min_periods 수정
        df['Upper'] = df['MA20'] + num_std * df['STD20']
        df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

# --- 차트 생성 함수 (미국 주식용) ---
def plot_technical_chart(df, ticker):
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols):
        logging.error(f"미국 차트 ({ticker}): 필수 OHLC 컬럼 부족.")
        return fig
    for col in required_candle_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_candle_cols, inplace=True)
    if df.empty:
        logging.error(f"미국 차트 ({ticker}): 유효한 OHLC 데이터 없음."); return fig

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))
    if 'VWAP' in df.columns and df['VWAP'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)'))
    valid_price_df = df.dropna(subset=['High', 'Low'])
    if not valid_price_df.empty:
        min_price = valid_price_df['Low'].min(); max_price = valid_price_df['High'].max(); diff = max_price - min_price
        if diff > 0:
            levels = {'0.0 (High)': max_price, '0.236': max_price - 0.236 * diff, '0.382': max_price - 0.382 * diff, '0.5': max_price - 0.5 * diff, '0.618': max_price - 0.618 * diff, '1.0 (Low)': min_price}
            fib_colors = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels.items(): fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: ${v:.2f}", line_color=fib_colors.get(k, 'navy'), annotation_position="bottom right", annotation_font_size=10)
    if 'RSI' in df.columns and df['RSI'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='purple', width=1), yaxis='y2'))
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines', line=dict(color='teal'), yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', mode='lines', line=dict(color='orange'), yaxis='y3'))
        if 'MACD_hist' in df.columns: colors = ['lightgreen' if v >= 0 else 'lightcoral' for v in df['MACD_hist']]; fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color=colors, yaxis='y3'))
    fig.update_layout(title=f"{ticker} - 기술적 분석 통합 차트 (미국 주식)", xaxis_title="날짜 / 시간", yaxis=dict(domain=[0.4, 1], title="가격 ($)"), yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right', showgrid=False), yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right', showgrid=False), xaxis_rangeslider_visible=False, legend_title_text="지표", hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
    return fig

# --- 한국 주식용 차트 생성 함수 ---
def plot_korean_technical_chart(df, ticker_code, company_name):
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols):
        logging.error(f"한국 차트 ({company_name}): 필수 OHLC 컬럼 부족.")
        return fig
    for col in required_candle_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_candle_cols, inplace=True)
    if df.empty:
        logging.error(f"한국 차트 ({company_name}): 유효한 OHLC 데이터 없음."); return fig

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{company_name} 캔들"))
    if 'VWAP' in df.columns and df['VWAP'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20 (BB 중심)', line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)'))
    valid_price_df_kr = df.dropna(subset=['High', 'Low'])
    if not valid_price_df_kr.empty:
        min_price_kr = valid_price_df_kr['Low'].min(); max_price_kr = valid_price_df_kr['High'].max(); diff_kr = max_price_kr - min_price_kr
        if diff_kr > 0:
            levels_kr = {'0.0 (High)': max_price_kr, '0.236': max_price_kr - 0.236 * diff_kr, '0.382': max_price_kr - 0.382 * diff_kr, '0.5': max_price_kr - 0.5 * diff_kr, '0.618': max_price_kr - 0.618 * diff_kr, '1.0 (Low)': min_price_kr}
            fib_colors_kr = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels_kr.items(): fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: {v:,.0f}원", line_color=fib_colors_kr.get(k, 'navy'), annotation_position="bottom right", annotation_font_size=10)
    if 'SMA5' in df.columns and df['SMA5'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'], mode='lines', name='SMA 5일', line=dict(color='green', width=1)))
    if 'SMA20' in df.columns and df['SMA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20일', line=dict(color='red', width=1)))
    if 'SMA60' in df.columns and df['SMA60'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], mode='lines', name='SMA 60일', line=dict(color='purple', width=1)))
    if 'RSI' in df.columns and df['RSI'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='purple', width=1), yaxis='y2'))
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines', line=dict(color='teal'), yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', mode='lines', line=dict(color='orange'), yaxis='y3'))
        if 'MACD_hist' in df.columns: colors_kr = ['lightgreen' if v >= 0 else 'lightcoral' for v in df['MACD_hist']]; fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color=colors_kr, yaxis='y3'))
    fig.update_layout(title=f"{company_name} ({ticker_code}) 기술적 분석", xaxis_title="날짜", yaxis=dict(domain=[0.4, 1], title="가격 (원)"), yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right', showgrid=False), yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right', showgrid=False), xaxis_rangeslider_visible=False, legend_title_text="지표", hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
    return fig

# --- pykrx 티커 조회 헬퍼 함수 (동기 버전) ---
@st.cache_data(ttl=3600)
def get_kr_ticker_map():
    name_to_ticker = {}
    ticker_to_name = {}
    try:
        today_str = datetime.now().strftime("%Y%m%d")
        for market_type in ["KOSPI", "KOSDAQ", "KONEX"]:
            tickers_on_date = stock.get_market_ticker_list(date=today_str, market=market_type)
            for ticker in tickers_on_date:
                try:
                    name = stock.get_market_ticker_name(ticker)
                    if name:
                        name_to_ticker[name] = ticker
                        ticker_to_name[ticker] = name
                except Exception: # 개별 티커 이름 조회 실패 시
                    pass # 그냥 넘어감
        if not name_to_ticker: logging.warning("KRX 종목 목록 로드 실패 또는 비어있음.")
        else: logging.info(f"KRX Ticker Map 로드 완료: {len(name_to_ticker)} 종목")
    except Exception as e:
        logging.error(f"KRX 전체 종목 목록 조회 중 오류: {e}")
    return name_to_ticker, ticker_to_name

# --- 사용자 입력 처리 헬퍼 함수 (동기 버전) ---
def get_ticker_from_input(user_input):
    user_input_stripped = user_input.strip()
    name_to_ticker_map, ticker_to_name_map = get_kr_ticker_map()

    if not name_to_ticker_map and not ticker_to_name_map:
        st.error("KRX 종목 목록을 불러올 수 없어 티커 변환이 불가능합니다.")
        return None, user_input_stripped

    if user_input_stripped.isdigit() and len(user_input_stripped) == 6:
        if user_input_stripped in ticker_to_name_map:
            return user_input_stripped, ticker_to_name_map[user_input_stripped]
        else:
            st.warning(f"입력하신 종목코드 '{user_input_stripped}'는 현재 KRX 목록에 없습니다.")
            return None, user_input_stripped
    if user_input_stripped in name_to_ticker_map:
        return name_to_ticker_map[user_input_stripped], user_input_stripped
    cleaned_input = user_input_stripped.lower()
    matches = [(name, ticker) for name, ticker in name_to_ticker_map.items() if cleaned_input in name.lower()]
    if len(matches) == 1:
        found_name, found_ticker = matches[0]
        st.info(f"'{user_input_stripped}' -> '{found_name}'(으)로 검색합니다. (티커: {found_ticker})")
        return found_ticker, found_name
    elif len(matches) > 1:
        st.warning(f"'{user_input_stripped}'와(과) 유사한 이름의 종목이 여러 개 있습니다. 더 정확한 회사명을 입력해주세요.")
        st.info(f"검색된 종목 예시: {', '.join([m[0] for m in matches[:3]])}...")
        return None, user_input_stripped
    else:
        st.warning(f"'{user_input_stripped}'에 해당하는 종목을 찾지 못했습니다.")
        return None, user_input_stripped

# --- 한국 주식 데이터 로딩 및 차트 표시 함수 (동기 버전) ---
def display_korean_stock_chart(ticker_input_kr, start_date_kr, end_date_kr, bb_window_val_kr, bb_std_val_kr, results_container):
    with results_container:
        ticker_code, company_name_krx = get_ticker_from_input(ticker_input_kr)
        if not ticker_code: return

        start_date_str_krx = start_date_kr.strftime("%Y%m%d")
        end_date_str_krx = end_date_kr.strftime("%Y%m%d")

        st.write(f"**{company_name_krx} ({ticker_code})** 기술적 분석 (BB: {bb_window_val_kr}일/{bb_std_val_kr:.1f}σ)")
        with st.spinner(f"{company_name_krx} ({ticker_code}) 데이터 로딩 및 차트 생성 중... (pykrx)"):
            try:
                df_kr = stock.get_market_ohlcv(start_date_str_krx, end_date_str_krx, ticker_code)
                if df_kr.empty:
                    st.error(f"❌ {company_name_krx}({ticker_code}) 데이터를 pykrx로 불러오지 못했습니다."); return

                df_kr.rename(columns={'시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'}, inplace=True)
                df_kr.index = pd.to_datetime(df_kr.index)

                required_cols_kr = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df_kr.columns for col in required_cols_kr):
                    st.error(f"❌ {company_name_krx} 데이터에 필수 OHLCV 컬럼 부족."); st.dataframe(df_kr.head()); return

                df_calculated_kr = df_kr.copy()
                df_calculated_kr.attrs['ticker'] = ticker_code

                for col in required_cols_kr: df_calculated_kr[col] = pd.to_numeric(df_calculated_kr[col], errors='coerce')
                df_calculated_kr.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                if df_calculated_kr.empty: st.error("❌ 유효한 가격 데이터 없음."); return

                try: df_calculated_kr = calculate_vwap(df_calculated_kr)
                except ValueError as ve: st.warning(f"VWAP 계산 실패: {ve}", icon="⚠️")
                except Exception as e: st.warning(f"VWAP 계산 중 오류: {e}", icon="⚠️")
                try: df_calculated_kr = calculate_bollinger_bands(df_calculated_kr, window=bb_window_val_kr, num_std=bb_std_val_kr)
                except Exception as e: st.warning(f"BB 계산 오류: {e}", icon="⚠️")
                try: df_calculated_kr = calculate_rsi(df_calculated_kr)
                except Exception as e: st.warning(f"RSI 계산 오류: {e}", icon="⚠️")
                try: df_calculated_kr = calculate_macd(df_calculated_kr)
                except Exception as e: st.warning(f"MACD 계산 오류: {e}", icon="⚠️")

                df_calculated_kr['SMA5'] = df_calculated_kr['Close'].rolling(window=5).mean()
                df_calculated_kr['SMA20'] = df_calculated_kr['Close'].rolling(window=20).mean()
                df_calculated_kr['SMA60'] = df_calculated_kr['Close'].rolling(window=60).mean()

                fig_kr = plot_korean_technical_chart(df_calculated_kr, ticker_code, company_name_krx)
                if fig_kr and fig_kr.data: st.plotly_chart(fig_kr, use_container_width=True)
                else: st.warning("한국 주식 차트 생성 실패.")

                st.subheader("📄 최근 데이터")
                display_cols_kr = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA5', 'SMA20', 'SMA60']
                display_cols_exist_kr = [col for col in display_cols_kr if col in df_calculated_kr.columns]
                format_dict_kr = {col: "{:,.0f}" for col in ['Open', 'High', 'Low', 'Close', 'VWAP', 'MA20', 'Upper', 'Lower', 'SMA5', 'SMA20', 'SMA60'] if col in display_cols_exist_kr}
                if 'Volume' in display_cols_exist_kr: format_dict_kr['Volume'] = "{:,.0f}"
                for col_rsi_macd in ['RSI', 'MACD', 'MACD_signal', 'MACD_hist']:
                    if col_rsi_macd in display_cols_exist_kr: format_dict_kr[col_rsi_macd] = "{:.2f}"
                st.dataframe(df_calculated_kr[display_cols_exist_kr].tail(10).style.format(format_dict_kr), use_container_width=True)

                st.divider()
                st.subheader("🧠 기술적 시그널 해석 (참고용)")
                if not df_calculated_kr.empty:
                    latest_row_kr = df_calculated_kr.iloc[-1].copy()
                    signal_messages_kr = []
                    try:
                        if 'interpret_technical_signals' in globals():
                            signal_messages_kr.extend(interpret_technical_signals(latest_row_kr, df_context=df_calculated_kr))
                        else: st.error("오류: 'interpret_technical_signals' 함수를 찾을 수 없습니다.")
                    except Exception as e: st.warning(f"기술적 시그널 해석 중 오류: {e}", icon="⚠️")
                    if signal_messages_kr: [st.info(msg) for msg in signal_messages_kr]
                    else: st.info("현재 특별히 감지된 기술적 시그널은 없습니다.")
                    st.caption("⚠️ **주의:** 자동 해석은 보조 지표이며 투자 결정은 반드시 종합적인 판단 하에 신중하게 내리시기 바랍니다.")
                else: st.warning("해석할 데이터가 부족합니다.")
            except Exception as e:
                st.error(f"한국 주식 데이터 처리 중 오류: {type(e).__name__} - {e}")
                logging.error(f"한국 주식 처리 오류 ({company_name_krx}): {traceback.format_exc()}")

# --- FMP API 키 로드 및 확인 ---
fmp_key_loaded = False
if 'fmp_api' in globals() and fmp_api: # fmp_api 모듈이 로드되었는지 확인
    secrets_available = hasattr(st, 'secrets')
    if secrets_available and "FMP_API_KEY" in st.secrets and st.secrets.FMP_API_KEY:
        fmp_api.FMP_API_KEY = st.secrets.FMP_API_KEY
        fmp_key_loaded = True
        logging.info("FMP API 키 로드 완료 (Streamlit Secrets).")
    elif os.getenv("FMP_API_KEY"):
        fmp_api.FMP_API_KEY = os.getenv("FMP_API_KEY")
        fmp_key_loaded = True
        logging.info("FMP API 키 로드 완료 (환경 변수).")
    else:
        logging.warning("FMP API 키가 Streamlit Secrets 또는 환경 변수에 설정되지 않았습니다.")
else:
    logging.warning("FMP API 모듈 로드 실패. FMP 관련 기능 사용 불가.")
comprehensive_analysis_possible = fmp_key_loaded

# --- DART API 키 확인 ---
dart_available = False
if dart_api_module_loaded and dart_api : # dart.py 모듈 로드 성공 및 dart_api 객체 유효성 확인
    # dart.py 내부에서 API_KEY 변수에 st.secrets 또는 os.getenv를 통해 값이 할당되었는지 확인
    try:
        if getattr(dart_api, 'API_KEY', None): # dart.py 모듈의 API_KEY 변수에 값이 있는지 확인
            dart_available = True
            logging.info(f"DART API 키 확인됨 (dart.py 내 API_KEY: {dart_api.API_KEY[:5]}...).")
        else:
            logging.warning("dart.py 모듈 내 API_KEY 변수에 값이 없습니다. dart.py의 키 로드 로직 확인 필요.")
    except Exception as e:
        logging.error(f"dart.py의 API_KEY 접근 중 오류: {e}")
else:
    logging.warning("dart.py 모듈이 로드되지 않았거나 dart_api 객체가 유효하지 않아 DART 키를 확인할 수 없습니다.")

# --- 사이드바 UI ---
with st.sidebar:
    if comprehensive_analysis_possible: st.success("FMP API 키 로드 완료.")
    else: st.error("FMP API 키 로드 실패! 미국 종합 분석 불가.")
    if dart_available: st.success("DART API 키 확인 완료.")
    else: st.warning("DART API 키 설정 확인 필요. 한국 기업 정보 조회 제한.")

    st.title("📊 주식 분석 도구")
    selected_country = st.radio("국가 선택", ["🇺🇸 미국 주식", "🇰🇷 한국 주식"], key="country_selector")
    st.markdown("---")
    page = None
    # ... (이하 사이드바 UI 설정은 이전과 거의 동일하게 유지) ...
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
        elif page == "📈 기술 분석 (FMP)":
            st.header("⚙️ 기술 분석 설정 (미국)")
            ticker_tech_us = st.text_input("종목 티커 (미국)", "AAPL", key="tech_ticker_input_us", help="예: AAPL")
            bb_window_us = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window_us")
            bb_std_us = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std_us", format="%.1f")
            today_us = datetime.now().date()
            default_start_us = today_us - relativedelta(months=3)
            start_date_us = st.date_input("시작일", default_start_us, key="tech_start_input_us", max_value=today_us - timedelta(days=1))
            end_date_us = st.date_input("종료일", today_us, key="tech_end_input_us", min_value=start_date_us + timedelta(days=1), max_value=today_us)
            available_intervals_tech_fmp = {"일봉": "1day", "1시간": "1hour", "15분": "15min"}
            interval_display_us = st.selectbox("데이터 간격", list(available_intervals_tech_fmp.keys()), key="tech_interval_display_input_us", help="FMP 플랜 따라 지원 간격 및 기간 다름")

    elif selected_country == "🇰🇷 한국 주식":
        page = st.radio("분석 유형 선택 (한국)", ["📈 기술 분석 (pykrx)", "📝 기본 정보 (DART)"],
                        captions=["차트 및 추세선", "기업 개요 및 공시정보"], key="page_selector_kr")
        st.markdown("---")
        if page == "📈 기술 분석 (pykrx)":
            st.header("⚙️ 기술 분석 설정 (한국)")
            ticker_tech_kr = st.text_input("종목명 또는 종목코드 (한국)", "삼성전자", key="tech_ticker_input_kr", help="예: 삼성전자 또는 005930")
            bb_window_kr = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window_kr")
            bb_std_kr = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std_kr", format="%.1f")
            today_kr = datetime.now().date()
            default_start_kr = today_kr - relativedelta(months=6)
            start_date_kr = st.date_input("시작일 (한국)", default_start_kr, key="tech_start_input_kr", max_value=today_kr - timedelta(days=1))
            end_date_kr = st.date_input("종료일 (한국)", today_kr, key="tech_end_input_kr", min_value=start_date_kr + timedelta(days=1), max_value=today_kr)
        elif page == "📝 기본 정보 (DART)":
            st.header("⚙️ 기본 정보 설정 (한국)")
            company_kr_info = st.text_input("기업명 또는 종목코드 (한국)", "삼성전자", key="company_info_input_kr", disabled=not dart_available)
            today_dart = datetime.now().date()
            default_start_dart = today_dart - relativedelta(years=1)
            start_date_dart = st.date_input("공시 검색 시작일", default_start_dart, key="dart_start_input", max_value=today_dart - timedelta(days=1), disabled=not dart_available)
            end_date_dart = st.date_input("공시 검색 종료일", today_dart, key="dart_end_input", min_value=start_date_dart + timedelta(days=1), max_value=today_dart, disabled=not dart_available)
    st.divider() # 모든 설정 UI 다음에 구분선 추가

# --- 캐시된 종합 분석 함수 ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, years, days, num_trend_periods, changepoint_prior_scale):
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    try:
        analysis_results = sa.analyze_stock(ticker, analysis_period_years=years, forecast_days=days, num_trend_periods=num_trend_periods, changepoint_prior_scale=changepoint_prior_scale)
        return analysis_results
    except NameError: return {"error": "종합 분석 모듈(stock_analysis.py) 로딩 실패"}
    except Exception as e: logging.error(f"analyze_stock ({ticker}): {e}\n{traceback.format_exc()}"); return {"error": f"종합 분석 중 오류: {e}"}

# --- 메인 화면 로직 ---
if page:
    if selected_country == "🇺🇸 미국 주식":
        if page == "📊 종합 분석 (FMP)":
            # ... (이전 미국 종합 분석 로직 전체: st.title, markdown, 버튼, 결과 표시 로직)
            # !!! 중요: 이 부분은 이전 답변에서 제공된 미국 종합 분석 로직을 그대로 가져와야 합니다.
            # !!! 길이가 매우 길어 여기에 다시 반복하지 않겠습니다.
            # !!! 예시:
            st.title("🇺🇸 미국 주식 종합 분석 (FMP API 기반)")
            st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공.")
            # ... (버튼 및 결과 표시 로직)
            pass # 실제 로직으로 대체 필요

        elif page == "📈 기술 분석 (FMP)":
            # ... (이전 미국 기술 분석 로직 전체: st.title, markdown, 버튼, 결과 표시 로직)
            # !!! 중요: 이 부분은 이전 답변에서 제공된 미국 기술 분석 로직을 그대로 가져와야 합니다.
            # !!! 길이가 매우 길어 여기에 다시 반복하지 않겠습니다.
            # !!! 예시:
            st.title("🇺🇸 미국 주식 기술적 분석 (FMP API)")
            st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준 등을 시각화하고 자동 해석합니다.")
            # ... (버튼 및 결과 표시 로직)
            pass # 실제 로직으로 대체 필요

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
                if not ticker_input_kr: results_placeholder_kr_tech.warning("종목명 또는 종목코드를 입력해주세요.")
                else:
                    display_korean_stock_chart(ticker_input_kr, start_date_kr, end_date_kr, bb_window_val_kr, bb_std_val_kr, results_placeholder_kr_tech)
            else: results_placeholder_kr_tech.info("⬅️ 사이드바에서 설정을 확인하고 '한국 주식 기술적 분석 실행' 버튼을 클릭하세요.")

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
                if not company_kr_info: results_placeholder_kr_info.warning("기업명 또는 종목코드를 입력해주세요.")
                elif not dart_api_module_loaded or not dart_api : results_placeholder_kr_info.error("DART 모듈 로드 실패로 조회할 수 없습니다.")
                elif not dart_available: results_placeholder_kr_info.error("DART API 키가 설정되지 않아 조회할 수 없습니다.")
                else:
                    with results_placeholder_kr_info:
                        st.info(f"{company_kr_info} 기업 정보를 DART에서 조회합니다...")
                        with st.spinner("DART 정보 조회 중..."):
                            async def run_dart_tasks(): # 비동기 작업을 위한 래퍼 함수
                                try:
                                    corp_code, matched_name = await dart_api.get_corp_code_by_name(company_kr_info)
                                    if not corp_code: st.error(f"DART에서 '{company_kr_info}' 기업을 찾을 수 없습니다: {matched_name}"); return
                                    st.success(f"기업 확인: {matched_name} (고유번호: {corp_code})")
                                    start_str = start_date_dart.strftime("%Y%m%d"); end_str = end_date_dart.strftime("%Y%m%d")
                                    disclosures, error_msg = await dart_api.get_disclosure_list(corp_code, start_str, end_str)
                                    if error_msg: st.error(f"공시 목록 조회 오류: {error_msg}"); return
                                    if not disclosures: st.warning(f"{start_str}~{end_str} 기간 내 {matched_name}의 정기 공시 없음."); return
                                    st.subheader(f"최근 정기 공시 ({len(disclosures)}건)"); df_disclosures = pd.DataFrame(disclosures)
                                    st.dataframe(df_disclosures[['rcept_dt', 'report_nm', 'corp_name', 'flr_nm']].rename(columns={'rcept_dt': '접수일', 'report_nm': '보고서명', 'corp_name': '회사명', 'flr_nm': '제출인'}), use_container_width=True)
                                    latest_business_report = next((d for d in disclosures if "사업보고서" in d.get('report_nm', '')), None)
                                    if latest_business_report and latest_business_report.get('rcept_no'):
                                        st.subheader("최근 사업보고서 개요"); overview_text = await dart_api.extract_business_section_from_dart(latest_business_report['rcept_no'], '사업의 개요')
                                        if any(err_msg in overview_text for err_msg in ["실패", "찾을 수 없습니다", "오류 발생"]): st.warning(f"사업 개요 추출 실패/오류: {overview_text}")
                                        else: st.text_area("사업의 개요 내용", overview_text, height=300)
                                except Exception as dart_e: st.error(f"DART 정보 조회 중 오류: {dart_e}"); logging.error(f"DART 정보 조회 오류: {traceback.format_exc()}")
                            try: asyncio.run(run_dart_tasks())
                            except RuntimeError as re_loop: st.warning(f"DART 비동기 작업 실행 루프 문제: {re_loop}. 이미 실행 중인 루프가 있을 수 있습니다.")
                            except Exception as e_async: st.error(f"DART 작업 실행 오류: {e_async}"); logging.error(f"DART 작업 실행 오류: {traceback.format_exc()}")
            else: results_placeholder_kr_info.info("⬅️ 사이드바에서 기업명을 입력하고 '한국 기업 정보 조회' 버튼을 클릭하세요.")
else:
    st.info("⬅️ 사이드바에서 국가와 분석 유형을 선택하세요.")

# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 | 정보 제공 목적")
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")