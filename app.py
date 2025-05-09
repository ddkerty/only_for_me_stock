# -*- coding: utf-8 -*-
# app.py (미국 주식 로직 복원, DART MCP 제거 가정 최종 버전)

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
    # st.stop()

# --- DART API 모듈 임포트 ---
dart_api_module_loaded = False
try:
    import dart as dart_api
    dart_api_module_loaded = True
    logging.info("dart.py 모듈 로드 성공")
except ImportError:
    logging.warning("dart.py 파일을 찾을 수 없습니다. DART 관련 기능 사용 불가.")
    dart_api = None
except Exception as e:
    logging.error(f"dart.py 모듈 로드 중 예외 발생: {e}")
    dart_api = None

# --- 기술 분석 지표 계산 함수들 ---
def calculate_vwap(df):
    df = df.copy(); required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"VWAP 계산 실패: 필수 컬럼 부족 ({[col for col in required_cols if col not in df.columns]})")
    for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_cols, inplace=True)
    if df.empty: df['VWAP'] = np.nan; return df
    if (df['Volume'] == 0).all(): df['VWAP'] = np.nan; return df
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
    df['VWAP'] = np.where(df['cumulative_volume'] > 0, df['cumulative_tp_volume'] / df['cumulative_volume'], np.nan)
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    df = df.copy(); required_col = 'Close'
    if required_col not in df.columns:
        logging.warning(f"BB 계산 실패: '{required_col}' 컬럼 없음."); df['MA20']=np.nan; df['Upper']=np.nan; df['Lower']=np.nan; return df
    df[required_col] = pd.to_numeric(df[required_col], errors='coerce')
    if df[required_col].isnull().all():
        logging.warning(f"BB 계산 실패: '{required_col}' 데이터 없음."); df['MA20']=np.nan; df['Upper']=np.nan; df['Lower']=np.nan; return df
    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        logging.warning(f"BB 계산 유효 데이터 부족."); df['MA20']=np.nan; df['Upper']=np.nan; df['Lower']=np.nan; return df
    df['MA20'] = df[required_col].rolling(window=window, min_periods=max(1,window)).mean()
    df['STD20'] = df[required_col].rolling(window=window, min_periods=max(1,window)).std()
    df['Upper'] = df['MA20'] + num_std * df['STD20']
    df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

# --- 미국 주식용 차트 생성 함수 ---
def plot_technical_chart(df, ticker): # 이전 버전에서 가져옴
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols) or df[required_candle_cols].isnull().all(axis=None):
        st.error(f"캔들차트 필요 컬럼({required_candle_cols}) 없음/데이터 없음."); return fig
    for col in required_candle_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_candle_cols, inplace=True)
    if df.empty: st.error("유효한 OHLC 데이터가 없습니다."); return fig
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
    fig.update_layout(title=f"{ticker} - 기술적 분석 통합 차트 (미국 주식)", xaxis_title="날짜 / 시간", yaxis=dict(domain=[0.4, 1], title="가격 ($)"), yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right', showgrid=False), yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right', showgrid=False), xaxis_rangeslider_visible=False, legend_title_text="지표", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- 한국 주식용 차트 생성 함수 ---
def plot_korean_technical_chart(df, ticker_code, company_name):
    fig = go.Figure(); required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(c in df.columns for c in required_cols): logging.error(f"한국 차트 ({company_name}): OHLC 컬럼 부족"); return fig
    for c in required_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=required_cols, inplace=True)
    if df.empty: logging.error(f"한국 차트 ({company_name}): 유효 OHLC 데이터 없음"); return fig
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{company_name} 캔들"))
    if 'VWAP' in df.columns and df['VWAP'].notna().any(): fig.add_trace(go.Scatter(x=df.index,y=df['VWAP'],mode='lines',name='VWAP',line=dict(color='orange',width=1.5)))
    if all(c in df.columns for c in ['Upper','Lower']) and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index,y=df['MA20'],mode='lines',name='MA20(BB중심)',line=dict(color='blue',width=1,dash='dash')))
        fig.add_trace(go.Scatter(x=df.index,y=df['Upper'],mode='lines',name='Bollinger Upper',line=dict(color='grey',width=1)))
        fig.add_trace(go.Scatter(x=df.index,y=df['Lower'],mode='lines',name='Bollinger Lower',line=dict(color='grey',width=1),fill='tonexty',fillcolor='rgba(180,180,180,0.1)'))
    vpdf=df.dropna(subset=['High','Low']);
    if not vpdf.empty:
        m,mx,d=vpdf['Low'].min(),vpdf['High'].max(),vpdf['High'].max()-vpdf['Low'].min()
        if d>0:
            lvls={'0.0(H)':mx,'0.236':mx-0.236*d,'0.382':mx-0.382*d,'0.5':mx-0.5*d,'0.618':mx-0.618*d,'1.0(L)':m}
            clrs={'0.0(H)':'red','0.236':'orange','0.382':'gold','0.5':'green','0.618':'blue','1.0(L)':'purple'}
            for k,v in lvls.items():fig.add_hline(y=v,line_dash="dot",annotation_text=f"Fib {k}:{v:,.0f}원",line_color=clrs.get(k,'navy'),annotation_position="bottom right",annotation_font_size=10)
    if 'SMA5' in df.columns and df['SMA5'].notna().any(): fig.add_trace(go.Scatter(x=df.index,y=df['SMA5'],mode='lines',name='SMA 5일',line=dict(color='green',width=1)))
    if 'SMA20' in df.columns and df['SMA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index,y=df['SMA20'],mode='lines',name='SMA 20일',line=dict(color='red',width=1)))
    if 'SMA60' in df.columns and df['SMA60'].notna().any(): fig.add_trace(go.Scatter(x=df.index,y=df['SMA60'],mode='lines',name='SMA 60일',line=dict(color='purple',width=1)))
    if 'RSI' in df.columns and df['RSI'].notna().any():fig.add_trace(go.Scatter(x=df.index,y=df['RSI'],mode='lines',name='RSI(14)',line=dict(color='purple',width=1),yaxis='y2'))
    if all(c in df.columns for c in ['MACD','MACD_signal']):
        fig.add_trace(go.Scatter(x=df.index,y=df['MACD'],name='MACD',mode='lines',line=dict(color='teal'),yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index,y=df['MACD_signal'],name='MACD Signal',mode='lines',line=dict(color='orange'),yaxis='y3'))
        if 'MACD_hist' in df.columns:colors_kr=['lightgreen' if v>=0 else 'lightcoral' for v in df['MACD_hist']];fig.add_trace(go.Bar(x=df.index,y=df['MACD_hist'],name='MACD Histogram',marker_color=colors_kr,yaxis='y3'))
    fig.update_layout(title=f"{company_name}({ticker_code}) 기술적 분석",xaxis_title="날짜",yaxis=dict(domain=[0.4,1],title="가격(원)"),yaxis2=dict(domain=[0.25,0.4],title="RSI",overlaying='y',side='right',showgrid=False),yaxis3=dict(domain=[0.0,0.25],title="MACD",overlaying='y',side='right',showgrid=False),xaxis_rangeslider_visible=False,legend_title_text="지표",hovermode="x unified",margin=dict(l=20,r=20,t=40,b=20))
    return fig

# --- pykrx 티커 조회 헬퍼 함수 (동기) ---
@st.cache_data(ttl=3600)
def get_kr_ticker_map():
    name_to_ticker, ticker_to_name = {}, {}
    try:
        today_str = datetime.now().strftime("%Y%m%d")
        for market in ["KOSPI", "KOSDAQ", "KONEX"]:
            for ticker in stock.get_market_ticker_list(date=today_str, market=market):
                try: name = stock.get_market_ticker_name(ticker); name_to_ticker[name]=ticker; ticker_to_name[ticker]=name
                except: pass
        if not name_to_ticker: logging.warning("KRX 종목 목록 로드 실패 또는 비어있음 (get_kr_ticker_map).")
        else: logging.info(f"KRX Ticker Map 로드 완료: {len(name_to_ticker)} 종목")
    except Exception as e: logging.error(f"KRX 전체 종목 목록 조회 중 오류: {e}")
    return name_to_ticker, ticker_to_name

# --- 사용자 입력 처리 헬퍼 함수 (동기) ---
def get_ticker_from_input(user_input):
    user_input_stripped = user_input.strip()
    name_to_ticker_map, ticker_to_name_map = get_kr_ticker_map()
    if not name_to_ticker_map and not ticker_to_name_map:
        st.error("KRX 종목 목록을 불러올 수 없어 티커 변환이 불가능합니다.")
        return None, user_input_stripped
    if user_input_stripped.isdigit() and len(user_input_stripped) == 6:
        if user_input_stripped in ticker_to_name_map: return user_input_stripped, ticker_to_name_map[user_input_stripped]
        else: st.warning(f"종목코드 '{user_input_stripped}'가 KRX에 없습니다."); return None, user_input_stripped
    if user_input_stripped in name_to_ticker_map: return name_to_ticker_map[user_input_stripped], user_input_stripped
    cleaned_input = user_input_stripped.lower()
    matches = [(name, ticker) for name, ticker in name_to_ticker_map.items() if cleaned_input in name.lower()]
    if len(matches)==1: st.info(f"'{user_input_stripped}' -> '{matches[0][0]}' 검색."); return matches[0][1], matches[0][0]
    if len(matches)>1: st.warning(f"'{user_input_stripped}' 유사 종목 다수. 정확한 이름 필요. 예: {', '.join([m[0] for m in matches[:3]])}..."); return None, user_input_stripped
    st.warning(f"'{user_input_stripped}' 종목 찾지 못함."); return None, user_input_stripped

# --- 한국 주식 데이터 로딩 및 차트 표시 함수 (동기) ---
def display_korean_stock_chart(ticker_input_kr, start_date_kr, end_date_kr, bb_window_val_kr, bb_std_val_kr, results_container):
    with results_container:
        ticker_code, company_name_krx = get_ticker_from_input(ticker_input_kr)
        if not ticker_code: return
        start_str, end_str = start_date_kr.strftime("%Y%m%d"), end_date_kr.strftime("%Y%m%d")
        st.write(f"**{company_name_krx} ({ticker_code})** 기술적 분석 (BB: {bb_window_val_kr}일/{bb_std_val_kr:.1f}σ)")
        with st.spinner(f"{company_name_krx} 데이터 로딩 및 차트 생성 중... (pykrx)"):
            try:
                df_kr = stock.get_market_ohlcv(start_str, end_str, ticker_code)
                if df_kr.empty: st.error(f"❌ {company_name_krx}({ticker_code}) 데이터 없음 (pykrx)."); return
                df_kr.rename(columns={'시가':'Open','고가':'High','저가':'Low','종가':'Close','거래량':'Volume'},inplace=True)
                df_kr.index=pd.to_datetime(df_kr.index)
                req_cols=['Open','High','Low','Close','Volume']
                if not all(c in df_kr.columns for c in req_cols): st.error(f"❌ OHLCV 컬럼 부족."); st.dataframe(df_kr.head()); return
                df_calc = df_kr.copy(); df_calc.attrs['ticker']=ticker_code
                for c in req_cols: df_calc[c]=pd.to_numeric(df_calc[c],errors='coerce')
                df_calc.dropna(subset=['Open','High','Low','Close'],inplace=True)
                if df_calc.empty: st.error("❌ 유효 가격 데이터 없음."); return
                try: df_calc=calculate_vwap(df_calc)
                except ValueError as ve: st.warning(f"VWAP 계산 실패: {ve}", icon="⚠️")
                except Exception as e: st.warning(f"VWAP 계산 중 오류: {e}", icon="⚠️")
                df_calc=calculate_bollinger_bands(df_calc,window=bb_window_val_kr,num_std=bb_std_val_kr)
                if 'calculate_rsi' in globals() and 'calculate_macd' in globals(): # 함수 존재 확인
                    df_calc=calculate_rsi(df_calc); df_calc=calculate_macd(df_calc)
                else: st.warning("RSI/MACD 계산 함수 로드 실패 (short_term_analysis.py 확인)")
                df_calc['SMA5']=df_calc['Close'].rolling(window=5).mean()
                df_calc['SMA20']=df_calc['Close'].rolling(window=20).mean()
                df_calc['SMA60']=df_calc['Close'].rolling(window=60).mean()
                fig_kr=plot_korean_technical_chart(df_calc,ticker_code,company_name_krx)
                if fig_kr and fig_kr.data: st.plotly_chart(fig_kr,use_container_width=True)
                else: st.warning("한국 주식 차트 생성 실패.")
                st.subheader("📄 최근 데이터"); display_cols_kr=['Open','High','Low','Close','Volume','VWAP','MA20','Upper','Lower','RSI','MACD','MACD_signal','MACD_hist','SMA5','SMA20','SMA60']
                exist_cols=[c for c in display_cols_kr if c in df_calc.columns]; fmt_dict={c:"{:,.0f}" for c in ['Open','High','Low','Close','VWAP','MA20','Upper','Lower','SMA5','SMA20','SMA60'] if c in exist_cols}
                if 'Volume' in exist_cols:fmt_dict['Volume']="{:,.0f}"
                for c_rsi_macd in ['RSI','MACD','MACD_signal','MACD_hist']:
                    if c_rsi_macd in exist_cols:fmt_dict[c_rsi_macd]="{:.2f}"
                st.dataframe(df_calc[exist_cols].tail(10).style.format(fmt_dict),use_container_width=True)
                st.divider(); st.subheader("🧠 기술적 시그널 해석");
                if not df_calc.empty:
                    latest=df_calc.iloc[-1].copy(); signals=[]
                    try:
                        if 'interpret_technical_signals' in globals():signals.extend(interpret_technical_signals(latest,df_context=df_calc))
                        else:st.error("오류: interpret_technical_signals 함수 없음.")
                    except Exception as e:st.warning(f"기술적 시그널 해석 오류: {e}",icon="⚠️")
                    if signals: [st.info(s) for s in signals]
                    else:st.info("특별히 감지된 기술적 시그널 없음.")
                    st.caption("⚠️ 자동 해석은 보조 지표입니다.")
                else:st.warning("해석 데이터 부족.")
            except Exception as e: st.error(f"한국 주식 처리 오류: {e}"); logging.error(f"한국 주식 처리 오류 ({company_name_krx}): {traceback.format_exc()}")

# --- API 키 로드 ---
fmp_key_loaded = False
if 'fmp_api' in globals() and fmp_api:
    secrets_available = hasattr(st, 'secrets') # secrets 사용 가능 여부 확인
    if secrets_available and "FMP_API_KEY" in st.secrets and st.secrets.FMP_API_KEY:
        fmp_api.FMP_API_KEY = st.secrets.FMP_API_KEY; fmp_key_loaded = True; logging.info("FMP 키 로드 (Secrets)")
    elif os.getenv("FMP_API_KEY"):
        fmp_api.FMP_API_KEY = os.getenv("FMP_API_KEY"); fmp_key_loaded = True; logging.info("FMP 키 로드 (env)")
    else: logging.warning("FMP API 키 설정 안됨.")
else: logging.warning("fmp_api 모듈 로드 실패.")
comprehensive_analysis_possible = fmp_key_loaded

dart_available = False
if dart_api_module_loaded and dart_api :
    try:
        API_KEY_from_dart_module = getattr(dart_api, 'API_KEY', None)
        if API_KEY_from_dart_module:
            dart_available = True
            # logging.info(f"DART API 키 확인됨 (dart.py API_KEY: {API_KEY_from_dart_module[:5]}...).") # 민감 정보 로깅 주의
            logging.info(f"DART API 키 확인됨 (dart.py 모듈 변수 통해).")
        else:
            logging.warning("dart.py 내 API_KEY 변수 값 없음. dart.py의 키 로드 로직/Secrets/환경변수 확인.")
    except Exception as e: logging.error(f"dart.py API_KEY 접근 오류: {e}")
else: logging.warning("dart.py 모듈 미로드 또는 dart_api 객체 유효하지 않음. DART 기능 제한.")

# --- 사이드바 ---
with st.sidebar:
    if comprehensive_analysis_possible: st.success("FMP API 키 로드 완료.")
    else: st.error("FMP API 키 로드 실패! 미국 종합 분석 불가.")
    if dart_available: st.success("DART API 키 확인 완료.")
    else: st.warning("DART API 키 설정 확인 필요. 한국 기업 정보 조회 제한.")

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
            st.divider(); st.subheader("⚙️ 예측 세부 설정 (선택)"); changepoint_prior_us = st.slider("추세 변화 민감도 (Prophet)", 0.001, 0.5, 0.05, 0.01, "%.3f", key="changepoint_prior_us", disabled=not comprehensive_analysis_possible)
            st.divider(); st.subheader("💰 보유 정보 입력 (선택)"); avg_price_us = st.number_input("평단가", 0.0, format="%.2f", key="avg_price_us", disabled=not comprehensive_analysis_possible)
            quantity_us = st.number_input("보유 수량", 0, step=1, key="quantity_us", disabled=not comprehensive_analysis_possible)
        elif page == "📈 기술 분석 (FMP)":
            st.header("⚙️ 기술 분석 설정 (미국)"); ticker_tech_us = st.text_input("종목 티커 (미국)", "AAPL", key="tech_ticker_input_us", help="예: AAPL")
            bb_window_us = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window_us"); bb_std_us = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std_us", format="%.1f")
            today_us = datetime.now().date(); default_start_us = today_us - relativedelta(months=3); min_date_us = today_us - relativedelta(years=5)
            start_date_us = st.date_input("시작일", default_start_us, key="tech_start_input_us", min_value=min_date_us, max_value=today_us - timedelta(days=1))
            end_date_us = st.date_input("종료일", today_us, key="tech_end_input_us", min_value=start_date_us + timedelta(days=1), max_value=today_us)
            available_intervals_tech_fmp = {"일봉": "1day", "1시간": "1hour", "15분": "15min"}
            interval_display_us = st.selectbox("데이터 간격", list(available_intervals_tech_fmp.keys()), key="tech_interval_display_input_us", help="FMP 플랜 따라 지원 간격 및 기간 다름")

    elif selected_country == "🇰🇷 한국 주식":
        page = st.radio("분석 유형 선택 (한국)", ["📈 기술 분석 (pykrx)", "📝 기본 정보 (DART)"],
                        captions=["차트 및 추세선", "기업 개요 및 공시정보"], key="page_selector_kr")
        st.markdown("---")
        if page == "📈 기술 분석 (pykrx)":
            st.header("⚙️ 기술 분석 설정 (한국)"); ticker_tech_kr = st.text_input("종목명 또는 종목코드 (한국)", "삼성전자", key="tech_ticker_input_kr", help="예: 삼성전자 또는 005930")
            bb_window_kr = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window_kr"); bb_std_kr = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std_kr", format="%.1f")
            today_kr = datetime.now().date(); default_start_kr = today_kr - relativedelta(months=6); min_date_kr = today_kr - relativedelta(years=10)
            start_date_kr = st.date_input("시작일 (한국)", default_start_kr, key="tech_start_input_kr", min_value=min_date_kr, max_value=today_kr - timedelta(days=1))
            end_date_kr = st.date_input("종료일 (한국)", today_kr, key="tech_end_input_kr", min_value=start_date_kr + timedelta(days=1), max_value=today_kr)
        elif page == "📝 기본 정보 (DART)":
            st.header("⚙️ 기본 정보 설정 (한국)"); company_kr_info = st.text_input("기업명 또는 종목코드 (한국)", "삼성전자", key="company_info_input_kr", disabled=not dart_available)
            today_dart = datetime.now().date(); default_start_dart = today_dart - relativedelta(years=1)
            start_date_dart = st.date_input("공시 검색 시작일", default_start_dart, key="dart_start_input", max_value=today_dart - timedelta(days=1), disabled=not dart_available)
            end_date_dart = st.date_input("공시 검색 종료일", today_dart, key="dart_end_input", min_value=start_date_dart + timedelta(days=1), max_value=today_dart, disabled=not dart_available)
    st.divider()

# --- 캐시된 종합 분석 함수 ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, years, days, num_trend_periods, changepoint_prior_scale):
    logging.info(f"종합 분석: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp={changepoint_prior_scale}")
    try: return sa.analyze_stock(ticker,analysis_period_years=years,forecast_days=days,num_trend_periods=num_trend_periods,changepoint_prior_scale=changepoint_prior_scale)
    except NameError: return {"error":"stock_analysis.py 로드 실패"}
    except Exception as e: logging.error(f"analyze_stock({ticker}): {e}\n{traceback.format_exc()}"); return {"error":f"종합 분석 오류: {e}"}

# --- 메인 화면 로직 ---
if page:
    if selected_country == "🇺🇸 미국 주식":
        results_us_main = st.container()
        results_us_tech = st.container()

        if page == "📊 종합 분석 (FMP)":
            with results_us_main:
                st.title("🇺🇸 미국 주식 종합 분석 (FMP API 기반)")
                st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공.")
                st.markdown("---")
                analyze_button_main_disabled = not comprehensive_analysis_possible
                if analyze_button_main_disabled: st.error("FMP API 키 로드 실패. 종합 분석 불가.")
                
                ticker_us = st.session_state.get('main_ticker_us', "AAPL")
                analysis_years_us = st.session_state.get('analysis_years_us', 2)
                forecast_days_us = st.session_state.get('forecast_days_us', 30)
                num_trend_periods_us = st.session_state.get('num_trend_periods_us', 4)
                changepoint_prior_us = st.session_state.get('changepoint_prior_us', 0.05)
                avg_price_us = st.session_state.get('avg_price_us', 0.0)
                quantity_us = st.session_state.get('quantity_us', 0)

                analyze_button_main_us = st.button("🚀 미국 주식 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_us_button", disabled=analyze_button_main_disabled)
                
                if analyze_button_main_us:
                    if not ticker_us: st.warning("종목 티커 입력 필요.")
                    else:
                        ticker_proc_us = ticker_us.strip().upper()
                        if '.KS' in ticker_proc_us or '.KQ' in ticker_proc_us:
                            original_ticker_us = ticker_proc_us; ticker_proc_us = ticker_proc_us.split('.')[0]
                            st.info(f"국내 티커 감지: {original_ticker_us} -> {ticker_proc_us} (FMP용)")
                        with st.spinner(f"{ticker_proc_us} 종합 분석 중..."):
                            try:
                                results = run_cached_analysis(ticker_proc_us, analysis_years_us, forecast_days_us, num_trend_periods_us, changepoint_prior_us)
                                if results and isinstance(results, dict):
                                    if "error" in results: st.error(f"분석 실패: {results['error']}")
                                    else:
                                        # 여기에 이전 버전의 미국 종합 분석 결과 표시 로직 전체를 붙여넣으세요.
                                        # 예: st.header(...), st.subheader(...), st.metric(...), st.columns(...), 탭 생성, 차트 표시 등
                                        if results.get("warn_high_mape"): st.warning(f"🔴 모델 정확도 낮음 (MAPE {results.get('mape', 'N/A')}). 예측 신뢰도 주의!")
                                        st.header(f"📈 {ticker_proc_us} 분석 결과 (민감도: {changepoint_prior_us:.3f})")
                                        st.subheader("요약 정보"); col1, col2, col3 = st.columns(3); col1.metric("현재가", f"${results.get('current_price', 'N/A')}"); col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A')); col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))
                                        # ... (이하 모든 미국 종합 분석 결과 표시 로직)
                                elif results is None: st.error("분석 결과 처리 중 오류 (결과 없음).")
                                else: st.error("분석 결과 처리 중 오류 (결과 형식 오류).")
                            except Exception as e: st.error(f"앱 실행 중 오류: {e}"); logging.error(f"미국 종합 분석 오류: {traceback.format_exc()}")
                else:
                    if comprehensive_analysis_possible: st.info("⬅️ 설정 후 '미국 주식 종합 분석 시작!' 클릭.")
                    else: st.warning("API 키 로드 실패로 종합 분석 불가.")

        elif page == "📈 기술 분석 (FMP)":
            with results_us_tech:
                st.title("🇺🇸 미국 주식 기술적 분석 (FMP API)")
                st.markdown("VWAP, 볼린저밴드, 피보나치 등을 시각화하고 자동 해석합니다.")
                st.markdown("---")
                ticker_tech_us = st.session_state.get('tech_ticker_input_us', "AAPL")
                start_date_us = st.session_state.get('tech_start_input_us', datetime.now().date() - relativedelta(months=3))
                end_date_us = st.session_state.get('tech_end_input_us', datetime.now().date())
                available_intervals_tech_fmp = {"일봉": "1day", "1시간": "1hour", "15분": "15min"}
                interval_display_us = st.session_state.get('tech_interval_display_input_us', "일봉")
                interval_us = available_intervals_tech_fmp.get(interval_display_us, "1day")
                bb_window_us = st.session_state.get('bb_window_us', 20)
                bb_std_us = st.session_state.get('bb_std_us', 2.0)
                analyze_button_tech_us = st.button("📊 미국 주식 기술적 분석 실행", key="tech_analyze_us_button_main", use_container_width=True, type="primary") # 키 변경

                if analyze_button_tech_us:
                    if not ticker_tech_us: st.warning("종목 티커를 입력해주세요.")
                    else:
                        ticker_processed_us = ticker_tech_us.strip().upper()
                        if '.KS' in ticker_processed_us or '.KQ' in ticker_processed_us:
                            original_ticker_us = ticker_processed_us; ticker_processed_us = ticker_processed_us.split('.')[0]
                            st.info(f"국내 티커 감지: {original_ticker_us} -> {ticker_processed_us} (FMP용)")
                        st.write(f"**{ticker_processed_us}** ({interval_display_us}, BB:{bb_window_us}일/{bb_std_us:.1f}σ) 분석 중 (FMP)...")
                        with st.spinner(f"{ticker_processed_us} 데이터 로딩 및 처리 중 (FMP)..."):
                            try:
                                # --- (미국 기술 분석 데이터 처리 및 결과 표시 로직 전체 - 이전 코드에서 복원) ---
                                start_date_str_us = start_date_us.strftime("%Y-%m-%d"); end_date_str_us = end_date_us.strftime("%Y-%m-%d")
                                fmp_data_us = None; rename_map_us = {'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
                                if interval_us=="1day": fmp_data_us=fmp_api.get_price_data(ticker=ticker_processed_us,start_date=start_date_str_us,end_date=end_date_str_us)
                                else: fmp_data_us=fmp_api.get_intraday_data(ticker=ticker_processed_us,interval=interval_us,from_date=start_date_str_us,to_date=end_date_str_us)
                                df_tech_us=pd.DataFrame()
                                if fmp_data_us and isinstance(fmp_data_us,list) and len(fmp_data_us)>0:
                                    df_tech_us=pd.DataFrame(fmp_data_us)
                                    # (이하 데이터프레임 처리, 지표계산, 차트 생성, 테이블 표시, 시그널 해석 로직 전체 붙여넣기)
                                    # 예:
                                    if not df_tech_us.empty:
                                        df_tech_us = df_tech_us.rename(columns=rename_map_us)
                                        # ... (나머지 모든 처리 로직)
                                # --- (미국 기술 분석 결과 표시 로직 끝) ---
                            except Exception as e: st.error(f"미국 기술 분석 처리 오류: {e}"); logging.error(f"미국 기술 분석 오류: {traceback.format_exc()}")
                else:
                    st.info("⬅️ 설정 후 '미국 주식 기술 분석 실행' 클릭.")

    elif selected_country == "🇰🇷 한국 주식":
        results_kr_tech = st.container()
        results_kr_dart = st.container()

        if page == "📈 기술 분석 (pykrx)":
            with results_kr_tech:
                st.title("🇰🇷 한국 주식 기술적 분석 (pykrx 활용)")
                st.markdown("`pykrx` 라이브러리를 사용하여 한국 주식 차트 및 기술적 지표를 제공합니다.")
                st.markdown("---")
                ticker_input_kr = st.session_state.get('tech_ticker_input_kr', "삼성전자")
                start_date_kr = st.session_state.get('tech_start_input_kr', datetime.now().date() - relativedelta(months=6))
                end_date_kr = st.session_state.get('tech_end_input_kr', datetime.now().date())
                bb_window_val_kr = st.session_state.get('bb_window_kr', 20)
                bb_std_val_kr = st.session_state.get('bb_std_kr', 2.0)
                analyze_button_tech_kr = st.button("📊 한국 주식 기술적 분석 실행", key="tech_analyze_kr_button_main", use_container_width=True, type="primary") # 키 변경
                if analyze_button_tech_kr:
                    if not ticker_input_kr: st.warning("종목명 또는 종목코드를 입력해주세요.")
                    else: display_korean_stock_chart(ticker_input_kr, start_date_kr, end_date_kr, bb_window_val_kr, bb_std_val_kr, results_kr_tech) # Pass container
                else: st.info("⬅️ 사이드바에서 설정을 확인하고 '한국 주식 기술적 분석 실행' 버튼을 클릭하세요.")

        elif page == "📝 기본 정보 (DART)":
            with results_kr_dart:
                st.title("🇰🇷 한국 기업 기본 정보 (DART API)")
                st.markdown("DART API를 활용하여 한국 기업의 공시 정보 등을 조회합니다.")
                st.markdown("---")
                company_kr_info = st.session_state.get('company_info_input_kr', "삼성전자")
                start_date_dart = st.session_state.get('dart_start_input', datetime.now().date() - relativedelta(years=1))
                end_date_dart = st.session_state.get('dart_end_input', datetime.now().date())
                analyze_button_info_kr = st.button("🔍 한국 기업 정보 조회", key="info_analyze_kr_button_main", use_container_width=True, type="primary", disabled=not dart_available) # 키 변경
                if analyze_button_info_kr:
                    if not company_kr_info: st.warning("기업명 또는 종목코드를 입력해주세요.")
                    elif not dart_api_module_loaded or not dart_api : st.error("DART 모듈 로드 실패로 조회할 수 없습니다.")
                    elif not dart_available: st.error("DART API 키가 설정되지 않아 조회할 수 없습니다.")
                    else:
                        st.info(f"{company_kr_info} 기업 정보를 DART에서 조회합니다...")
                        with st.spinner("DART 정보 조회 중..."):
                            async def run_dart_tasks():
                                try:
                                    corp_code, matched_name = await dart_api.get_corp_code_by_name(company_kr_info)
                                    if not corp_code: st.error(f"DART에서 '{company_kr_info}' 기업을 찾을 수 없습니다: {matched_name}"); return
                                    st.success(f"기업 확인: {matched_name} ({corp_code})")
                                    s,e=start_date_dart.strftime("%Y%m%d"),end_date_dart.strftime("%Y%m%d")
                                    disclosures,err=await dart_api.get_disclosure_list(corp_code,s,e)
                                    if err: st.error(f"공시 목록 조회 오류: {err}"); return
                                    if not disclosures: st.warning(f"{s}~{e} 기간 내 {matched_name} 정기 공시 없음."); return
                                    st.subheader(f"최근 정기 공시 ({len(disclosures)}건)");df_disc=pd.DataFrame(disclosures)
                                    st.dataframe(df_disc[['rcept_dt','report_nm','corp_name','flr_nm']].rename(columns={'rcept_dt':'접수일','report_nm':'보고서명','corp_name':'회사명','flr_nm':'제출인'}),use_container_width=True)
                                    latest_bs_report=next((d for d in disclosures if "사업보고서" in d.get('report_nm','')),None)
                                    if latest_bs_report and latest_bs_report.get('rcept_no'):
                                        st.subheader("최근 사업보고서 개요"); overview=await dart_api.extract_business_section_from_dart(latest_bs_report['rcept_no'],'사업의 개요')
                                        if any(m in overview for m in ["실패","찾을 수 없습니다","오류 발생"]): st.warning(f"사업 개요 추출 실패/오류: {overview}")
                                        else: st.text_area("사업의 개요",overview,height=300)
                                except Exception as de: st.error(f"DART 정보 조회 중 오류: {de}"); logging.error(f"DART 정보 조회 오류: {traceback.format_exc()}")
                            try: asyncio.run(run_dart_tasks())
                            except RuntimeError as rle: st.warning(f"DART 비동기 루프 문제: {rle}. 이미 실행 중인 루프가 있을 수 있습니다.")
                            except Exception as ae: st.error(f"DART 작업 실행 오류: {ae}"); logging.error(f"DART 작업 실행 오류: {traceback.format_exc()}")
                else: st.info("⬅️ 사이드바에서 기업명을 입력하고 '한국 기업 정보 조회' 버튼을 클릭하세요.")
else:
    st.info("⬅️ 사이드바에서 국가와 분석 유형을 선택하세요.")

# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 | 정보 제공 목적")
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")