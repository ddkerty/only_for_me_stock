# short_term_analysis.py (중복 제거 및 Wilder's RSI 적용)

import pandas as pd
import numpy as np

def interpret_fibonacci(df: pd.DataFrame,
                        close_value: float | None = None,
                        prev_close: float | None = None) -> str | None:
    """
    피보나치 되돌림 수준별 시나리오·전략 제안 포함 해석
    - prev_close: 직전 봉 종가(있으면 ‘돌파/이탈’ 판단)
    """
    if df.empty or close_value is None:
        return None

    try:
        # 유효한 High/Low 값만 필터링하여 min/max 계산
        valid_df = df.dropna(subset=['Low', 'High'])
        if valid_df.empty:
            return "피보나치 분석 불가 (유효한 고가/저가 데이터 없음)"

        low, high = valid_df['Low'].min(), valid_df['High'].max()
        diff = high - low
        if diff <= 0:
            return "피보나치 분석 불가 (고가·저가 차이 없음)"

        # 레벨 값 계산
        levels = {
            0.0:  high,
            0.236: high - 0.236 * diff,
            0.382: high - 0.382 * diff,
            0.5:   high - 0.5   * diff,
            0.618: high - 0.618 * diff,
            1.0:  low,
        }

        # 가까운 레벨 찾기 (±1.5 % 이내) - diff로 나누어 비율로 비교
        nearest = min(levels.items(),
                      key=lambda kv: abs(close_value - kv[1]))
        ratio, lvl_price = nearest

        # diff가 매우 작은 경우 division by zero 방지
        threshold_ratio = 0.015 # 1.5%
        if diff > 1e-9 and abs(close_value - lvl_price) / diff > threshold_ratio:
             return "현재가는 주요 피보나치 레벨에서 멀리 떨어져 있어요."
        elif diff <= 1e-9 and abs(close_value - lvl_price) > lvl_price * threshold_ratio: # diff가 거의 0일 때 가격 자체의 1.5%로 비교
             return "현재가는 주요 피보나치 레벨에서 멀리 떨어져 있어요."


        # 레벨별 시나리오/전략
        comments = {
            0.236: ("얕은 되돌림 후 강세 재개 가능성이 커 보입니다.",
                    "전 고점 돌파 시 추세추종 매수 고려"),
            0.382: ("첫 번째 핵심 지지선입니다.",
                    "하향 돌파 시 0.5까지 눌림 가능성을 염두에 두세요."),
            0.5:   ("추세가 중립으로 전환되는 분기점이에요.",
                    "방향 확인 전까지 관망 또는 포지션 축소가 안전합니다."),
            0.618: ("되돌림의 마지막 보루로 평가됩니다.",
                    "반등 캔들 + 거래량 증가 시 진입, 반대로 종가 이탈 시 손절 고려"),
            1.0:   ("저점을 다시 시험 중입니다.",
                    "지지 실패 시 하락 추세 강화, 성공 시 쌍바닥 반등 시도 가능"),
            0.0:   ("고점 부근이며 차익 실현 압력이 커질 수 있어요.",
                    "음봉 전환·거래량 감소 확인 시 익절 분할 매도 고려"),
        }

        # 이전 봉 대비 돌파/이탈 감지
        breach_msg = ""
        if prev_close is not None:
            if prev_close < lvl_price <= close_value:
                breach_msg = "▶ **상향 돌파** 신호가 나왔습니다."
            elif prev_close > lvl_price >= close_value:
                breach_msg = "▶ **하향 이탈** 신호가 나왔습니다."

        headline = (f"🔍 **현재가가 피보나치 {ratio:.3f}"
                    f" 레벨(${lvl_price:.2f}) 근처입니다.**") # 통화 기호 추가
        body, strategy = comments.get(ratio, ("", ""))
        msg = f"{headline}\n- {body}\n- {strategy}"
        if breach_msg:
            msg += f"\n{breach_msg}"
        return msg

    except Exception as e:
        # 로깅 추가 권장
        # logging.error(f"피보나치 해석 중 오류: {e}\n{traceback.format_exc()}")
        return f"⚠️ 피보나치 해석 오류: {e}"


# --- Wilder's Smoothing 적용된 RSI 계산 함수 ---
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Wilder’s RSI 계산 (EMA smoothing 방식). DataFrame에 'RSI' 컬럼 추가.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("RSI 계산을 위해 'Close' 컬럼이 필요합니다.")

    delta = df['Close'].diff()

    # gain과 loss 계산 시 초기 NaN 값 처리 (fillna(0))
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)

    # Wilder’s smoothing (EMA 방식, com = period - 1)
    # min_periods=period 를 설정하여 초기 기간 동안은 NaN 반환
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    # 0으로 나누기 방지: avg_loss가 0이면 RS를 무한대로 설정 (RSI = 100)
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)

    # RSI 계산
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # RSI 결과가 무한대인 경우 100으로 설정 (avg_loss가 0인 경우)
    df['RSI'] = df['RSI'].replace([np.inf, -np.inf], 100)

    # 초기 NaN 값 처리 (선택적: 예를 들어 이전 값으로 채우거나 그냥 NaN으로 둘 수 있음)
    # df['RSI'] = df['RSI'].fillna(method='bfill') # 예시: 뒤의 값으로 채우기

    return df


# --- MACD 계산 함수 ---
def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD line, signal line, histogram 계산. DataFrame에 컬럼 추가.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("MACD 계산을 위해 'Close' 컬럼이 필요합니다.")

    # EMA 계산 (adjust=False는 초기값에 덜 민감하게 만듦)
    ema_fast   = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow   = df['Close'].ewm(span=slow, adjust=False).mean()

    # MACD Line 계산
    macd_line  = ema_fast - ema_slow

    # Signal Line 계산 (MACD Line의 EMA)
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram 계산
    macd_hist  = macd_line - sig_line

    # DataFrame에 결과 컬럼 추가
    df['MACD']         = macd_line
    df['MACD_signal']  = sig_line
    df['MACD_hist']    = macd_hist

    return df

# --- 하단의 중복 함수 정의는 제거됨 ---