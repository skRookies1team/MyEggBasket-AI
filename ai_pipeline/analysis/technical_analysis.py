import FinanceDataReader as fdr
import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    def __init__(self):
        pass

    def calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) 

    def get_technical_score(self, code):
        """
        [핵심] 특정 종목의 기술적 점수 계산 (0~100점)
        """
        try:
            # 최근 100일치 데이터 가져오기 (데이터 없으면 0.5 반환)
            df = fdr.DataReader(code)
            if df.empty or len(df) < 60:
                return 50.0

            # 1. RSI (상대강도지수): 낮을수록(과매도) 좋음 -> 반등 기대
            rsi = self.calculate_rsi(df['Close']).iloc[-1]
            if rsi <= 30: rsi_score = 100
            elif rsi >= 70: rsi_score = 0
            else: rsi_score = 100 - rsi

            # 2. 이동평균선 (추세): 현재가가 20일선 위에 있으면 좋음
            ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
            close = df['Close'].iloc[-1]
            trend_score = 100 if close > ma20 else 0

            # 3. 거래량 (수급): 거래량 터지면 호재 가능성
            vol_ratio = df['Volume'].iloc[-1] / (df['Volume'].iloc[-2] + 1)
            vol_score = 100 if vol_ratio > 1.5 else 50

            # 종합 점수 (가중 평균)
            final_score = (rsi_score * 0.4) + (trend_score * 0.4) + (vol_score * 0.2)
            
            return round(final_score, 2)

        except Exception as e:
            print(f" {code} 분석 실패: {e}")
            return 50.0 # 에러나면 기본 50점