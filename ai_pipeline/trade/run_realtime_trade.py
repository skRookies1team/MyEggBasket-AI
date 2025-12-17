import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# 프로젝트 루트 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# .env 로드
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# 모듈 Import
from ai_pipeline.trade.kis_api import KISMockTrader
from ai_pipeline.trade.value_chain import ValueChainLoader
from ai_pipeline.feature_store.__init__ import OnlineFeatureStore
from ai_pipeline.boosting_model.train import StackingEnsemble

class AIAutoTrader:
    def __init__(self):
        print(" [System] AI 자동매매 시스템 초기화 중...")
        
        # 1. 모듈 초기화
        self.kis = KISMockTrader()
        self.store = OnlineFeatureStore() # 실시간 데이터 파이프라인
        self.vc_loader = ValueChainLoader() # 밸류체인 로더
        
        # 2. 학습된 모델 로드
        self.model = StackingEnsemble()
        model_dir = os.path.join(project_root, "ai_pipeline", "boosting_model", "models")
        try:
            self.model.load_model(model_dir)
            print(" [Model] XGBoost/LGBM 모델 로드 완료")
        except Exception as e:
            print(f" [Critical] 모델 로드 실패: {e}")
            sys.exit(1)

        # 3. 감시 대상 종목 (초기 관심 종목)
        # 예시: 삼성전자, SK하이닉스, 기아, 현대차, 네이버 등
        self.target_list = ['005930', '000660', '000270', '005380', '035420'] 
        print(f" [Target] 초기 감시 종목: {len(self.target_list)}개")


    def analyze_market(self, stock_codes):
        results = []
        for code in stock_codes:
            features_df = self.store.get_realtime_features(code)
            if features_df is None or features_df.empty:
                continue
            
            try:
                probs = self.model.predict_proba(features_df)
                if hasattr(probs, 'ndim') and probs.ndim == 2:
                    up_prob = probs[0, 1]
                else:
                    up_prob = probs[0]
                
                ai_score = round(up_prob * 100, 2)
                
                if ai_score >= 70:
                    print(f"    🔍 [{code}] 분석: 점수 {ai_score}점")
                
                results.append({
                    'stock_code': code,
                    'ai_score': ai_score,
                    'current_price': features_df['close'].values[0] if 'close' in features_df.columns else 0
                })

            except Exception as e:
                pass
        return pd.DataFrame(results)

    def calc_buy_qty(self, current_price, total_cash, portfolio_ratio=0.2):
        """
        매수 수량 계산기
        - total_cash: 현재 예수금
        - portfolio_ratio: 한 종목당 최대 투자 비중 (기본 20%)
        """
        if current_price <= 0: return 0
        
        # 1. 한 종목에 투자할 금액 산정
        invest_amount = total_cash * portfolio_ratio
        
        # 2. 주문 가능 수량 계산 (시장가 변동 고려해 95%만 사용 등의 안전장치 가능)
        qty = int(invest_amount // current_price)
        
        return qty

    def run_once(self):
        """
        [1회 실행 모드]
        전체 분석 -> 포트폴리오 리밸런싱 -> 매매 -> 종료
        """
        print(f"\n [Run] {datetime.now().strftime('%H:%M:%S')} 포트폴리오 점검 및 리밸런싱 시작")
        
        # ---------------------------------------------------------
        # 1. 잔고 및 예수금 조회 (가장 먼저 해야 함)
        # ---------------------------------------------------------
        balance_data = self.kis.get_balance()
        if not balance_data:
            print(" [Error] 잔고 조회 실패로 중단합니다.")
            return

        # 보유 종목 파싱
        holdings = {}
        if 'output1' in balance_data:
            for item in balance_data['output1']:
                holdings[item['pdno']] = int(item['hldg_qty'])
        
        # 예수금 파싱 (output2의 dnca_tot_amt: 예수금총액)
        # 모의투자는 포맷이 다를 수 있으니 확인 필요. 보통 dnca_tot_amt 사용
        try:
            total_cash = int(balance_data['output2'][0]['dnca_tot_amt'])
            print(f" [Asset] 현재 예수금: {total_cash:,}원 / 보유종목: {len(holdings)}개")
        except:
            total_cash = 0
            print(" [Warning] 예수금 정보를 찾을 수 없어 0원으로 가정합니다.")

        # ---------------------------------------------------------
        # 2. 시장 분석 (기본 + 밸류체인)
        # ---------------------------------------------------------
        df_primary = self.analyze_market(self.target_list)
        if df_primary.empty:
            print("   분석 데이터 없음.")
            return

        # 밸류체인 확장
        high_score_stocks = df_primary[df_primary['ai_score'] >= 80]['stock_code'].tolist()
        expanded_codes = set()
        for code in high_score_stocks:
            related = self.vc_loader.get_related_stocks(code)
            for r_code in related:
                if r_code not in self.target_list:
                    expanded_codes.add(r_code)
        
        if expanded_codes:
            print(f"   [ValueChain] 관계사 {len(expanded_codes)}개 추가 분석")
            df_secondary = self.analyze_market(list(expanded_codes))
            final_df = pd.concat([df_primary, df_secondary]).drop_duplicates('stock_code')
        else:
            final_df = df_primary

        # ---------------------------------------------------------
        # 3. 매도 로직 (리밸런싱)
        # ---------------------------------------------------------
        print("\n [1] 매도 점검")
        for code, qty in holdings.items():
            if qty <= 0: continue
            
            row = final_df[final_df['stock_code'] == code]
            if not row.empty:
                score = row.iloc[0]['ai_score']
                price = row.iloc[0]['current_price']
                
                # 기준: 40점 이하 전량 매도
                if score <= 40:
                    print(f"   📉 [매도] {code} (점수 {score:.1f}) -> {qty}주 청산")
                    self.kis.sell_limit(code, int(price), qty)
                    # 매도하면 예수금이 늘어나므로 추정치 반영 (정확하진 않음)
                    total_cash += (price * qty)

        # ---------------------------------------------------------
        # 4. 매수 로직 (포트폴리오 구성)
        # ---------------------------------------------------------
        print("\n [2] 매수 점검")
        # 70점 이상인 종목들을 점수순으로 정렬
        buy_candidates = final_df[final_df['ai_score'] >= 70].sort_values('ai_score', ascending=False)
        
        for _, row in buy_candidates.iterrows():
            code = row['stock_code']
            score = row['ai_score']
            price = row['current_price']
            
            if code not in holdings:
                # 1. 자금 관리: 한 종목당 예수금의 20%만 투자 (최대 5종목 분산)
                #    또는 고정금액(예: 100만원)으로 설정 가능
                buy_qty = self.calc_buy_qty(price, total_cash, portfolio_ratio=0.2)
                
                if buy_qty > 0:
                    # 90점 이상이면 조금 더 공격적 투자 (비중 조절은 여기서)
                    if score >= 90:
                        pass # 원래대로 20%
                    elif score >= 70:
                        buy_qty = int(buy_qty * 0.5) # 70점대는 절반만(10%) 매수
                    
                    if buy_qty > 0:
                        print(f"   🚀 [매수] {code} ({score:.1f}점) -> {buy_qty}주 주문 (약 {buy_qty*price:,}원)")
                        res = self.kis.buy_limit(code, int(price), buy_qty)
                        
                        # 주문 성공 시 예상 예수금 차감
                        if res and res.get('rt_cd') == '0':
                            total_cash -= (buy_qty * price)
                            time.sleep(0.2)
                else:
                    if total_cash < price:
                        print(f"   Pass {code}: 예수금 부족 ({total_cash:,}원 < {price:,}원)")

        print("\n [Done] 트레이딩 사이클 종료.")

if __name__ == "__main__":
    trader = AIAutoTrader()
    
    # [선택 1] 한 번만 실행하고 종료 (스케줄러/Cron용)
    trader.run_once()
    
