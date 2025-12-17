import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------
# 1. 프로젝트 루트 및 경로 설정
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# .env 로드
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# -----------------------------------------------------------
# 2. 모듈 Import
# -----------------------------------------------------------
from ai_pipeline.trade.kis_api import KISMockTrader
from ai_pipeline.trade.value_chain import ValueChainLoader
from ai_pipeline.feature_store.__init__ import OnlineFeatureStore
from ai_pipeline.boosting_model.train import StackingEnsemble
# [NEW] 포트폴리오 리밸런서 모듈 Import
# (rebalancer.py가 ai_pipeline/portfolio/ 폴더 혹은 루트에 있는지 확인 필요)
# 사용자가 올린 파일 위치가 루트라고 가정하고 경로 추가
try:
    from ai_pipeline.portfolio.rebalancer import PortfolioRebalancer
except ImportError:
    # 혹시 파일이 루트에 바로 있다면
    try:
        from rebalancer import PortfolioRebalancer
    except:
        print(" [Error] rebalancer.py를 찾을 수 없습니다.")
        sys.exit(1)

class AIAutoTrader:
    def __init__(self):
        print(" [System] AI 포트폴리오 매매 시스템 초기화...")
        
        self.kis = KISMockTrader()
        self.store = OnlineFeatureStore() 
        self.vc_loader = ValueChainLoader()
        
        # 모델 로드
        self.model = StackingEnsemble()
        model_dir = os.path.join(project_root, "ai_pipeline", "boosting_model", "models")
        try:
            self.model.load_model(model_dir)
            print(" [Model] AI 모델 로드 완료")
        except Exception as e:
            print(f" [Critical] 모델 로드 실패: {e}")
            sys.exit(1)

        # 리밸런싱 엔진 (Risk Neutral 모드)
        self.rebalancer = PortfolioRebalancer(risk_aversion='neutral')

        # 초기 감시 대상 (유니버스)
        self.target_list = ['005930', '000660', '000270', '005380', '035420'] 

    def analyze_market(self, stock_codes):
        """종목 리스트의 실시간 AI 점수 계산"""
        results = []
        for code in list(set(stock_codes)): # 중복제거
            # 1. Feature Store 조회
            features_df = self.store.get_realtime_features(code)
            if features_df is None or features_df.empty:
                continue
            
            # 2. 모델 예측
            try:
                probs = self.model.predict_proba(features_df)
                if hasattr(probs, 'ndim') and probs.ndim == 2:
                    score = probs[0, 1] * 100
                else:
                    score = probs[1] * 100
                
                results.append({
                    'code': code, # rebalancer는 'code' 컬럼을 원함
                    'ai_score': round(score, 2),
                    'current_price': int(features_df['close'].values[0])
                })
            except Exception:
                pass
        return pd.DataFrame(results)

    def get_account_status(self):
        """
        KIS API를 통해 현재 잔고와 보유 종목 현황을 가져옴
        Returns:
            total_asset (int): 총 평가 자산 (예수금 + 주식평가액)
            holdings (dict): {종목코드: 보유수량}
            cash (int): 주문가능 예수금
        """
        balance = self.kis.get_balance()
        if not balance:
            return 0, {}, 0

        holdings = {}
        stock_eval_total = 0
        
        # 보유 종목 파싱
        if 'output1' in balance:
            for item in balance['output1']:
                code = item['pdno']
                qty = int(item['hldg_qty'])
                # 현재가(prpr)나 평가금액(evlu_amt) 사용
                # API가 주는 평가금액은 지연될 수 있으므로 수량만 가져가서 나중에 최신가로 계산 추천
                holdings[code] = qty

        # 예수금 파싱 (output2 -> dnca_tot_amt: 예수금총액)
        try:
            cash = int(balance['output2'][0]['dnca_tot_amt'])
        except:
            cash = 0
            
        return holdings, cash

    def run_rebalancing(self):
        """
        [핵심 로직]
        1. 내 계좌 상태 확인 (현금 + 보유주식)
        2. 시장 분석 (보유주식 + 관심종목 + 밸류체인)
        3. 리밸런서 돌려서 목표 비중 산출
        4. 매도 먼저 -> 매수 나중 실행
        """
        print(f"\n [Run] {datetime.now().strftime('%H:%M:%S')} 리밸런싱 시작")

        # ---------------------------------------------------------
        # 1. 내 계좌 상태 확인
        # ---------------------------------------------------------
        current_holdings_qty, cash = self.get_account_status()
        print(f" [Asset] 예수금: {cash:,}원 / 보유종목: {len(current_holdings_qty)}개")

        # ---------------------------------------------------------
        # 2. 분석 대상 유니버스 구성
        # ---------------------------------------------------------
        # (1) 기본 타겟 + (2) 현재 보유 종목 (반드시 분석해야 함)
        analysis_universe = set(self.target_list) | set(current_holdings_qty.keys())
        
        # 1차 분석
        print(" [Analysis] 유니버스 분석 중...")
        df_scores = self.analyze_market(list(analysis_universe))
        
        if df_scores.empty:
            print(" 분석된 데이터가 없습니다.")
            return

        # (3) 밸류체인 확장 (점수 높은 종목의 관계사 추가)
        high_score_stocks = df_scores[df_scores['ai_score'] >= 80]['code'].tolist()
        expanded_universe = set()
        
        for code in high_score_stocks:
            related = self.vc_loader.get_related_stocks(code)
            for r_code in related:
                if r_code not in analysis_universe:
                    expanded_universe.add(r_code)
        
        if expanded_universe:
            print(f" [ValueChain] 관계사 {len(expanded_universe)}개 추가 분석")
            df_secondary = self.analyze_market(list(expanded_universe))
            df_scores = pd.concat([df_scores, df_secondary]).drop_duplicates('code')

        # ---------------------------------------------------------
        # 3. 리밸런싱 계획 수립 (Rebalancer)
        # ---------------------------------------------------------
        # Rebalancer는 금액(Amount) 기준으로 동작하므로 변환 필요
        # 현재 보유 수량 * 현재가 = 현재 보유 평가금액
        current_holdings_amt = {}
        price_map = df_scores.set_index('code')['current_price'].to_dict()
        
        total_stock_eval = 0
        for code, qty in current_holdings_qty.items():
            price = price_map.get(code, 0) # 만약 분석 실패했으면 0원 처리됨 (주의)
            if price == 0:
                # 분석 실패 시 API가 준 평가금액이라도 써야 하지만 여기선 패스
                print(f" [Warning] {code} 현재가 조회 실패. 리밸런싱에서 제외될 수 있음.")
                continue
            amt = qty * price
            current_holdings_amt[code] = amt
            total_stock_eval += amt

        total_assets = cash + total_stock_eval
        print(f" [Asset] 총 운용 자산: {total_assets:,}원 (주식: {total_stock_eval:,} + 현금: {cash:,})")

        # 리밸런서 실행
        # (내 보유금액 딕셔너리, AI 점수표, 총 자산)
        plan_df = self.rebalancer.run_ai_rebalancing(current_holdings_amt, df_scores, total_budget=total_assets)
        
        if plan_df.empty:
            print(" [Plan] 매매할 종목이 없습니다.")
            return

        print("\n [Plan] 매매 계획 (상위 5건)")
        print(plan_df.head(5)[['code', 'ai_score', 'action', 'diff', 'target_ratio']].to_string(index=False))

        # ---------------------------------------------------------
        # 4. 주문 실행 (Execution)
        # ---------------------------------------------------------
        # 중요: 매도를 먼저 해서 현금을 확보해야 매수가 가능함
        
        # (1) 매도 주문 (diff < 0)
        sell_orders = plan_df[plan_df['diff'] < 0]
        for _, row in sell_orders.iterrows():
            code = row['code']
            amount_to_sell = abs(row['diff']) # 팔아야 할 금액
            current_price = price_map.get(code, 0)
            
            if current_price > 0:
                # 금액 -> 수량 환산
                qty_to_sell = int(amount_to_sell // current_price)
                
                # 보유 수량보다 많이 팔 수 없음 (안전장치)
                current_qty = current_holdings_qty.get(code, 0)
                qty_to_sell = min(qty_to_sell, current_qty)

                if qty_to_sell > 0:
                    print(f"  [매도] {code} ({row['action']}) {qty_to_sell}주 (약 {qty_to_sell*current_price:,}원)")
                    self.kis.sell_limit(code, current_price, qty_to_sell)
                    time.sleep(0.2)
                    # 현금 확보 반영 (추정치)
                    cash += (qty_to_sell * current_price)

        # (2) 매수 주문 (diff > 0)
        buy_orders = plan_df[plan_df['diff'] > 0]
        for _, row in buy_orders.iterrows():
            code = row['code']
            amount_to_buy = row['diff']
            current_price = price_map.get(code, 0)
            
            if current_price > 0:
                qty_to_buy = int(amount_to_buy // current_price)
                
                # 예수금 체크
                cost = qty_to_buy * current_price
                if cost > cash:
                    # 돈 모자라면 살 수 있는 만큼만 다시 계산
                    qty_to_buy = int(cash // current_price)
                    cost = qty_to_buy * current_price
                
                if qty_to_buy > 0:
                    print(f"  [매수] {code} ({row['action']}) {qty_to_buy}주 (약 {cost:,}원)")
                    res = self.kis.buy_limit(code, current_price, qty_to_buy)
                    
                    if res and res.get('rt_cd') == '0':
                        cash -= cost
                        time.sleep(0.2)
                else:
                    pass # 살 돈이 없거나 1주 가격보다 적음

        print("\n [Done] 리밸런싱 종료")

if __name__ == "__main__":
    trader = AIAutoTrader()
    # 1회 실행 (스케줄러에 등록해서 사용하세요)
    trader.run_rebalancing()