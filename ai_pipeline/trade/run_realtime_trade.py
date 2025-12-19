import sys
import os
import time
import requests
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------
# 1. 프로젝트 경로 및 환경 설정
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/app")
TEST_EMAIL = os.getenv("TEST_EMAIL", "testuser@example.com")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "password1234")
LOG_FILE_PATH = os.path.join(current_dir, "trade_record.csv")

# -----------------------------------------------------------
# 2. 모듈 Import
# -----------------------------------------------------------
from ai_pipeline.feature_store import OnlineFeatureStore
from ai_pipeline.boosting_model.train import StackingEnsemble

# 밸류체인 전략 모듈 Import
try:
    from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy
except ImportError:
    # 경로 문제 발생 시 현재 폴더에서 import 시도
    try:
        from value_chain_strategy import ValueChainStrategy
    except ImportError:
        print(" [Warning] ValueChainStrategy를 찾을 수 없습니다. 밸류체인 기능이 비활성화됩니다.")
        ValueChainStrategy = None


# -----------------------------------------------------------
# 3. 포트폴리오 리밸런서 (적극적 익절 + 매도 후 쿨타임)
# -----------------------------------------------------------
class PortfolioRebalancer:
    """
    - 적극적 익절 (3% 이상)
    - 전량 매도 종목 재진입 금지 (쿨타임)
    - 확정 신호(점수) 기반 칼같은 매도
    """

    def __init__(self, risk_aversion='neutral'):
        self.risk_aversion = risk_aversion

    def run_ai_rebalancing(self, current_holdings_detail, ai_scores_df, total_budget, last_sell_times):
        """
        last_sell_times: { '종목코드': datetime객체 } -> 전량 매도한 시간 기록
        """
        if ai_scores_df is None or ai_scores_df.empty:
            return pd.DataFrame()

        # 1. 데이터 정제
        cleaned_holdings = {str(k).strip().zfill(6): v for k, v in current_holdings_detail.items()}
        # 보유 종목 평가금 합계
        total_stock_val = sum([v['amt'] for v in cleaned_holdings.values()])

        ai_scores_df = ai_scores_df.copy()
        ai_scores_df['code'] = ai_scores_df['code'].astype(str).str.strip().str.zfill(6)

        if total_budget is None:
            # 예산이 안 넘어오면 보유분만 계산 (비상시)
            total_budget = total_stock_val

        # 2. 보유 종목 데이터 보정
        merged_df = ai_scores_df.copy()
        held_codes = set(cleaned_holdings.keys())

        prediction_codes = set(merged_df['code'].values)
        missing_holdings = held_codes - prediction_codes
        if missing_holdings:
            missing_data = []
            for code in missing_holdings:
                # 데이터 없으면 45점(유지) 부여
                missing_data.append({'code': code, 'ai_score': 45.0})
            merged_df = pd.concat([merged_df, pd.DataFrame(missing_data)], ignore_index=True)

        # -------------------------------------------------------
        # [쿨타임 필터링] 전량 매도한 지 얼마 안 된 종목은 매수 후보에서 제외
        # -------------------------------------------------------
        SELL_COOLDOWN_MINUTES = 24
        now = datetime.now()

        # 3. 필터링 (매수/유지 대상)
        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= 69)
        cond_hold = (merged_df['code'].isin(held_codes))

        # 전체 후보군
        candidates = merged_df[cond_new_buy | cond_hold].copy()

        # 매수 금지 필터 적용 (이미 보유중인건 제외하고, 신규 진입만 막음)
        def check_buyable(row):
            code = row['code']
            # 이미 보유중이면 쿨타임 상관없이 계산 (매도는 해야하니까)
            if code in held_codes:
                return True
            # 신규 진입인데 최근에 팔았다? -> 금지
            if code in last_sell_times:
                elapsed = (now - last_sell_times[code]).total_seconds() / 60.0
                if elapsed < SELL_COOLDOWN_MINUTES:
                    return False
            return True

        candidates = candidates[candidates.apply(check_buyable, axis=1)].copy()

        # 4. 비중 산출
        # 점수가 40점 미만이면 비중 0 (확정 매도 신호)
        candidates['calc_score'] = candidates['ai_score'].apply(lambda x: x if x >= 39 else 0)
        candidates['weight_score'] = np.power(candidates['calc_score'], 2)
        total_weight_score = candidates['weight_score'].sum()

        if total_weight_score > 0:
            candidates['target_ratio'] = candidates['weight_score'] / total_weight_score
        else:
            candidates['target_ratio'] = 0

        # 5. 최종 주문 생성
        rebalancing_plan = []
        THRESHOLD_RATIO = 0.02
        threshold_amt = total_budget * THRESHOLD_RATIO

        # [설정] 전략 파라미터
        PROFIT_TAKE_RATE = 2.2  # 익절 기준 3% (적극적)
        STOP_LOSS_RATE = -5.6  # 손절 기준 -3%

        for _, row in candidates.iterrows():
            code = row['code']
            ai_score = row['ai_score']
            target_ratio = row['target_ratio']

            holding_info = cleaned_holdings.get(code, {'qty': 0, 'avg_price': 0, 'current_price': 0, 'amt': 0})
            current_amt = holding_info['amt']
            avg_price = holding_info['avg_price']
            current_price = row.get('current_price', holding_info['current_price'])

            target_amt = int(total_budget * target_ratio)
            diff = target_amt - current_amt

            # 수익률 계산 (%)
            profit_rate = 0.0
            if avg_price > 0 and current_price > 0:
                profit_rate = ((current_price - avg_price) / avg_price) * 100

            # -----------------------------------------------------
            # [스마트 매매 결정 로직]
            # -----------------------------------------------------

            # 기본 액션 (리밸런싱)
            if diff > threshold_amt:
                base_action = '매수'
            elif diff < -threshold_amt:
                base_action = '비중축소'
            else:
                base_action = '유지'

            final_action = '유지'
            reason = f"목표비중 {target_ratio * 100:.1f}%"

            # [CASE 1] 손절매 (최우선)
            if profit_rate <= STOP_LOSS_RATE:
                final_action = '전량매도' if ai_score < 40 else '비중축소'
                reason = f" 손절매(수익률 {profit_rate:.2f}%)"

            # [CASE 2] 적극적 익절
            elif profit_rate >= PROFIT_TAKE_RATE:
                if ai_score < 90:
                    # 90점 미만이면 무조건 수익 실현 (보유 비중을 줄임)
                    final_action = '비중축소'
                    if target_amt == 0: final_action = '전량매도'  # 점수 낮으면 다 팜
                    reason = f" 익절(수익률 {profit_rate:.2f}%) - 이익 확정"
                else:
                    # 90점 이상 초강세 -> 불타기 or 유지
                    if base_action == '매수':
                        final_action = '매수'
                        reason = f" 급등({profit_rate:.2f}%) + AI강력({ai_score}점)"
                    else:
                        final_action = '유지'
                        reason = f" 익절권이나 상승세 유지({ai_score}점)"

            # [CASE 3] AI 점수 기반 확정 매도
            elif ai_score < 40:
                # 점수 미달은 묻지도 따지지도 않고 매도 (관망/완충 없음)
                final_action = '전량매도'
                reason = f"AI 점수 미달({ai_score}점) - 확정 매도"

            # [CASE 4] 일반 리밸런싱
            else:
                final_action = base_action
                if final_action == '비중축소':
                    reason = "리밸런싱 비중 축소"
                elif final_action == '매수':
                    reason = "리밸런싱 추가 매수"

            # 결과 담기
            if final_action != '유지':
                rebalancing_plan.append({
                    'code': code,
                    'ai_score': ai_score,
                    'current_amt': current_amt,
                    'target_amt': target_amt,
                    'diff': int(diff),
                    'action': final_action,
                    'profit_rate': profit_rate,
                    'reason': reason
                })

        df_plan = pd.DataFrame(rebalancing_plan)
        if not df_plan.empty:
            df_plan = df_plan.sort_values(by='diff', ascending=True)

        return df_plan


# -----------------------------------------------------------
# 4. 메인 트레이더 (AIAutoTrader)
# -----------------------------------------------------------
class AIAutoTrader:
    def __init__(self):
        print("\n" + "=" * 60)
        print("  [AI AutoTrader] 관제형 자동매매 시스템 (Aggressive)")
        print("    - 예산: D+2 예수금 사용 (가용금액 최대화)")
        print("    - 적극적 익절(3%) / 확정 매도(<40점) / 매도 후 20분 쿨타임")
        print("=" * 60)

        self.store = OnlineFeatureStore()
        self.rebalancer = PortfolioRebalancer(risk_aversion='neutral')

        # 밸류체인 전략 초기화
        if ValueChainStrategy:
            self.vc_strategy = ValueChainStrategy()
            print(" [Init] 밸류체인 전략 모듈 로드 완료")
        else:
            self.vc_strategy = None

        self.model = StackingEnsemble()
        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        try:
            self.model.load_model(model_path)
            print(" [Init] AI 모델 로드 완료")
        except Exception as e:
            print(f" [Error] 모델 로드 실패: {e}")
            sys.exit(1)

        self.target_codes = ["005930", "000660", "207940", "005380", "000270", "055550", "105560", "068270", "015760",
                             "028260", "032830", "012330", "035420", "006400", "086790", "006405", "000810", "010140",
                             "064350", "138040", "051910", "010130", "009540", "267260", "066570", "066575", "033780",
                             "003550", "003555", "310200", "034020", "012450", "009830", "011070", "071050", "081660",
                             "046890", "323410", "017670", "010620", "047050", "009155", "275630", "009835", "001440",
                             "138930", "175330", "051900", "005490", "034220"]
        self.auth_token = None

        # 상태 관리
        self.last_ai_scores = {}
        self.last_sell_times = {}  # 전량 매도 시간 기록 (재진입 금지용)

        self.init_csv_log()

    def init_csv_log(self):
        if not os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'code', 'action', 'qty', 'price', 'profit_rate', 'total_amt', 'reason'])

    def save_trade_log(self, code, action, qty, price, profit_rate, reason):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_amt = qty * price
            p_rate_str = f"{profit_rate:.2f}%"
            with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, code, action, qty, price, p_rate_str, total_amt, reason])
            print(f"       로그 저장 완료 (수익률: {p_rate_str})")
        except Exception as e:
            print(f"       로그 저장 실패: {e}")

    def login(self):
        url = f"{BACKEND_API_URL}/auth/login"
        payload = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                self.auth_token = resp.json().get('accessToken')
                print(f" [Auth] 백엔드 로그인 성공")
                return True
            else:
                print(f" [Auth] 로그인 실패: Status={resp.status_code}, Msg={resp.text}")
            return False
        except Exception as e:
            print(f" [Auth] 서버 연결 오류: {e}")
            return False

    def get_headers(self):
        if not self.auth_token: return {}
        return {'Authorization': f'Bearer {self.auth_token}'}

    def get_balance(self):
        url = f"{BACKEND_API_URL}/kis/trade/balance"
        params = {'virtual': 'true'}

        try:
            resp = requests.get(url, headers=self.get_headers(), params=params)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                print(f"       [Balance] 401 인증 실패. 재로그인 시도...")
                self.login()
                return None
            else:
                print(f"       [Balance] 조회 실패: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            print(f"       [Balance] 요청 중 예외 발생: {e}")
            return None

    def send_order(self, code, action, price, qty, profit_rate, reason):
        print(f"       주문 전송... [{code} {qty}주 {action}] (수익률 {profit_rate:.2f}%) ({reason})")

        time.sleep(1.0)

        url = f"{BACKEND_API_URL}/kis/trade"
        params = {'virtual': 'true'}
        order_type = "BUY" if action == '매수' else "SELL"

        payload = {
            "stockCode": code,
            "orderType": order_type,
            "quantity": qty,
            "price": price,
            "triggerSource": "MANUAL"
        }

        try:
            res = requests.post(url, headers=self.get_headers(), params=params, json=payload)

            if res.status_code == 200:
                msg = res.json().get('msg1', '주문 완료')
                print(f"       주문 성공! - {msg}")
                self.save_trade_log(code, action, qty, price, profit_rate, reason)

                # 전량 매도 시 쿨타임 시작
                if action == '전량매도':
                    self.last_sell_times[code] = datetime.now()
                    print(f"       [{code}] 매수 금지 쿨타임 시작 (20분)")

                return True
            else:
                print(f"       주문 실패: {res.status_code} - {res.text}")
                return False
        except Exception as e:
            print(f"       주문 중 에러 발생: {e}")
            return False

    def analyze_stock(self, code):
        features = self.store.get_realtime_features(code)

        if features is None or features.empty:
            if code in self.last_ai_scores:
                return self.last_ai_scores[code]
            else:
                return None

        try:
            probs = self.model.predict_proba(features)
            score = probs[0, 1] * 100 if hasattr(probs, 'ndim') and probs.ndim == 2 else probs[1] * 100

            current_price = int(features['close'].values[0])

            result = {
                'code': code,
                'ai_score': round(score, 2),
                'current_price': current_price
            }

            self.last_ai_scores[code] = result
            return result

        except Exception as e:
            print(f"       [{code}] 분석 중 오류: {e}")
            return None

    def run_cycle(self):
        print(f"\n  [Cycle] {datetime.now().strftime('%H:%M:%S')} 리밸런싱 시작")

        if not self.auth_token:
            if not self.login(): return

        balance_data = self.get_balance()
        if not balance_data: return

        summary = balance_data.get('summary', {})
        holdings_list = balance_data.get('holdings') or []

        # D+2 예수금 우선 사용 (없으면 totalCashAmount)
        # API 구조상 d2CashAmount가 있다면 그걸 쓰고, 없다면 totalCashAmount 사용
        d2_cash = summary.get('d2CashAmount')
        total_cash = summary.get('totalCashAmount', 0)

        # d2CashAmount가 null이거나 0이어도 totalCash보다 크거나,
        # d1이 엄청 커서 d2가 잡혀있는 구조일 수 있으니 안전하게 확인
        if d2_cash is not None:
            cash = int(d2_cash)
        else:
            cash = int(total_cash)

        my_holdings_detail = {}
        for h in holdings_list:
            code = h['stockCode']
            qty = int(h.get('quantity', 0))
            avg_price = float(h.get('avgPrice', 0))

            my_holdings_detail[code] = {
                'qty': qty,
                'avg_price': avg_price,
                'current_price': 0,
                'amt': 0
            }

        print(f"  가용예산(D+2): {cash:,}원 | 보유종목: {len(my_holdings_detail)}개")

        universe = set(self.target_codes) | set(my_holdings_detail.keys())
        ai_results = []

        for code in universe:
            data = self.analyze_stock(code)
            if data:
                if code in my_holdings_detail:
                    price = data['current_price']
                    my_holdings_detail[code]['current_price'] = price
                    my_holdings_detail[code]['amt'] = price * my_holdings_detail[code]['qty']

                ai_results.append(data)

        if not ai_results:
            print(" [Info] 분석 가능한 종목 데이터가 없습니다.")
            return
        
        # 밸류체인 확장 (대장주 발견 시 연관 종목 추가 분석) 
        # 80점 이상인 종목을 '대장주'로 간주

        high_scorers = [res for res in ai_results if res['ai_score'] >= 80]
        
        expanded_codes = set()
        if self.vc_strategy and self.vc_strategy.vc_analyzer:
            for item in high_scorers:
                main_code = item['code']
                # ValueChainAnalyzer를 통해 연관 종목 검색
                related = self.vc_strategy.vc_analyzer.find_similar_stocks(main_code)
                for rel in related:
                    r_code = rel['code']
                    # 이미 분석한 종목이거나 보유중이면 패스
                    if r_code not in universe and r_code not in expanded_codes:
                        expanded_codes.add(r_code)

        if expanded_codes:
            print(f"  [ValueChain] 연관 유망 종목 {len(expanded_codes)}개 추가 분석 진행...")
            for r_code in expanded_codes:
                data = self.analyze_stock(r_code)
                if data:
                    ai_results.append(data)
                    # 관계사 로그 출력
                    print(f"    -> 밸류체인 추가: {r_code} ({data['ai_score']}점)")


        ai_scores_df = pd.DataFrame(ai_results)

        total_stock_val = sum([h['amt'] for h in my_holdings_detail.values()])
        total_asset = cash + total_stock_val

        print(f" [Asset] 총 자산: {total_asset:,}원 (주식:{total_stock_val:,} + 예수금:{cash:,})")

        plan_df = self.rebalancer.run_ai_rebalancing(
            my_holdings_detail,
            ai_scores_df,
            total_budget=total_asset,
            last_sell_times=self.last_sell_times
        )

        if plan_df.empty:
            print(" [Plan] 매매할 건이 없습니다.")
            return

        print("\n [리밸런싱 계획]")
        print(plan_df[['code', 'ai_score', 'action', 'diff', 'profit_rate', 'reason']].to_string(index=False))

        # 주문 실행
        for _, row in plan_df.iterrows():
            action = row['action']
            if action == '유지': continue

            code = row['code']
            price = 0
            found = [x for x in ai_results if x['code'] == code]
            if found: price = found[0]['current_price']

            if price == 0 and code in my_holdings_detail:
                price = int(my_holdings_detail[code]['current_price'])

            if price == 0:
                print(f"       [{code}] 현재가를 알 수 없어 주문 생략")
                continue

            profit_rate = row.get('profit_rate', 0.0)

            if action in ['비중축소', '전량매도']:
                amt_to_sell = abs(row['diff'])
                qty_to_sell = int(amt_to_sell // price)
                if qty_to_sell > 0:
                    self.send_order(code, action, price, qty_to_sell, profit_rate, row['reason'])

            elif action == '매수':
                amt_to_buy = row['diff']
                # 예수금(D+2) 기준으로 매수 가능 여부 체크
                # 미수 발생 방지를 위해 95%만 사용
                safe_cash = cash * 0.95

                if safe_cash >= amt_to_buy:
                    qty_to_buy = int(amt_to_buy // price)
                    if qty_to_buy > 0:
                        if self.send_order(code, action, price, qty_to_buy, profit_rate, row['reason']):
                            cash -= (qty_to_buy * price)
                else:
                    if safe_cash > 100000 and amt_to_buy > 0:
                        print(f"       예수금 부족 ({code}): 필요 {amt_to_buy:,} > 가능 {safe_cash:,.0f}")

        print(" [Cycle] 종료")


if __name__ == "__main__":
    trader = AIAutoTrader()
    try:
        while True:
            trader.run_cycle()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n [System] 자동매매 종료")