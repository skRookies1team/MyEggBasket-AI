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
from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy

STOCK_NAME_MAP = {
    "005930": "삼성전자", "000660": "SK하이닉스", "207940": "삼성바이오로직스",
    "005380": "현대차", "000270": "기아", "055550": "신한지주",
    "105560": "KB금융", "068270": "셀트리온", "015760": "한국전력",
    "028260": "삼성물산", "032830": "삼성생명", "012330": "현대모비스",
    "035420": "NAVER", "006400": "삼성SDI", "086790": "하나금융지주",
    "006405": "삼성SDI우", "000810": "삼성화재", "010140": "삼성중공업",
    "064350": "현대로템", "138040": "메리츠금융지주", "051910": "LG화학",
    "010130": "고려아연", "009540": "HD한국조선해양", "267260": "HD현대",
    "066570": "LG전자", "066575": "LG전자우", "033780": "KT&G",
    "003550": "LG", "003555": "LG우", "310200": "애니플러스",
    "034020": "두산에너빌리티", "012450": "한화에어로스페이스", "009830": "한화솔루션",
    "011070": "LG이노텍", "071050": "한국금융지주", "081660": "휠라홀딩스",
    "046890": "서울반도체", "323410": "카카오뱅크", "017670": "SK텔레콤",
    "010620": "현대미포조선", "047050": "포스코인터내셔널", "009155": "삼성전기우",
    "275630": "에이치시티", "009835": "한화솔루션우", "001440": "대한전선",
    "138930": "BNK금융지주", "175330": "JB금융지주", "051900": "LG생활건강",
    "005490": "POSCO홀딩스", "034220": "LG디스플레이"
}


def get_stock_name(code):
    return STOCK_NAME_MAP.get(code, code)


# -----------------------------------------------------------
# 3. 포트폴리오 리밸런서
# -----------------------------------------------------------
class PortfolioRebalancer:
    def __init__(self, risk_aversion='neutral'):
        self.risk_aversion = risk_aversion

    def run_ai_rebalancing(self, current_holdings_detail, ai_scores_df, total_budget, last_sell_times, last_buy_times):
        # 1. AI 데이터 검증
        if ai_scores_df is None or ai_scores_df.empty:
            return pd.DataFrame()

        if 'ai_score' not in ai_scores_df.columns:
            print(" [Error] AI 데이터에 'ai_score' 컬럼이 없습니다.")
            return pd.DataFrame()

        # 데이터 정제
        ai_scores_df = ai_scores_df.copy()
        ai_scores_df['code'] = ai_scores_df['code'].astype(str).str.strip().str.zfill(6)

        # 2. 보유 종목 데이터와 AI 점수 병합
        merged_df = ai_scores_df.copy()
        held_codes = set(current_holdings_detail.keys())

        prediction_codes = set(merged_df['code'].values)
        missing_holdings = held_codes - prediction_codes

        if missing_holdings:
            missing_data = []
            for code in missing_holdings:
                h_info = current_holdings_detail.get(code, {})
                current_price = h_info.get('current_price', 0)
                missing_data.append({
                    'code': code,
                    'name': get_stock_name(code),
                    'ai_score': 45.0,
                    'current_price': current_price
                })

            if missing_data:
                missing_df = pd.DataFrame(missing_data)
                merged_df = pd.concat([merged_df, missing_df], ignore_index=True)

        # -------------------------------------------------------
        # [파라미터 설정]
        # -------------------------------------------------------
        SELL_COOLDOWN_MINUTES = 41
        PROFIT_TAKE_RATE = 10.577529547538221
        STOP_LOSS_RATE = -10.227408445313205
        BUY_SCORE_THRESHOLD = 86
        SELL_SCORE_THRESHOLD = 50

        # [추가 설정] 자금 관리
        THRESHOLD_RATIO = 0.05
        BUY_MIN_HOLD_MINUTES = 30
        MAX_INDIVIDUAL_WEIGHT = 0.20  # [NEW] 종목당 최대 비중 20% 제한

        now = datetime.now()

        # 3. 필터링 (매수/유지 대상)
        if 'ai_score' not in merged_df.columns:
            print(" [Error] 병합된 데이터에 'ai_score' 컬럼이 누락되었습니다.")
            return pd.DataFrame()

        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= BUY_SCORE_THRESHOLD)
        cond_hold = (merged_df['code'].isin(held_codes))

        candidates = merged_df[cond_new_buy | cond_hold].copy()

        def check_buyable(row):
            code = row['code']
            if code in held_codes:
                return True
            if code in last_sell_times:
                elapsed = (now - last_sell_times[code]).total_seconds() / 60.0
                if elapsed < SELL_COOLDOWN_MINUTES:
                    return False
            return True

        if not candidates.empty:
            candidates = candidates[candidates.apply(check_buyable, axis=1)].copy()

        if candidates.empty:
            return pd.DataFrame()

        # 4. 비중 산출 (최대 비중 제한 적용)
        candidates['calc_score'] = candidates['ai_score'].apply(lambda x: x if x >= SELL_SCORE_THRESHOLD else 0)
        candidates['weight_score'] = np.power(candidates['calc_score'], 2)
        total_weight_score = candidates['weight_score'].sum()

        if total_weight_score > 0:
            # 전체 비중 계산
            raw_ratios = candidates['weight_score'] / total_weight_score
            # [핵심] 개별 종목 비중이 20%를 넘지 않도록 Cap 씌우기
            candidates['target_ratio'] = raw_ratios.apply(lambda x: min(x, MAX_INDIVIDUAL_WEIGHT))
        else:
            candidates['target_ratio'] = 0

        # 5. 최종 주문 생성
        rebalancing_plan = []
        threshold_amt = total_budget * THRESHOLD_RATIO

        for _, row in candidates.iterrows():
            code = row['code']
            name = row.get('name', code)
            ai_score = row['ai_score']
            target_ratio = row['target_ratio']

            holding_info = current_holdings_detail.get(code, {'qty': 0, 'avg_price': 0, 'current_price': 0, 'amt': 0})
            current_amt = holding_info['amt']
            avg_price = holding_info['avg_price']

            current_price = row.get('current_price', 0)
            if current_price == 0:
                current_price = holding_info.get('current_price', 0)

            target_amt = int(total_budget * target_ratio)
            diff = target_amt - current_amt

            profit_rate = 0.0
            if avg_price > 0 and current_price > 0:
                profit_rate = ((current_price - avg_price) / avg_price) * 100

            # 매매 결정 로직
            if diff > threshold_amt:
                base_action = '매수'
            elif diff < -threshold_amt:
                base_action = '비중축소'
            else:
                base_action = '유지'

            final_action = '유지'
            reason = f"목표비중 {target_ratio * 100:.1f}%"

            # [CASE 1] 손절매
            if profit_rate <= STOP_LOSS_RATE:
                final_action = '전량매도' if ai_score < 40 else '비중축소'
                reason = f"📉 손절매({profit_rate:.2f}%)"

            # [CASE 2] 익절
            elif profit_rate >= PROFIT_TAKE_RATE:
                if ai_score < 90:
                    final_action = '비중축소'
                    if target_amt == 0: final_action = '전량매도'
                    reason = f"💰 익절({profit_rate:.2f}%)"
                else:
                    if base_action == '매수':
                        final_action = '매수'
                        reason = f"🚀 급등({profit_rate:.2f}%) + AI강세"
                    else:
                        final_action = '유지'
                        reason = f"💰 익절권이나 상승세 유지"

            # [CASE 3] AI 점수 미달 확정 매도
            elif ai_score < 20:
                if code in last_buy_times:
                    elapsed_buy = (now - last_buy_times[code]).total_seconds() / 60.0
                    if elapsed_buy < BUY_MIN_HOLD_MINUTES:
                        final_action = '유지'
                        reason = f"⏳ 보유 대기({int(elapsed_buy)}분)"
                    else:
                        final_action = '전량매도'
                        reason = f"AI 점수 미달({ai_score}점)"
                else:
                    final_action = '전량매도'
                    reason = f"AI 점수 미달({ai_score}점)"

            # [CASE 4] 일반 리밸런싱
            else:
                if base_action == '비중축소':
                    if code in last_buy_times:
                        elapsed_buy = (now - last_buy_times[code]).total_seconds() / 60.0
                        if elapsed_buy < BUY_MIN_HOLD_MINUTES:
                            final_action = '유지'
                            reason = f"⏳ 보유 대기({int(elapsed_buy)}분)"
                        else:
                            final_action = base_action
                            reason = "비중 축소"
                    else:
                        final_action = base_action
                        reason = "비중 축소"
                else:
                    final_action = base_action
                    if final_action == '매수':
                        reason = "추가 매수"

            if final_action != '유지':
                rebalancing_plan.append({
                    'code': code,
                    'name': name,
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
            df_plan['abs_diff'] = df_plan['diff'].abs()
            df_plan = df_plan.sort_values(by='abs_diff', ascending=False).head(5)
            df_plan = df_plan.sort_values(by='diff', ascending=True)

        return df_plan


# -----------------------------------------------------------
# 4. 메인 트레이더 (AIAutoTrader)
# -----------------------------------------------------------
class AIAutoTrader:
    def __init__(self):
        print("\n" + "=" * 60)
        print("[AI AutoTrader] 관제형 자동매매 시스템 (Smart-Sync)")
        print("    - 수정사항: 종목당 최대 비중 20% 제한 적용")
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
        self.last_ai_scores = {}
        self.last_sell_times = {}
        self.last_buy_times = {}
        self.my_calculated_cash = None
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
            print(f"     로그 저장 완료 (수익률: {p_rate_str})")
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
        name = get_stock_name(code)
        print(f"      📡 주문 전송... [{name}({code}) {qty}주 {action}] (수익률 {profit_rate:.2f}%) ({reason})")
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
                print(f"     주문 성공! - {msg}")
                self.save_trade_log(code, action, qty, price, profit_rate, reason)
                if action == '매수':
                    self.last_buy_times[code] = datetime.now()
                if action == '전량매도':
                    self.last_sell_times[code] = datetime.now()
                    print(f"      🕒 [{code}] 매수 금지 쿨타임 시작 (24분)")
                    if code in self.last_buy_times:
                        del self.last_buy_times[code]
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
                'name': get_stock_name(code),
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

        d2_cash = summary.get('d2CashAmount')
        total_cash = summary.get('totalCashAmount', 0)

        def _parse_amount(val, default=0):
            if val is None: return default
            try:
                if isinstance(val, str): val = val.replace(',', '').strip()
                if val == '': return default
                return int(float(val))
            except:
                return default

        d2_amt = _parse_amount(d2_cash)
        total_amt = _parse_amount(total_cash)
        api_cash = d2_amt if (d2_amt is not None and d2_amt > 0) else total_amt

        if self.my_calculated_cash is None:
            self.my_calculated_cash = api_cash
            final_cash = api_cash
            print(f" [Init] 초기 예수금 설정: {final_cash:,}원")
        else:
            diff = api_cash - self.my_calculated_cash
            BUG_THRESHOLD = 30000000
            if diff > BUG_THRESHOLD:
                print(f" [Defense] 서버 예수금 급증 감지! (차이: {diff:,}원)")
                final_cash = self.my_calculated_cash
            else:
                if diff > 0:
                    print(f"️ [Sync] 예수금 차이({diff:,}원) 발생 -> 동기화")
                final_cash = api_cash
                self.my_calculated_cash = api_cash

        my_holdings_detail = {}
        for h in holdings_list:
            qty = int(h.get('quantity', 0))
            if qty > 0:
                code = h['stockCode']
                avg_price = float(h.get('avgPrice', 0))
                my_holdings_detail[code] = {'qty': qty, 'avg_price': avg_price, 'current_price': 0, 'amt': 0}

        print(f" 💰 가용예산(확정): {final_cash:,}원 | 보유종목: {len(my_holdings_detail)}개")

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

        high_scorers = [res for res in ai_results if res['ai_score'] >= 80]
        expanded_codes = set()
        if self.vc_strategy and self.vc_strategy.vc_analyzer:
            for item in high_scorers:
                main_code = item['code']
                related = self.vc_strategy.vc_analyzer.find_similar_stocks(main_code)
                for rel in related:
                    r_code = rel['code']
                    if r_code not in universe and r_code not in expanded_codes:
                        expanded_codes.add(r_code)

        if expanded_codes:
            print(f"  [ValueChain] 연관 유망 종목 {len(expanded_codes)}개 추가 분석 진행...")
            for r_code in expanded_codes:
                data = self.analyze_stock(r_code)
                if data:
                    ai_results.append(data)
                    print(f"    -> 밸류체인 추가: {r_code} ({data['ai_score']}점)")

        ai_scores_df = pd.DataFrame(ai_results)
        total_stock_val = sum([h['amt'] for h in my_holdings_detail.values()])
        total_asset = final_cash + total_stock_val

        print(f" [Asset] 총 자산: {total_asset:,}원 (주식:{total_stock_val:,} + 예수금:{final_cash:,})")

        plan_df = self.rebalancer.run_ai_rebalancing(
            my_holdings_detail,
            ai_scores_df,
            total_budget=total_asset,
            last_sell_times=self.last_sell_times,
            last_buy_times=self.last_buy_times
        )

        if plan_df.empty:
            print(" [Plan] 매매할 건이 없습니다.")
            return

        print("\n [리밸런싱 계획]")
        print(plan_df[['code', 'name', 'ai_score', 'action', 'diff', 'profit_rate', 'reason']].to_string(index=False))

        # 주문 실행
        for _, row in plan_df.iterrows():
            action = row['action']
            if action == '유지': continue

            code = row['code']
            name = row.get('name', code)
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
                    if self.send_order(code, action, price, qty_to_sell, profit_rate, row['reason']):
                        pass

            elif action == '매수':
                amt_to_buy = row['diff']
                safe_cash = final_cash * 0.95

                if amt_to_buy > safe_cash:
                    if safe_cash > 0:
                        print(f"       ⚠️ 예산 조정: 목표({amt_to_buy:,}) > 가능({safe_cash:,.0f}) -> 가능 금액만큼만 주문")
                        amt_to_buy = safe_cash
                    else:
                        print(f"       🚫 예수금 부족으로 매수 불가 ({code})")
                        continue

                qty_to_buy = int(amt_to_buy // price)
                if qty_to_buy > 0:
                    if self.send_order(code, action, price, qty_to_buy, profit_rate, row['reason']):
                        used_cash = (qty_to_buy * price)
                        final_cash -= used_cash
                        self.my_calculated_cash -= used_cash
                        print(f"     [Cash Update] 잔고 차감: -{used_cash:,}원 -> 남은예산: {self.my_calculated_cash:,}원")

        print(" [Cycle] 종료")


if __name__ == "__main__":
    trader = AIAutoTrader()
    try:
        while True:
            trader.run_cycle()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n [System] 자동매매 종료")