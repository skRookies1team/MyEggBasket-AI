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
# 3. 포트폴리오 리밸런서 (신중한 매매: Threshold 상향 + Buy Cooldown + Top5)
# -----------------------------------------------------------
class PortfolioRebalancer:
    """
    - 적극적 익절 (2.25% 이상) / 손절 (-5.67% 이하)
    - 전량 매도 종목 재진입 금지 (24분 쿨타임)
    - 확정 신호(점수) 기반 칼같은 매도
    - [New] 매수 후 30분 보유, 비중차 5% 이상만 리밸런싱, Top 5 제한
    """

    def __init__(self, risk_aversion='neutral'):
        self.risk_aversion = risk_aversion

    def run_ai_rebalancing(self, current_holdings_detail, ai_scores_df, total_budget, last_sell_times, last_buy_times):
        """
        last_sell_times: { '종목코드': datetime } -> 전량 매도 시간 (재진입 금지)
        last_buy_times:  { '종목코드': datetime } -> 매수 시간 (단기 매도 방지)
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
        # 요청하신 고정 파라미터 적용
        SELL_COOLDOWN_MINUTES = 24

        # [설정] 전략 파라미터 (요청하신 정밀 값 적용)
        PROFIT_TAKE_RATE = 2.2498702085610604  # 약 2.25%
        STOP_LOSS_RATE = -5.6749957044906034  # 약 -5.67%
        BUY_SCORE_THRESHOLD = 69
        SELL_SCORE_THRESHOLD = 39  # 코드 로직상 39 미만 매도 등 활용

        # [추가 설정] 과도한 매매 방지
        THRESHOLD_RATIO = 0.05  # 비중 5% 이상 차이날 때만 매매 (기존 0.02)
        BUY_MIN_HOLD_MINUTES = 30  # 매수 후 최소 보유 시간

        now = datetime.now()

        # 3. 필터링 (매수/유지 대상)
        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= BUY_SCORE_THRESHOLD)
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
        # 점수가 39점 미만이면 비중 0 (확정 매도 신호) - 기존 로직 유지 (x >= 39)
        candidates['calc_score'] = candidates['ai_score'].apply(lambda x: x if x >= SELL_SCORE_THRESHOLD else 0)
        candidates['weight_score'] = np.power(candidates['calc_score'], 2)
        total_weight_score = candidates['weight_score'].sum()

        if total_weight_score > 0:
            candidates['target_ratio'] = candidates['weight_score'] / total_weight_score
        else:
            candidates['target_ratio'] = 0

        # 5. 최종 주문 생성
        rebalancing_plan = []
        threshold_amt = total_budget * THRESHOLD_RATIO

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

            # 기본 액션 (리밸런싱 - Threshold 적용)
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
                # 40점 미만이면 전량, 아니면 축소 (기존 로직 유지)
                final_action = '전량매도' if ai_score < 40 else '비중축소'
                reason = f" 손절매(수익률 {profit_rate:.2f}%)"

            # [CASE 2] 적극적 익절
            elif profit_rate >= PROFIT_TAKE_RATE:
                if ai_score < 90:
                    # 90점 미만이면 무조건 수익 실현
                    final_action = '비중축소'
                    if target_amt == 0: final_action = '전량매도'
                    reason = f"💰 익절(수익률 {profit_rate:.2f}%) - 이익 확정"
                else:
                    # 90점 이상 초강세 -> 불타기 or 유지
                    if base_action == '매수':
                        final_action = '매수'
                        reason = f" 급등({profit_rate:.2f}%) + AI강력({ai_score}점)"
                    else:
                        final_action = '유지'
                        reason = f" 익절권이나 상승세 유지({ai_score}점)"

            # [CASE 3] AI 점수 기반 확정 매도
            elif ai_score < 40:  # 기존 < 40 유지 (39점 포함 여부는 기존 로직 존중)
                final_action = '전량매도'
                reason = f"AI 점수 미달({ai_score}점) - 확정 매도"

            # [CASE 4] 일반 리밸런싱
            else:
                # [New] 매수 쿨타임 체크 (단순 비중 축소인 경우만)
                if base_action == '비중축소':
                    if code in last_buy_times:
                        elapsed_buy = (now - last_buy_times[code]).total_seconds() / 60.0
                        if elapsed_buy < BUY_MIN_HOLD_MINUTES:
                            final_action = '유지'  # 샀으면 좀 기다려라
                            reason = f"⏳ 매수 후 보유 대기({int(elapsed_buy)}분 경과)"
                        else:
                            final_action = base_action
                            reason = "리밸런싱 비중 축소"
                    else:
                        final_action = base_action
                        reason = "리밸런싱 비중 축소"
                else:
                    final_action = base_action
                    if final_action == '매수':
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

        # [New] 거래 건수 제한 (Top 5)
        # 중요도(금액 차이의 절대값) 순으로 정렬하여 상위 5개만 실행
        if not df_plan.empty:
            df_plan['abs_diff'] = df_plan['diff'].abs()
            df_plan = df_plan.sort_values(by='abs_diff', ascending=False).head(5)
            # 실행 순서는 기존대로 (매도 -> 매수) diff 오름차순
            df_plan = df_plan.sort_values(by='diff', ascending=True)

        return df_plan


# -----------------------------------------------------------
# 4. 메인 트레이더 (AIAutoTrader)
# -----------------------------------------------------------
class AIAutoTrader:
    def __init__(self):
        print("\n" + "=" * 60)
        print(" 🤖 [AI AutoTrader] 관제형 자동매매 시스템 (Hybrid Balance)")
        print("    - 수정사항: 서버 잔고 리셋 버그 방어 로직 적용")
        print("    - 로컬 잔고와 서버 잔고를 비교하여 비정상 급증 시 로컬 값 사용")
        print("=" * 60)

        self.store = OnlineFeatureStore()
        # [참고] 앞서 수정한 '신중한 매매' 로직이 적용된 Rebalancer라고 가정합니다.
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
        self.last_sell_times = {}
        self.last_buy_times = {}

        # [New] 내부 계산용 예수금 변수 (내 장부)
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
        # ... (생략: 기존과 동일) ...
        # 주의: 이 함수 내부 로직은 그대로 두셔도 됩니다.
        # 실제 예수금 차감 로직은 run_cycle에서 수행합니다.

        # 다만, 기존 코드 흐름상 여기 내용을 복붙해드릴 순 없으니
        # 아래 run_cycle 수정본을 참고해주세요.
        print(f"      📡 주문 전송... [{code} {qty}주 {action}] (수익률 {profit_rate:.2f}%) ({reason})")
        time.sleep(1.0)

        url = f"{BACKEND_API_URL}/kis/trade"
        params = {'virtual': 'true'}
        order_type = "BUY" if action == '매수' else "SELL"
        payload = {
            "stockCode": code, "orderType": order_type,
            "quantity": qty, "price": price, "triggerSource": "MANUAL"
        }
        try:
            res = requests.post(url, headers=self.get_headers(), params=params, json=payload)
            if res.status_code == 200:
                msg = res.json().get('msg1', '주문 완료')
                print(f"    주문 성공! - {msg}")
                self.save_trade_log(code, action, qty, price, profit_rate, reason)
                if action == '매수': self.last_buy_times[code] = datetime.now()
                if action == '전량매도':
                    self.last_sell_times[code] = datetime.now()
                    print(f"      🕒 [{code}] 매수 금지 쿨타임 시작 (24분)")
                    if code in self.last_buy_times: del self.last_buy_times[code]
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

        # [수정] D+2 예수금 우선 사용 (없으면 totalCashAmount)
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

        # API가 주는 현재 예수금
        api_cash = d2_amt if (d2_amt is not None and d2_amt > 0) else total_amt

        # ------------------------------------------------------------------
        # [Hybrid Logic] 서버 잔고 vs 로컬 잔고 검증
        # ------------------------------------------------------------------

        # 최초 실행 시에는 API 값을 믿고 초기화
        if self.my_calculated_cash is None:
            self.my_calculated_cash = api_cash
            final_cash = api_cash
            print(f"[Init] 초기 예수금 설정: {final_cash:,}원")
        else:
            # 검증: API 잔고가 내 계산보다 20% 이상 많으면 '버그(리셋)'로 간주
            # (단, 100만원 이하 소액일 때는 20% 차이날 수 있으므로 예외 처리)
            diff = api_cash - self.my_calculated_cash
            is_abnormal_increase = (diff > 0) and (api_cash > self.my_calculated_cash * 1.2) and (
                        self.my_calculated_cash > 1000000)

            if is_abnormal_increase:
                print(f" [Defense] 서버 예수금 리셋 감지! (API: {api_cash:,}원 vs My: {self.my_calculated_cash:,}원)")
                print(f"    -> API 값을 무시하고, 내부 계산된 잔고({self.my_calculated_cash:,}원)를 사용합니다.")
                final_cash = self.my_calculated_cash
            else:
                # 정상 범위라면 API 값으로 동기화 (이자, 수수료 등의 미세 차이 보정)
                # 단, API 값이 내 계산보다 작으면(정상 출금 등) API를 믿음
                final_cash = api_cash
                self.my_calculated_cash = api_cash  # 동기화

        print(f" 💰 가용예산(확정): {final_cash:,}원 | 보유종목: {len(holdings_list)}개")

        my_holdings_detail = {}
        for h in holdings_list:
            code = h['stockCode']
            qty = int(h.get('quantity', 0))
            avg_price = float(h.get('avgPrice', 0))
            my_holdings_detail[code] = {'qty': qty, 'avg_price': avg_price, 'current_price': 0, 'amt': 0}

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

        # 총 자산 계산 시 final_cash 사용
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
                    # 매도 성공 시에는 일단 API 반영을 기다림 (세금/수수료 계산 복잡하므로)
                    # 단, 보수적으로 로컬 잔고는 안 늘려도 됨 (어차피 다음 턴에 API가 정상이면 동기화됨)
                    if self.send_order(code, action, price, qty_to_sell, profit_rate, row['reason']):
                        # (선택) 매도 시 추정 현금 증가분을 로컬에 반영하고 싶다면:
                        # estimated_income = qty_to_sell * price * 0.997 # 수수료 대충 반영
                        # self.my_calculated_cash += int(estimated_income)
                        pass

            elif action == '매수':
                amt_to_buy = row['diff']
                # 예수금 체크는 final_cash 기준
                safe_cash = final_cash * 0.95

                if safe_cash >= amt_to_buy:
                    qty_to_buy = int(amt_to_buy // price)
                    if qty_to_buy > 0:
                        if self.send_order(code, action, price, qty_to_buy, profit_rate, row['reason']):
                            # [중요] 매수 성공 시, 로컬 잔고와 현재 사용 중인 잔고 즉시 차감
                            used_cash = (qty_to_buy * price)
                            final_cash -= used_cash
                            self.my_calculated_cash -= used_cash
                            print(f"     [Cash Update] 잔고 차감: -{used_cash:,}원 -> 남은예산: {self.my_calculated_cash:,}원")
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