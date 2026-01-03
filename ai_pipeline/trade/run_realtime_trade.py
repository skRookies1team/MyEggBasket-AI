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

# [설정 변경] Secret Key 사용
AI_SECRET_KEY = os.getenv("AI_SECRET_KEY", "myeggbasketsecretkey00")
BACKEND_API_URL = "http://localhost:8081/api/internal"  # 내부 API 주소
LOG_FILE_PATH = os.path.join(current_dir, "trade_record1.csv")

# -----------------------------------------------------------
# 2. 모듈 Import
# -----------------------------------------------------------
try:
    from ai_pipeline.feature_store import OnlineFeatureStore
    from ai_pipeline.boosting_model.train import StackingEnsemble
    from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy
except ImportError:
    # 경로 문제 시 로컬 경로 기준으로 다시 시도 (Docker 등 환경 고려)
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../../")))
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
            raw_ratios = candidates['weight_score'] / total_weight_score
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
        print("[AI AutoTrader] 관제형 자동매매 시스템 (Multi-User Support)")
        print(f"    - Target API: {BACKEND_API_URL}")
        print("=" * 60)

        self.store = OnlineFeatureStore()
        self.rebalancer = PortfolioRebalancer(risk_aversion='neutral')
        self.headers = {"X-AI-SECRET": AI_SECRET_KEY}  # 인증 헤더

        # 밸류체인 전략 초기화
        try:
            if ValueChainStrategy:
                self.vc_strategy = ValueChainStrategy()
                print(" [Init] 밸류체인 전략 모듈 로드 완료")
            else:
                self.vc_strategy = None
        except Exception:
            self.vc_strategy = None

        # AI 모델 로드
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

        # 유저별 상태 관리 (last_sell_times 등)
        self.user_states = {}
        self.init_csv_log()

    def init_csv_log(self):
        if not os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['timestamp', 'user_id', 'code', 'action', 'qty', 'price', 'profit_rate', 'total_amt', 'reason'])

    def save_trade_log(self, user_id, code, action, qty, price, profit_rate, reason):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_amt = qty * price
            p_rate_str = f"{profit_rate:.2f}%"
            with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, user_id, code, action, qty, price, p_rate_str, total_amt, reason])
            print(f"     [Log] 저장 완료 (User: {user_id})")
        except Exception as e:
            print(f"       [Log] 저장 실패: {e}")

    def get_active_users(self):
        try:
            url = f"{BACKEND_API_URL}/users"
            res = requests.get(url, headers=self.headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                return data
            else:
                print(f"[System] 유저 목록 조회 실패: {res.status_code}")
                return []
        except Exception as e:
            print(f"[System] API 오류 (get_active_users): {e}")
            return []

    def get_user_balance(self, user_id):
        url = f"{BACKEND_API_URL}/balance/{user_id}"
        try:
            res = requests.get(url, headers=self.headers, timeout=5)
            if res.status_code == 200:
                return res.json()
            else:
                print(f" [User {user_id}] 잔고 조회 실패: {res.status_code}")
                return None
        except Exception as e:
            print(f" [User {user_id}] 잔고 조회 API 예외: {e}")
            return None

    def send_user_order(self, user_id, code, action, price, qty, reason):
        url = f"{BACKEND_API_URL}/trade/{user_id}"
        order_type = "BUY" if action == '매수' else "SELL"
        
        # [수정] numpy.int64 타입을 일반 int로 변환 (JSON 직렬화 오류 해결)
        try:
            qty = int(qty)
            price = int(price)
        except Exception:
            pass
        
        payload = {
            "stockCode": code,
            "orderType": order_type,
            "quantity": qty,
            "price": price,
            "triggerSource": "AI"
        }
        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=5)
            if res.status_code == 200:
                print(f"   ✅ [User {user_id}] 주문 접수 성공: {code} {action} {qty}주")
                return True
            else:
                print(f"   ❌ [User {user_id}] 주문 접수 실패: {res.text}")
                return False
        except Exception as e:
            print(f"   ⚠️ [User {user_id}] 주문 전송 중 오류: {e}")
            return False

    def analyze_stock(self, code):
        """
        단일 종목 AI 분석 (feature_store + model)
        """
        try:
            df = self.store.get_realtime_features(code)
            if df is None or df.empty: 
                return None

            # 예측 수행
            pred_score = self.model.predict(df)
            current_price = df['close'].iloc[-1]

            # (옵션) 밸류체인 점수 반영 등 추가 로직 가능

            return {
                'code': code,
                'name': get_stock_name(code),
                'ai_score': pred_score,
                'current_price': current_price
            }
        except Exception as e:
            print(f" [Error] {code} 분석 실패: {e}")  # 주석 해제!
            import traceback
            traceback.print_exc() # 상세 에러 위치 확인을 위해 추가 권장
            return None

    def process_user(self, user_id):
        """
        개별 사용자 포트폴리오 분석 및 매매 실행
        """
        print(f"\n >>> [User {user_id}] 포트폴리오 분석 시작")

        # 1. 사용자 상태 초기화 (메모리 관리)
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                'last_sell_times': {},
                'last_buy_times': {},
                'calculated_cash': None
            }
        state = self.user_states[user_id]

        # 2. 잔고 조회
        balance_data = self.get_user_balance(user_id)
        
        if not balance_data:
            print(f"     [Skip] 잔고 정보를 가져올 수 없어 건너뜁니다.")
            return

        # 3. 데이터 파싱
        # (API 응답 필드명에 따라 수정 필요: totalDeposit, depositReceived 등)
        output1 = balance_data.get('output1') or []  # 보유 주식 리스트
        output2 = balance_data.get('output2') or []  # 예수금/자산 정보 리스트

        # 1. 예수금 추출
        d2_cash = 0
        total_cash = 0
        
        if output2:
            balance_info = output2[0]
            # prvs_rcdl_excc_amt: 가수금제외 D+2 예수금 (실제 주문 가능 금액에 가까움)
            # dnca_tot_amt: 예수금 총액
            d2_cash = balance_info.get('prvs_rcdl_excc_amt', 0)
            total_cash = balance_info.get('dnca_tot_amt', 0)

        # 숫자 변환 헬퍼
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
        
        # 예수금 우선순위: D+2예수금 > 총예수금
        api_cash = d2_amt if (d2_amt is not None and d2_amt > 0) else total_amt

        # 예수금 동기화
        if state['calculated_cash'] is None:
            state['calculated_cash'] = api_cash
            final_cash = api_cash
        else:
            diff = api_cash - state['calculated_cash']
            BUG_THRESHOLD = 30000000 
            if diff > BUG_THRESHOLD:
                print(f"     [Defense] 예수금 급증 감지 (차이: {diff:,}원) -> 내부 계산값 사용")
                final_cash = state['calculated_cash']
            else:
                if diff > 0:
                    print(f"     [Sync] 예수금 변동 반영: +{diff:,}원")
                final_cash = api_cash
                state['calculated_cash'] = api_cash

        # 2. 보유종목 구조화 (KIS API 키 매핑: pdno, hldg_qty, pchs_avg_pric)
        my_holdings_detail = {}
        for h in output1:
            # pdno: 종목코드, hldg_qty: 보유수량, pchs_avg_pric: 매입평균가
            qty = _parse_amount(h.get('hldg_qty', 0))
            
            if qty > 0:
                code = h.get('pdno')
                avg_price = float(h.get('pchs_avg_pric', 0))
                
                my_holdings_detail[code] = {
                    'qty': qty,
                    'avg_price': avg_price,
                    'current_price': 0, # 이후 분석 단계에서 채워짐
                    'amt': 0
                }

        # 5. 분석 대상 유니버스 설정 (관심종목 + 보유종목)
        universe = set(self.target_codes) | set(my_holdings_detail.keys())

        # 6. AI 종목 분석
        ai_results = []
        for code in universe:
            data = self.analyze_stock(code)
            if data:
                # 보유종목이면 현재가 업데이트
                if code in my_holdings_detail:
                    price = data['current_price']
                    my_holdings_detail[code]['current_price'] = price
                    my_holdings_detail[code]['amt'] = price * my_holdings_detail[code]['qty']
                ai_results.append(data)

        if not ai_results:
            print("     [Info] 분석 가능한 데이터가 없어 종료합니다.")
            return

        # (옵션) 밸류체인 추가 분석
        if self.vc_strategy and self.vc_strategy.vc_analyzer:
            high_scorers = [res for res in ai_results if res['ai_score'] >= 80]
            expanded_codes = set()
            for item in high_scorers:
                main_code = item['code']
                related = self.vc_strategy.vc_analyzer.find_similar_stocks(main_code)
                for rel in related:
                    r_code = rel['code']
                    if r_code not in universe and r_code not in expanded_codes:
                        expanded_codes.add(r_code)

            if expanded_codes:
                print(f"     [ValueChain] 추가 종목 분석: {len(expanded_codes)}개")
                for r_code in expanded_codes:
                    data = self.analyze_stock(r_code)
                    if data:
                        ai_results.append(data)

        ai_scores_df = pd.DataFrame(ai_results)

        # 7. 총 자산 계산 (예수금 + 주식 평가금)
        total_stock_val = sum([h['amt'] for h in my_holdings_detail.values()])
        total_asset = final_cash + total_stock_val
        print(f"     [Asset] 총 자산: {total_asset:,}원 (예수금: {final_cash:,}원)")

        # 8. 리밸런싱 계획 수립
        plan_df = self.rebalancer.run_ai_rebalancing(
            current_holdings_detail=my_holdings_detail,
            ai_scores_df=ai_scores_df,
            total_budget=total_asset,
            last_sell_times=state['last_sell_times'],
            last_buy_times=state['last_buy_times']
        )

        if plan_df.empty:
            return

        print(f"     [Plan] 매매 계획 {len(plan_df)}건 감지")

        # 9. 주문 실행
        for _, row in plan_df.iterrows():
            action = row['action']
            if action == '유지': continue

            code = row['code']
            price = 0

            # 현재가 찾기
            found = [x for x in ai_results if x['code'] == code]
            if found:
                price = found[0]['current_price']
            elif code in my_holdings_detail:
                price = int(my_holdings_detail[code]['current_price'])

            if price <= 0:
                print(f"       [Skip] {code} 현재가 확인 불가")
                continue

            profit_rate = row.get('profit_rate', 0.0)
            reason = row['reason']

            # 매도 주문
            if action in ['비중축소', '전량매도']:
                amt_to_sell = abs(row['diff'])
                qty_to_sell = int(amt_to_sell // price)
                if qty_to_sell > 0:
                    success = self.send_user_order(user_id, code, action, price, qty_to_sell, reason)
                    if success:
                        self.save_trade_log(user_id, code, action, qty_to_sell, price, profit_rate, reason)
                        state['last_sell_times'][code] = datetime.now()

            # 매수 주문
            elif action == '매수':
                amt_to_buy = row['diff']
                # 안전 마진 (예수금의 95%까지만 사용)
                safe_cash = final_cash * 0.95

                if amt_to_buy > safe_cash:
                    if safe_cash > 0:
                        amt_to_buy = safe_cash
                    else:
                        print(f"       🚫 예수금 부족 ({code})")
                        continue

                qty_to_buy = int(amt_to_buy // price)
                if qty_to_buy > 0:
                    success = self.send_user_order(user_id, code, action, price, qty_to_buy, reason)
                    if success:
                        self.save_trade_log(user_id, code, action, qty_to_buy, price, profit_rate, reason)
                        state['last_buy_times'][code] = datetime.now()

                        # 예수금 차감 반영 (다음 종목 매수 위함)
                        used_cash = qty_to_buy * price
                        final_cash -= used_cash
                        state['calculated_cash'] = final_cash

    def run_cycle(self):
        print(f"\n[Cycle] {datetime.now().strftime('%H:%M:%S')} 전체 사용자 자동매매 시작")

        # 1. 활성 사용자 목록 조회
        users = self.get_active_users()
        print(f"[System] 활성 사용자 수: {len(users)}명")

        # 2. 유저별 순차 처리
        for user_id in users:
            try:
                self.process_user(user_id)
            except Exception as e:
                print(f" [Error] User {user_id} 처리 중 예외 발생: {e}")

        print("[Cycle] 전체 사용자 처리 완료")


if __name__ == "__main__":
    trader = AIAutoTrader()
    try:
        while True:
            trader.run_cycle()
            # 1분(60초) 대기 후 다음 사이클
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n [System] 자동매매 종료")