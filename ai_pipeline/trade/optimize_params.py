import sys
import os
import glob
import pandas as pd
import numpy as np
import optuna
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm

# -----------------------------------------------------------
# 1. 프로젝트 경로 설정 및 모듈 임포트
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from ai_pipeline.boosting_model.train import StackingEnsemble
except ImportError:
    pass


# -----------------------------------------------------------
# 2. 데이터 준비 (통합 파일 로드 및 AI 점수 산출)
# -----------------------------------------------------------
def load_and_score_data():
    print("\n" + "=" * 60)
    print(" 🛠️ [Step 1] 통합 데이터 로드 및 AI 점수 예측")
    print("=" * 60)

    # 1. 데이터 폴더 경로
    data_dir = os.path.join(project_root, "data")
    csv_files = glob.glob(os.path.join(data_dir, "optimize.csv"))

    if not csv_files:
        print(f"❌ optimize.csv 파일이 없습니다. make_full_dataset.py를 먼저 실행하세요.")
        return None

    target_file = csv_files[0]
    print(f"📂 데이터 파일 로드 중...: {os.path.basename(target_file)}")

    df = pd.read_csv(target_file)

    # 컬럼명 소문자 통일 및 공백 제거
    df.columns = [c.strip().lower() for c in df.columns]

    # 컬럼 이름 매핑
    rename_map = {'stck_shrn_iscd': 'code', 'stock_code': 'code', 'price': 'close'}
    df.rename(columns=rename_map, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]

    # 필수 컬럼 체크
    required_cols = ['code', 'timestamp', 'close']
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        if 'timestamp' in missing and 'date' in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'].astype(str) + df['time'].astype(str).str.zfill(6),
                                             format='%Y%m%d%H%M%S', errors='coerce')
            if 'timestamp' in df.columns:
                missing.remove('timestamp')

        if missing:
            print(f"❌ 필수 컬럼 부족: {missing}")
            return None

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # -------------------------------------------------------
    # [핵심 수정 1] 중복 데이터 제거 (Timestamp + Code 기준)
    # -------------------------------------------------------
    before_len = len(df)
    df.drop_duplicates(subset=['timestamp', 'code'], inplace=True)
    after_len = len(df)
    if before_len != after_len:
        print(f"⚠️ 중복 데이터 {before_len - after_len:,}개를 제거했습니다.")

    # -------------------------------------------------------
    # 3. AI 점수 생성
    # -------------------------------------------------------
    if 'ai_score' not in df.columns:
        print("🤖 AI 모델 로드 및 스코어링 시작...")
        model = StackingEnsemble()
        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")

        try:
            model.load_model(model_path)

            meta_cols = ['code', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'target',
                         'stck_shrn_iscd']
            feature_cols = [c for c in df.columns if c not in meta_cols]
            X = df[feature_cols].select_dtypes(include=[np.number])

            if X.empty: raise ValueError("입력 피처(X)가 비어있습니다.")

            probs = model.predict_proba(X)
            # 확률값(0~1)을 100점 만점으로 변환
            scores = probs[:, 1] * 100 if probs.shape[1] == 2 else probs[:, 0] * 100
            df['ai_score'] = scores

        except Exception as e:
            print(f"⚠️ 모델 로드/예측 실패: {e}")
            print("🎲 [Fallback] 랜덤 점수(40~95)를 부여합니다.")
            df['ai_score'] = np.random.uniform(40, 95, size=len(df))
    else:
        print("✅ 이미 ai_score 컬럼이 존재합니다.")

    # 점수 스케일링 (변별력 확보)
    min_score = df['ai_score'].min()
    max_score = df['ai_score'].max()
    print(f"📊 원본 점수 범위: {min_score:.2f} ~ {max_score:.2f}")

    if max_score < 20 or (max_score - min_score) < 5:
        print("⚠️ 점수 변별력 부족 -> 0~100점 스케일링 적용")
        if max_score - min_score == 0:
            df['ai_score'] = 50
        else:
            df['ai_score'] = (df['ai_score'] - min_score) / (max_score - min_score) * 100
        print(f"✅ 변환 후 점수 범위: {df['ai_score'].min():.1f} ~ {df['ai_score'].max():.1f}")

    return df[['timestamp', 'code', 'close', 'ai_score']].copy()


# -----------------------------------------------------------
# 3. 백테스터 엔진
# -----------------------------------------------------------
class FastBacktester:
    def __init__(self, df):
        self.df = df
        self.timestamps = df['timestamp'].sort_values().unique()
        self.data_by_time = {ts: grp.set_index('code') for ts, grp in df.groupby('timestamp')}

    def run(self, params):
        cash = 10_000_000
        holdings = {}
        last_sell_times = {}
        trade_count = 0

        for ts in self.timestamps:
            if ts not in self.data_by_time: continue
            market_snapshot = self.data_by_time[ts]

            # 1. 매도 (Sell)
            current_codes = list(holdings.keys())
            for code in current_codes:
                if code not in market_snapshot.index: continue

                try:
                    # [핵심 수정 2] 중복 데이터 방어 코드 (Scalar 추출)
                    row = market_snapshot.loc[code]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]  # 중복이면 첫 번째 행 선택

                    curr_price = row['close']
                    ai_score = row['ai_score']
                except Exception:
                    continue

                info = holdings[code]
                if info['avg_price'] == 0: continue  # 0 나누기 방지

                profit_rate = (curr_price - info['avg_price']) / info['avg_price']

                should_sell = False
                if profit_rate <= params['stop_loss']:
                    should_sell = True
                elif profit_rate >= params['profit_take']:
                    if ai_score < 90: should_sell = True
                elif ai_score < params['sell_score']:
                    should_sell = True

                if should_sell:
                    # 매도 금액에서 수수료 0.2% 차감 (0.998을 곱함)
                    sell_amount = info['qty'] * curr_price * 0.998
                    cash += sell_amount

                    del holdings[code]
                    last_sell_times[code] = ts
                    trade_count += 1

            # 2. 매수 (Buy)
            # 조건에 맞는 후보 추출
            candidates = market_snapshot[market_snapshot['ai_score'] >= params['buy_score']]

            # [추가] candidates가 중복된 인덱스를 가질 경우 대비
            if candidates.index.duplicated().any():
                candidates = candidates[~candidates.index.duplicated(keep='first')]

            buy_list = []
            for code, row in candidates.iterrows():
                if code in holdings: continue
                if code in last_sell_times:
                    if (ts - last_sell_times[code]).total_seconds() / 60 < params['cooldown']:
                        continue
                buy_list.append((code, row['close'], row['ai_score']))

            if buy_list and cash > 0:
                buy_list.sort(key=lambda x: x[2], reverse=True)
                slot_count = 5 - len(holdings)

                if slot_count > 0:
                    budget = cash / slot_count
                    for i in range(min(len(buy_list), slot_count)):
                        code, price, _ = buy_list[i]
                        if price <= 0: continue

                        qty = int(budget // price)
                        if qty > 0:
                            buy_amount = qty * price * 1.002  # 살 때는 0.2% 더 비싸게 샀다고 가정
                            if cash >= buy_amount:
                                cash -= buy_amount
                                holdings[code] = {'qty': qty, 'avg_price': price}
                                trade_count += 1

        if trade_count == 0:
            return 0

        final_asset = cash
        last_snapshot = self.data_by_time[self.timestamps[-1]]

        for code, info in holdings.items():
            price = info['avg_price']  # 기본값
            if code in last_snapshot.index:
                row = last_snapshot.loc[code]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                price = row['close']

            final_asset += info['qty'] * price

        # 수익률 계산
        return_rate = (final_asset - 10_000_000) / 10_000_000 * 100

        # (옵션) 로그가 너무 많으면, 수익률이 -50% 이하일 때만 출력
        if return_rate < -50:
            print(f"💀 [폭망] 거래횟수: {trade_count}회 | 최종자산: {final_asset:,.0f}원 | 수익률: {return_rate:.2f}%")
        elif return_rate > 0:
            print(f"🎉 [수익] 거래횟수: {trade_count}회 | 최종자산: {final_asset:,.0f}원 | 수익률: {return_rate:.2f}%")
        return final_asset


# -----------------------------------------------------------
# 4. Optuna Objective
# -----------------------------------------------------------
def objective(trial):
    profit_take = trial.suggest_float('profit_take', 0.02, 0.15)
    stop_loss = trial.suggest_float('stop_loss', -0.15, -0.02)
    buy_score = trial.suggest_int('buy_score', 50, 90)
    sell_score = trial.suggest_int('sell_score', 30, 60)
    cooldown = trial.suggest_int('cooldown', 10, 120)

    params = {
        'profit_take': profit_take,
        'stop_loss': stop_loss,
        'buy_score': buy_score,
        'sell_score': sell_score,
        'cooldown': cooldown
    }
    return backtester.run(params)


# -----------------------------------------------------------
# 5. 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    scored_df = load_and_score_data()

    if scored_df is not None:
        backtester = FastBacktester(scored_df)

        print("\n 🚀 Optuna 파라미터 최적화 시작...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        print("\n" + "=" * 60)
        print(" 🎉 최적 파라미터 결과")
        print("=" * 60)

        if study.best_value == 0:
            print("⚠️ 경고: 모든 시도에서 거래가 발생하지 않았습니다.")
        else:
            print(f"💰 최고 자산: {study.best_value:,.0f}원")
            print("🔧 최적 설정값:")
            for k, v in study.best_params.items():
                print(f"   - {k}: {v}")

            # [추가] 최적 파라미터 저장
            import json

            with open('best_params.json', 'w') as f:
                json.dump(study.best_params, f, indent=4)
            print("💾 best_params.json 저장 완료")
    else:
        print("❌ 데이터 로드 실패")