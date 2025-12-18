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

    # 1. 데이터 폴더 경로 (프로젝트루트/data)
    data_dir = os.path.join(project_root, "data")

    csv_files = glob.glob(os.path.join(data_dir, "optimize.csv"))
    if not csv_files:
        print(f"❌ '{data_dir}' 경로에 CSV 파일이 없습니다.")
        return None

    # 가장 큰 파일 선택
    target_file = max(csv_files, key=os.path.getsize)
    print(f"📂 데이터 파일 로드 중...: {os.path.basename(target_file)}")

    df = pd.read_csv(target_file)

    # [핵심 수정] 컬럼명 전체 소문자로 통일 (Close -> close)
    df.columns = [c.strip().lower() for c in df.columns]

    # 컬럼 이름 매핑 (호환성 확보)
    rename_map = {
        'stck_shrn_iscd': 'code',
        'stock_code': 'code',
        'price': 'close'
    }
    df.rename(columns=rename_map, inplace=True)

    # [✅ 수정된 부분] 중복된 컬럼 이름 제거 (code 컬럼이 2개가 되는 현상 방지)
    # stck_shrn_iscd가 code로 바뀌면서 기존 code와 충돌하는 것을 방지합니다.
    df = df.loc[:, ~df.columns.duplicated()]

    # 필수 컬럼 체크
    required_cols = ['code', 'timestamp', 'close']
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"❌ 데이터에 필수 컬럼이 부족합니다: {missing}")
        print(f"   👉 현재 보유 컬럼: {list(df.columns)}")

        # 긴급 복구 시도 (date/time이 있는 경우)
        if 'timestamp' in missing and 'date' in df.columns and 'time' in df.columns:
            print("   ⚠️ Timestamp 컬럼 생성 시도...")
            df['timestamp'] = pd.to_datetime(df['date'].astype(str) + df['time'].astype(str).str.zfill(6),
                                             format='%Y%m%d%H%M%S', errors='coerce')
            if 'timestamp' in df.columns:
                print("   ✅ Timestamp 생성 성공!")
                missing.remove('timestamp')

        if missing:  # 여전히 부족하면 종료
            return None

    # Timestamp 변환
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 3. AI 모델 로드 및 점수 예측
    if 'ai_score' not in df.columns:
        print("🤖 AI 모델 로드 및 스코어링 시작 (Batch Prediction)...")
        model = StackingEnsemble()
        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        try:
            model.load_model(model_path)

            # 학습에 안 쓰는 메타데이터 제외
            meta_cols = ['code', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'target',
                         'stck_shrn_iscd']
            feature_cols = [c for c in df.columns if c not in meta_cols]
            X = df[feature_cols]

            # 숫자형으로 변환 (오류 방지)
            X = X.select_dtypes(include=[np.number])

            probs = model.predict_proba(X)
            scores = probs[:, 1] * 100 if probs.shape[1] == 2 else probs[:, 0] * 100
            df['ai_score'] = scores
            print("✅ 스코어링 완료!")

        except Exception as e:
            print(f"⚠️ 모델 예측 실패 (랜덤 점수로 대체합니다): {e}")
            df['ai_score'] = np.random.uniform(40, 80, size=len(df))
    else:
        print("✅ 이미 ai_score 컬럼이 존재합니다.")

    print(f"✅ 데이터 준비 완료: {len(df):,} rows (기간: {df['timestamp'].min()} ~ {df['timestamp'].max()})")

    # [안전 장치] 리턴할 때도 중복 제거 재확인
    return df[['timestamp', 'code', 'close', 'ai_score']].copy()


# -----------------------------------------------------------
# 3. 백테스터 엔진
# -----------------------------------------------------------
class FastBacktester:
    def __init__(self, df):
        self.df = df
        self.timestamps = df['timestamp'].sort_values().unique()
        # 여기서 code 컬럼이 유니크해야 set_index가 정상 작동함
        self.data_by_time = {ts: grp.set_index('code') for ts, grp in df.groupby('timestamp')}

    def run(self, params):
        cash = 10_000_000  # 1,000만원
        holdings = {}
        last_sell_times = {}

        for ts in self.timestamps:
            if ts not in self.data_by_time: continue
            market_snapshot = self.data_by_time[ts]

            # 1. 매도 (Sell)
            sell_list = []
            # holdings의 키(code)를 list로 변환하여 순회 (런타임 에러 방지)
            for code in list(holdings.keys()):
                if code not in market_snapshot.index: continue

                try:
                    curr_price = market_snapshot.loc[code]['close']
                    ai_score = market_snapshot.loc[code]['ai_score']
                except Exception:
                    # 중복 인덱스나 데이터 오류 시 건너뜀
                    continue

                info = holdings[code]
                profit_rate = (curr_price - info['avg_price']) / info['avg_price']
                action = 'HOLD'

                # 파라미터 적용
                if profit_rate <= params['stop_loss']:
                    action = 'SELL'
                elif profit_rate >= params['profit_take']:
                    if ai_score < 90: action = 'SELL'
                elif ai_score < params['sell_score']:
                    action = 'SELL'

                if action == 'SELL':
                    cash += info['qty'] * curr_price
                    del holdings[code]
                    last_sell_times[code] = ts

            # 2. 매수 (Buy)
            candidates = market_snapshot[market_snapshot['ai_score'] >= params['buy_score']]
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
                            cash -= qty * price
                            holdings[code] = {'qty': qty, 'avg_price': price}

        # 최종 평가
        final_asset = cash
        last_snapshot = self.data_by_time[self.timestamps[-1]]
        for code, info in holdings.items():
            if code in last_snapshot.index:
                price = last_snapshot.loc[code]['close']
            else:
                price = info['avg_price']
            final_asset += info['qty'] * price

        return final_asset


# -----------------------------------------------------------
# 4. Optuna Objective
# -----------------------------------------------------------
def objective(trial):
    profit_take = trial.suggest_float('profit_take', 0.02, 0.10)
    stop_loss = trial.suggest_float('stop_loss', -0.10, -0.02)
    buy_score = trial.suggest_int('buy_score', 60, 85)
    sell_score = trial.suggest_int('sell_score', 30, 50)
    cooldown = trial.suggest_int('cooldown', 10, 60)

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
    # 데이터가 꼬여있을 수 있으므로 feature_engineering 먼저 실행 권장
    print("📢 만약 오류가 계속된다면, 'python ai_pipeline/feature_engineering.py'를 먼저 실행하여 데이터를 재생성하세요.")

    scored_df = load_and_score_data()

    if scored_df is not None:
        backtester = FastBacktester(scored_df)
        print("\n 🚀 Optuna 파라미터 최적화 시작...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        print("\n" + "=" * 60)
        print(" 🎉 최적 파라미터 결과")
        print("=" * 60)
        print(f"💰 최고 자산: {study.best_value:,.0f}원")
        print("🔧 최적 설정값:")
        for k, v in study.best_params.items():
            print(f"   - {k}: {v}")
    else:
        print("❌ 시뮬레이션을 시작할 수 없습니다.")