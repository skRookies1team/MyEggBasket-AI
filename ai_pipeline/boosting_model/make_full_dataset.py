import pandas as pd
import numpy as np
import torch
import os
import sys
import glob
from tqdm import tqdm

# 프로젝트 루트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from ai_pipeline.boosting_model.feature_expander import FeatureExpander
except ImportError:
    print("⚠️ FeatureExpander를 찾을 수 없습니다. 관련 피처가 누락될 수 있습니다.")
    FeatureExpander = None


# =========================================================
# GCN 로더
# =========================================================
class GCNFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cpu')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gcn_dir = os.path.abspath(os.path.join(current_dir, "../../data"))
        self.npy_path = os.path.join(self.gcn_dir, "gcn_embeddings.npy")
        self.csv_path = os.path.join(self.gcn_dir, "gcn_node_list.csv")
        self.mapping = {}
        if os.path.exists(self.npy_path) and os.path.exists(self.csv_path):
            try:
                node_df = pd.read_csv(self.csv_path, dtype=str)
                code_col = next((c for c in node_df.columns if c in ['code', 'stock_code', 'stck_shrn_iscd']), None)
                if code_col:
                    codes = node_df[code_col].tolist()
                    embs = np.load(self.npy_path)
                    for i, code in enumerate(codes):
                        self.mapping[str(code).strip().zfill(6)] = embs[i] if i < len(embs) else np.zeros(16)
                    print(f"✅ [GCN] 임베딩 로드 완료: {len(self.mapping)}개 종목")
            except Exception as e:
                print(f"⚠️ [GCN] 로드 실패: {e}")

    def get_features(self, code):
        code = str(code).strip().zfill(6)
        if code in self.mapping:
            emb = self.mapping[code]
            return {f'gcn_emb_{i}': val for i, val in enumerate(emb)}
        return {f'gcn_emb_{i}': 0.0 for i in range(16)}


# =========================================================
# ⚙️ 핵심: 학습용 피처 생성 함수 (RealtimeFeatureLoader 로직 이식)
# =========================================================
def calculate_base_features(df):
    """
    RealtimeFeatureLoader와 동일한 로직으로 기초 기술적 지표를 생성합니다.
    """
    # 필수 컬럼 확인 (close, volume)
    if 'close' not in df.columns:
        return df

    # 원본 보존을 위해 복사
    df = df.copy()

    # 1. 가격 변화율 (Returns)
    df['price_change_1'] = df['close'].pct_change(1).fillna(0)
    df['price_change_5'] = df['close'].pct_change(5).fillna(0)
    df['price_change_10'] = df['close'].pct_change(10).fillna(0)

    # 2. 이동평균 (MA)
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()

    # 3. 이격도 (Price vs MA)
    df['price_vs_ma5'] = (df['close'] - df['ma_5']) / (df['ma_5'] + 1e-8)
    df['price_vs_ma20'] = (df['close'] - df['ma_20']) / (df['ma_20'] + 1e-8)
    # feature_names.json에는 price_vs_ma60은 없고 price_vs_ma20까지만 있을 수도 있으나,
    # 코드상 안전을 위해 계산해둠 (모델이 안 쓰면 무시됨)

    # 4. 거래대금 변화율
    # 거래대금(acml_tr_pbmn)이 없다면 계산 (가격 * 거래량)
    if 'acml_tr_pbmn' not in df.columns:
        if 'volume' in df.columns:
            df['acml_tr_pbmn'] = df['close'] * df['volume']
        else:
            df['acml_tr_pbmn'] = 0

    df['tr_amount_change'] = df['acml_tr_pbmn'].pct_change(1).fillna(0)

    # 5. 변동성 (Volatility)
    df['volatility_5'] = df['price_change_1'].rolling(window=5).std().fillna(0)
    df['volatility_10'] = df['price_change_1'].rolling(window=10).std().fillna(0)

    # 6. 전일 대비 등락률 (prdy_ctrt) - 데이터에 없으면 계산
    if 'prdy_ctrt' not in df.columns:
        # 일별 종가(prev_close)를 구하기 어려우므로 임시로 0 처리하거나
        # 데이터에 이미 있는 경우가 많으므로 패스. 없다면 0.0
        df['prdy_ctrt'] = 0.0

    return df


# =========================================================
# 🚀 메인 데이터 생성 함수
# =========================================================
def generate_full_csv(data_dir, output_file):
    print(f"\n📂 데이터 디렉토리: {data_dir}")
    print(f"💾 저장 경로: {output_file}")

    csv_files = glob.glob(os.path.join(data_dir, "*_1Year.csv"))
    csv_files = [f for f in csv_files if "cached_final_features" not in f and "optimize.csv" not in f]

    if not csv_files:
        print("❌ 처리할 CSV 파일이 없습니다.")
        return

    expander = FeatureExpander() if FeatureExpander else None
    gcn_extractor = GCNFeatureExtractor()

    all_data = []

    for file_path in tqdm(csv_files, desc="데이터 처리 중"):
        try:
            filename = os.path.basename(file_path)
            code = filename.split('_')[0]

            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]

            # 컬럼명 통일
            rename_map = {
                'price': 'close',
                'stck_prpr': 'close',
                'acml_vol': 'volume',
                'vol': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)

            if 'close' not in df.columns: continue

            # Timestamp 생성
            if 'timestamp' not in df.columns and 'date' in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(
                    df['date'].astype(str) + df['time'].astype(str).str.zfill(6),
                    format='%Y%m%d%H%M%S', errors='coerce'
                )

            # 1. 기초 기술적 지표 생성 (price_change, volatility 등)
            df = calculate_base_features(df)

            # 2. 확장 지표 생성 (hist_...)
            if expander:
                if hasattr(expander, 'add_technical_indicators'):
                    df = expander.add_technical_indicators(df)

            # 3. GCN 임베딩 추가
            gcn_feats = gcn_extractor.get_features(code)
            for k, v in gcn_feats.items():
                df[k] = v

            # 4. Sentiment 피처 (없으면 0.0으로 채움 - 모델 에러 방지)
            sent_cols = ['sentiment_score', 'sentiment_volatility', 'sentiment_trend']
            for col in sent_cols:
                if col not in df.columns:
                    df[col] = 0.0

            df['code'] = code

            # AI Score 컬럼이 있다면 삭제 (새로 계산하기 위해)
            if 'ai_score' in df.columns:
                df.drop(columns=['ai_score'], inplace=True)

            # NaN 제거 (지표 계산 초반부 등)
            df.dropna(subset=['close', 'price_change_10', 'ma_60'], inplace=True)

            all_data.append(df)

        except Exception as e:
            print(f"❌ {filename} 에러: {e}")

    if all_data:
        print("\n📊 데이터 병합 중...")
        final_df = pd.concat(all_data, ignore_index=True)
        if 'timestamp' in final_df.columns:
            final_df.sort_values(by=['timestamp', 'code'], inplace=True)

        print(f"💾 CSV 파일 저장 중... ({len(final_df)} rows)")
        final_df.to_csv(output_file, index=False)
        print("✅ 작업 완료! 이제 optimize_params.py를 실행하면 모델이 정상적인 피처를 입력받습니다.")
    else:
        print("❌ 저장할 데이터가 없습니다.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    data_dir = os.path.join(project_root, "data")
    output_file = os.path.join(data_dir, "optimize.csv")

    generate_full_csv(data_dir, output_file)