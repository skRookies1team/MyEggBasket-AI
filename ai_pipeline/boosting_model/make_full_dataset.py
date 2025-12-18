import pandas as pd
import numpy as np
import torch
import os
import sys
import glob
from datetime import datetime
from tqdm import tqdm  # 진행상황 표시

# 프로젝트 루트 경로 설정 (환경에 맞게 조정 필요)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# =========================================================
# 필요한 모듈 Import (없으면 패스하거나 더미 처리)
# =========================================================
try:
    from ai_pipeline.boosting_model.feature_expander import FeatureExpander
except ImportError:
    print("⚠️ FeatureExpander를 찾을 수 없습니다. 기술적 지표 생성이 건너뛰어질 수 있습니다.")
    FeatureExpander = None

try:
    from ai_pipeline.gcn_model.model import get_gae_model
except ImportError:
    get_gae_model = None


# =========================================================
# GCN 로더 (임베딩 추가용)
# =========================================================
class GCNFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cpu')  # 데이터 생성용이므로 CPU로 충분
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gcn_dir = os.path.abspath(os.path.join(current_dir, "../../data"))

        self.npy_path = os.path.join(self.gcn_dir, "gcn_embeddings.npy")
        self.csv_path = os.path.join(self.gcn_dir, "gcn_node_list.csv")
        self.mapping = {}

        if os.path.exists(self.npy_path) and os.path.exists(self.csv_path):
            try:
                node_df = pd.read_csv(self.csv_path, dtype=str)
                # 컬럼명 처리
                code_col = next((c for c in node_df.columns if c in ['code', 'stock_code', 'stck_shrn_iscd']), None)

                if code_col:
                    codes = node_df[code_col].tolist()
                    embs = np.load(self.npy_path)
                    for i, code in enumerate(codes):
                        code_str = str(code).strip().zfill(6)
                        if i < len(embs):
                            self.mapping[code_str] = embs[i]
                    print(f"✅ [GCN] 임베딩 로드 완료: {len(self.mapping)}개 종목")
            except Exception as e:
                print(f"⚠️ [GCN] 로드 실패: {e}")

    def get_features(self, code):
        # 해당 종목 코드에 대한 임베딩 벡터 반환 (딕셔너리 형태)
        code = str(code).strip().zfill(6)
        if code in self.mapping:
            emb = self.mapping[code]
            return {f'gcn_emb_{i}': val for i, val in enumerate(emb)}
        return {}


# =========================================================
# 🚀 메인 데이터 생성 함수
# =========================================================
def generate_full_csv(data_dir, output_file):
    print(f"\n📂 데이터 디렉토리: {data_dir}")
    print(f"💾 저장 경로: {output_file}")

    # 1. 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(data_dir, "*_1Year.csv"))
    # 캐시 파일이 섞여있으면 제외
    csv_files = [f for f in csv_files if "cached_final_features" not in f]

    if not csv_files:
        print("❌ 처리할 CSV 파일이 없습니다 (*_1Year.csv).")
        return

    # 2. 객체 초기화
    expander = FeatureExpander() if FeatureExpander else None
    gcn_extractor = GCNFeatureExtractor()

    all_data = []

    # 3. 파일 순회 및 처리
    for file_path in tqdm(csv_files, desc="데이터 처리 중"):
        try:
            # 파일명에서 종목코드 추출 (예: 005930_1Year.csv -> 005930)
            filename = os.path.basename(file_path)
            code = filename.split('_')[0]

            # CSV 로드
            df = pd.read_csv(file_path)

            # --- [핵심] 컬럼명 전처리 ---
            # 1. 공백 제거 및 원본 컬럼명 보존
            df.columns = [c.strip() for c in df.columns]

            # 2. Close -> close 강제 변환
            rename_map = {}
            for col in df.columns:
                if col.lower() == 'close':
                    rename_map[col] = 'close'  # 무조건 소문자 close로
                elif col.lower() == 'price':
                    rename_map[col] = 'close'

            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            # 필수 컬럼 'close' 확인
            if 'close' not in df.columns:
                # 닫는 가격이 없으면 데이터 가치가 없으므로 스킵하거나 로그 남김
                print(f"⚠️ {filename}: 'close' 컬럼 없음. 스킵.")
                continue

            # 3. Timestamp 생성 (없을 경우)
            if 'timestamp' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    # date, time이 문자열이나 정수형일 수 있으므로 str 변환 후 처리
                    df['timestamp'] = pd.to_datetime(
                        df['date'].astype(str) + df['time'].astype(str).str.zfill(6),
                        format='%Y%m%d%H%M%S',
                        errors='coerce'
                    )
                else:
                    # 임시 시간 생성 (필요하다면)
                    pass

            # --- [피처 엔지니어링] ---
            # 기술적 지표 추가
            if expander:
                # expander가 내부적으로 컬럼을 drop하지 않도록 주의 (보통 add_technical_indicators는 추가만 함)
                if hasattr(expander, 'add_technical_indicators'):
                    df = expander.add_technical_indicators(df)
                elif hasattr(expander, 'expand'):
                    df = expander.expand(df)

            # GCN 임베딩 추가 (모든 행에 동일한 값 부여)
            gcn_feats = gcn_extractor.get_features(code)
            if gcn_feats:
                for k, v in gcn_feats.items():
                    df[k] = v

            # 종목 코드 컬럼 추가
            df['stck_shrn_iscd'] = code
            df['code'] = code  # 편의상 둘 다 둠

            # 타겟 변수 생성 (필요 시)
            # df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

            # --- [중요] 데이터 정제 최소화 ---
            # NaN이 있는 행만 제거 (지표 계산 초반부 등)
            # 특정 컬럼을 drop하거나 숫자형만 남기는 로직은 제거함!
            df.dropna(inplace=True)

            all_data.append(df)

        except Exception as e:
            print(f"❌ {filename} 에러: {e}")

    # 4. 병합 및 저장
    if all_data:
        print("\n📊 데이터 병합 중...")
        final_df = pd.concat(all_data, ignore_index=True)

        # 날짜순 정렬 (timestamp가 있다면)
        if 'timestamp' in final_df.columns:
            final_df.sort_values(by=['timestamp', 'code'], inplace=True)

        print(f"💾 CSV 파일 저장 중... ({len(final_df)} rows)")

        # 인덱스는 저장하지 않음, 헤더는 저장
        final_df.to_csv(output_file, index=False)

        print("✅ 작업 완료!")
        print(f"   - 파일 위치: {output_file}")
        print(f"   - 컬럼 목록 (Total {len(final_df.columns)}): {list(final_df.columns)[:10]} ...")

        # 검증
        if 'close' in final_df.columns:
            print("   - ✅ 'close' 컬럼 존재함")
        else:
            print("   - 🚨 'close' 컬럼이 보이지 않습니다!")

    else:
        print("❌ 저장할 데이터가 없습니다.")


if __name__ == "__main__":
    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))

    # 데이터 폴더 경로 (사용자 환경에 맞게 수정)
    data_dir = os.path.join(project_root, "data")

    # 결과 파일명
    output_file = os.path.join(data_dir, "optimize.csv")

    generate_full_csv(data_dir, output_file)