import sys
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# [주석 처리] FeatureEngineer 사용 안 함
# from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

# [필수] 모델 로드용 클래스
from ai_pipeline.boosting_model.train import StackingEnsemble

# [New] 실시간 API 데이터 스토어
from ai_pipeline.feature_store import OnlineFeatureStore
from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy

def run_prediction_pipeline(csv_path=None):
    print(" [Prediction] Inference Mode 시작 (No Training, No Raw Data)")

    # 1. 모델 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")

    if not os.path.exists(os.path.join(model_dir, 'meta_model.pkl')):
        print(" [Error] 학습된 모델 파일(.pkl)이 없습니다.")
        return None

    model = StackingEnsemble()
    try:
        model.load_model(model_dir)
        print(" [Model] 저장된 모델 로드 완료")
    except Exception as e:
        print(f" [Error] 모델 로드 실패: {e}")
        return None

    # ------------------------------------------------------------------
    # 2. 데이터 준비 (CSV 방식 -> API 방식으로 변경)
    # ------------------------------------------------------------------
    
    # [주석 처리] 기존 로컬 파일 기반 방식
    # if csv_path is None:
    #     project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    #     data_dir = os.path.join(project_root, "data", "krx_data")
    #     engineer = FeatureEngineer(data_dir=data_dir)
    # else:
    #     engineer = FeatureEngineer(csv_path=csv_path)

    # [New] 실시간 API 데이터 스토어 연결
    store = OnlineFeatureStore()
    
    # 예측할 주요 종목 리스트 (필요한 종목들을 여기에 넣어주세요)
    target_codes = [
        "005930", "000660", "207940", "005380", "000270", "055550", "105560", "068270", "015760",
        "028260", "032830", "012330", "035420", "006400", "086790", "006405", "000810", "010140",
        "064350", "138040", "051910", "010130", "009540", "267260", "066570", "066575", "033780",
        "003550", "003555", "310200", "034020", "012450", "009830", "011070", "071050", "081660",
        "046890", "323410", "017670", "010620", "047050", "009155", "275630", "009835", "001440",
        "138930", "175330", "051900", "005490", "034220"
    ]

    # ------------------------------------------------------------------
    # 3. 피처 생성 및 예측
    # ------------------------------------------------------------------
    print(f" [Data] OnlineFeatureStore를 통해 {len(target_codes)}개 종목 분석 중... (멀티스레드)")

    # [내부 함수] 개별 종목 처리 로직 (스레드에서 실행될 함수)
    def predict_single_stock(code):
        try:
            # 1) 실시간 피처 가져오기 (Network I/O)
            features = store.get_realtime_features(code)
            
            if features is None or features.empty:
                return None

            # 2) 모델 예측 (CPU)
            probs = model.predict_proba(features)
            
            # 차원 확인
            if hasattr(probs, 'ndim') and probs.ndim == 2:
                score = probs[0, 1] * 100
            else:
                score = probs[1] * 100
            
            return {
                'code': str(code).zfill(6),
                'ai_score': round(score, 2)
            }
        except Exception as e:
            # 개별 종목 에러는 로그만 찍고 넘어감
            # print(f" [Warning] {code} 처리 실패: {e}")
            return None

    predictions = []

    # [핵심 변경] ThreadPoolExecutor를 사용한 병렬 처리
    # max_workers=10 : 동시에 10개씩 요청을 보냄 (API 제한에 따라 조절 가능)
    with ThreadPoolExecutor(max_workers=10) as executor:
        # map을 사용하면 target_codes 순서대로 작업이 할당됨
        results = executor.map(predict_single_stock, target_codes)
        
        # 결과 수집 (None이 아닌 것만)
        for res in results:
            if res is not None:
                predictions.append(res)

    if not predictions:
        print(" [Warning] 예측 결과가 없습니다. (API 연결 상태나 장 운영시간 확인 필요)")
        return None

    # 4. 결과 생성 (DataFrame 변환)
    result_df = pd.DataFrame(predictions)
    
    # 강력매수/매수/관망 의견 달기
    conditions = [
        (result_df['ai_score'] >= 90),
        (result_df['ai_score'] >= 70),
        (result_df['ai_score'] <= 40)
    ]
    choices = ['강력매수', '매수', '매도/관망']
    result_df['opinion'] = np.select(conditions, choices, default='중립')

    print(f" [Done] 총 {len(result_df)}개 종목 예측 완료")
    return result_df

if __name__ == "__main__":
    df = run_prediction_pipeline()
    # (하단 main 실행 코드는 그대로 두셔도 됩니다)