import sys
import os

# 경로 설정
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    # 사용자님이 지금 쓰고 있는 감성분석 함수 가져오기
    from ai_pipeline.nlp.sentiment import analyze_sentiment
except ImportError:
    print("❌ 경로 에러: ai_pipeline 폴더가 있는 위치에서 실행해주세요.")
    sys.exit()

def test_model():
    print("🧪 감성 분석 모델 테스트 시작\n")

    # 1. 확실한 긍정 문장
    text_pos = ["삼성전자 역대 최고 실적 달성, 주가 10% 급등. 분위기 아주 좋다."]
    score_pos = analyze_sentiment(text_pos)
    print(f"✅ 긍정 예문: {text_pos[0]}")
    print(f"   👉 결과 점수: {score_pos[0]} (예상: 0.5 이상 양수)")
    print("-" * 50)

    # 2. 확실한 부정 문장
    text_neg = ["SK하이닉스 적자 전환, 공장 가동 중단 위기. 주가 폭락."]
    score_neg = analyze_sentiment(text_neg)
    print(f"❌ 부정 예문: {text_neg[0]}")
    print(f"   👉 결과 점수: {score_neg[0]} (예상: -0.5 이하 음수)")
    print("-" * 50)

    # 3. 중립/애매한 문장
    text_neu = ["오늘 코스피 지수는 보합세로 마감했다."]
    score_neu = analyze_sentiment(text_neu)
    print(f"😐 중립 예문: {text_neu[0]}")
    print(f"   👉 결과 점수: {score_neu[0]} (예상: 0에 가까움)")
    print("-" * 50)

if __name__ == "__main__":
    test_model()