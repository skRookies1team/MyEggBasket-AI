import sys
import os
import re
import numpy as np
import pandas as pd

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 기존 모듈 재사용
from ai_pipeline.mapping.stock_mapping import get_stock_mentions
from ai_pipeline.nlp.sentiment import analyze_sentiment

class NewsAnalyzer:
    def __init__(self):
        print("📰 [NLP] 정밀 뉴스 분석기(Sentence-Level) 초기화 중...")

    def split_sentences(self, text):
        """
        [Step 1] 기사 본문을 문장 단위로 분리 (정규식 활용)
        """
        if not text: return []
        
        # 다 쓴 뒤에 온점(.), 물음표(?), 느낌표(!) 뒤에 공백이나 줄바꿈이 오면 자름
        # 예: "상승했다. 그러나" -> ["상승했다.", "그러나"]
        sentences = re.split(r'(?<=[.?!])\s+', text)
        
        # 너무 짧은 문장(5글자 미만)은 노이즈로 보고 제거
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def analyze_article(self, article_text):
        """
        [Core] 기사 하나를 통째로 분석하여 종목별 감성 지표를 산출
        """
        # 1. 문장 분리
        sentences = self.split_sentences(article_text)
        if not sentences: return {}

        # 2. 문장별 분석 결과 저장소
        # 구조: { '005930': [0.8, 0.9, -0.1], '035720': [-0.5] }
        stock_sentiments = {}
        
        # 디버깅용 상세 로그
        details = []

        for sent in sentences:
            # 3. [Filtering] 이 문장에 등장한 종목 찾기
            # get_stock_mentions 함수가 문장 내 종목 코드를 리스트로 반환한다고 가정
            found_stocks = get_stock_mentions(sent)
            
            if not found_stocks:
                continue # 종목 없는 문장은 쿨하게 버림 (시황, 광고 등)

            # 4. [Sentiment] 문장 감성 분석 (FinBERT)
            # analyze_sentiment는 리스트를 받으므로 [sent]로 감싸서 호출
            # 결과는 [0.95] 같은 리스트로 나옴
            scores = analyze_sentiment([sent])
            score = scores[0] if scores else 0.0

            # 5. 결과 저장 (종목별로 점수 모으기)
            for code in found_stocks:
                if code not in stock_sentiments:
                    stock_sentiments[code] = []
                stock_sentiments[code].append(score)
                
                details.append({
                    "sentence": sent[:50] + "...", # 로그용 말줄임
                    "code": code,
                    "score": score
                })

        # 6. [Aggregation] 종목별 최종 지표 산출
        final_results = {}
        
        for code, scores_list in stock_sentiments.items():
            if not scores_list: continue
            
            scores_np = np.array(scores_list)
            
            # (1) sentiment_score: 평균 감성 (지금 감정이 좋냐 나쁘냐)
            avg_score = np.mean(scores_np)
            
            # (2) sentiment_volatility: 감정 기복 (안정적이냐 불안하냐)
            # 문장이 1개면 변동성 0
            volatility = np.std(scores_np) if len(scores_np) > 1 else 0.0
            
            # (3) sentiment_trend: 감정 추세 (좋아지냐 나빠지냐)
            # (마지막 문장 점수 - 첫 문장 점수) -> 긍정적 결론이면 양수
            trend = 0.0
            if len(scores_np) > 1:
                # 단순 차이 or 선형회귀 기울기 사용 가능. 여기선 단순 차이
                trend = scores_np[-1] - scores_np[0]
            
            final_results[code] = {
                "sentiment_score": round(avg_score, 4),       # -1 ~ 1
                "sentiment_volatility": round(volatility, 4), # 0 ~ 1 (높을수록 의견 분분)
                "sentiment_trend": round(trend, 4),           # 양수면 개선, 음수면 악화
                "mention_count": len(scores_list)             # 언급 횟수 (신뢰도 지표)
            }

        return final_results, details

# ==========================================
# 테스트 실행 코드
# ==========================================
if __name__ == "__main__":
    analyzer = NewsAnalyzer()
    
    # 테스트용 기사 (가상)
    test_article = """
    오늘 국내 증시는 전반적으로 하락세였다.
    하지만 삼성전자는 3분기 실적 호조 소식에 장 초반 강세를 보였다. 외국인의 매수세가 유입되며 주가를 끌어올렸다.
    반면 SK하이닉스는 반도체 업황 둔화 우려로 약세를 면치 못했다. 장 막판에는 매도 물량이 쏟아지며 하락폭을 키웠다.
    삼성전자는 결국 2% 상승 마감하며 견조한 흐름을 유지했다.
    """
    
    print("\n🔍 기사 분석 시작...")
    results, logs = analyzer.analyze_article(test_article)
    
    print("\n📝 [문장별 상세 분석]")
    for log in logs:
        print(f"   - {log['code']}: {log['score']:.2f} | {log['sentence']}")
        
    print("\n📊 [최종 집계 결과]")
    for code, metrics in results.items():
        print(f"   [{code}]")
        print(f"     ✅ 감성 점수 (Score): {metrics['sentiment_score']} (높으면 긍정)")
        print(f"     🌊 변동성 (Volatility): {metrics['sentiment_volatility']} (높으면 불안정)")
        print(f"     📈 추세 (Trend): {metrics['sentiment_trend']} (양수면 긍정적 마무리)")
        print(f"     🗣️ 언급 횟수: {metrics['mention_count']}문장")