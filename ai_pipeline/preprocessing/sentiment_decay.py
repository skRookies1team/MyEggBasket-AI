import sys
import os
from datetime import datetime
import math

def calculate_time_weighted_sentiment(
    news_items: list, 
    base_time_str: str, 
    decay_rate: float = 0.9,
    time_unit: str = 'hour'
) -> float:
    """
    뉴스 리스트의 감성 점수에 '지수 감쇠(Exponential Decay)'를 적용하여 가중 평균을 계산합니다.
    
    Args:
        news_items (list): 뉴스 딕셔너리 리스트 
                           예: [{'sentiment_score': 0.5, 'published_at': '2025-12-09 10:00:00'}, ...]
        base_time_str (str): 기준 시간 (보통 현재 시간 또는 장 마감 시간). 형식: 'YYYY-MM-DD HH:MM:SS'
        decay_rate (float): 감쇠율 (0~1 사이). 
                            - 0.9: 1단위 시간당 영향력이 90%로 감소 (완만함)
                            - 0.5: 1단위 시간당 영향력이 50%로 급감 (민감함)
        time_unit (str): 시간 단위 ('hour' 또는 'day'). 기본값 'hour'.

    Returns:
        float: 시간 가중치가 적용된 최종 감성 점수 (-1.0 ~ 1.0)
    """
    
    # 1. 뉴스 데이터가 없으면 0점 반환
    if not news_items:
        return 0.0

    try:
        base_time = datetime.strptime(base_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f" [Error] 날짜 형식이 잘못되었습니다: {base_time_str}")
        return 0.0

    weighted_score_sum = 0.0
    total_weight = 0.0

    for news in news_items:
        try:
            # 딕셔너리 키값은 프로젝트 상황에 맞게 수정 가능 (score, published_at 등)
            score = news.get('sentiment_score', 0.0) 
            pub_date_str = news.get('published_at', base_time_str)
            
            pub_time = datetime.strptime(pub_date_str, "%Y-%m-%d %H:%M:%S")
            
            # 2. 시간 차이 계산 (초 단위 -> 시간/일 단위 변환)
            time_diff_seconds = (base_time - pub_time).total_seconds()
            
            if time_unit == 'day':
                elapsed_time = time_diff_seconds / (3600 * 24)
            else: # hour
                elapsed_time = time_diff_seconds / 3600
            
            # 미래 데이터(버그 방지)는 0으로 처리
            if elapsed_time < 0:
                elapsed_time = 0

            # 3. 지수 감쇠 가중치 계산 (Weight = Decay ^ Time)
            # 예: 0.9의 2승 = 0.81 (2시간 전 뉴스는 81%만 반영)
            weight = math.pow(decay_rate, elapsed_time)

            weighted_score_sum += score * weight
            total_weight += weight

        except Exception as e:
            print(f" 개별 뉴스 처리 중 오류 발생: {e}")
            continue

    # 4. 가중 평균 계산
    if total_weight == 0:
        return 0.0
        
    final_score = weighted_score_sum / total_weight
    return round(final_score, 4)

# ==========================================
# 테스트 코드 (이 파일을 직접 실행할 때만 작동)
# ==========================================
if __name__ == "__main__":
    # 상황: 현재 시각 15시
    current_time = "2025-12-09 15:00:00"
    
    test_news = [
        # 1. 방금 나온 호재 (15:00) -> 가중치 1.0 (가장 강력)
        {'sentiment_score': 0.8, 'published_at': "2025-12-09 15:00:00"},
        
        # 2. 2시간 전 악재 (13:00) -> 가중치 0.81 (0.9^2)
        {'sentiment_score': -0.7, 'published_at': "2025-12-09 13:00:00"},
        
        # 3. 어제 나온 호재 (24시간 전) -> 가중치 0.07 (거의 무시됨)
        {'sentiment_score': 0.5, 'published_at': "2025-12-08 15:00:00"},
    ]
    
    result = calculate_time_weighted_sentiment(test_news, current_time, decay_rate=0.9)
    
    print(f" 기준 시간: {current_time}")
    print(f" 계산된 최종 점수: {result}")
    
    # 단순 평균이었다면? (0.8 - 0.7 + 0.5) / 3 = 0.2
    # 시간 가중 평균은? 더 최신 뉴스인 0.8의 영향이 커서 점수가 더 높게 나올 것임.