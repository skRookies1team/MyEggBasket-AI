import pandas as pd
import numpy as np

class PortfolioRebalancer:
    """
    포트폴리오 리밸런싱 계산기 (고정 비율 전략)
    """
    
    def __init__(self):
        pass

    def calculate_rebalancing(self, current_holdings, target_ratios):
        """
        사용자가 정한 목표 비율(target_ratios)대로 매수/매도 금액을 계산합니다.
        (Constant Mix Strategy)
        
        Parameters:
        - current_holdings: 현재 보유 금액 (Dictionary) 
          예: {'005930': 5000000, '000660': 4000000}
        - target_ratios: 목표 비중 (Dictionary, 합은 1.0 권장)
          예: {'005930': 0.5, '000660': 0.3, '005380': 0.2}
        
        Returns:
        - DataFrame: 종목별 매수/매도 제안서
        """
        
        # 1. 총 자산 계산
        total_value = sum(current_holdings.values())
        
        # 2. 목표 비중 검증 (합이 1이 아니면 정규화하거나 경고)
        ratio_sum = sum(target_ratios.values())
        if abs(ratio_sum - 1.0) > 0.01:
            print(f"⚠️ 목표 비중의 합({ratio_sum:.2f})이 100%가 아닙니다. 비율대로 자동 조정합니다.")
        
        print(f"\n💰 [리밸런싱 분석] 총 자산: {total_value:,.0f}원")
        print("-" * 60)
        
        result_list = []
        
        # 보유 중이거나 목표에 있는 모든 종목 코드 확인
        all_codes = set(current_holdings.keys()) | set(target_ratios.keys())
        
        for code in all_codes:
            current_amt = current_holdings.get(code, 0)
            target_ratio = target_ratios.get(code, 0.0)
            
            # 비율 정규화 (합이 1이 아닐 경우 대비)
            if ratio_sum != 0:
                normalized_ratio = target_ratio / ratio_sum
            else:
                normalized_ratio = 0
            
            # 목표 금액 = 총 자산 * 목표 비중
            target_amt = total_value * normalized_ratio
            
            # 매매 필요 금액 (차액)
            diff_amt = target_amt - current_amt
            
            # 매매 신호 결정
            action = "HOLD"  # 유지
            if diff_amt > 1000: # 1000원 미만 차이는 무시 (수수료 등 고려)
                action = "BUY"
            elif diff_amt < -1000:
                action = "SELL"
            
            result_list.append({
                '종목코드': code,
                '현재금액': int(current_amt),
                '목표비중': f"{normalized_ratio*100:.1f}%",
                '목표금액': int(target_amt),
                '차액': int(diff_amt),
                '매매제안': action
            })
            
        # 데이터프레임 변환 및 정렬
        df_result = pd.DataFrame(result_list)
        
        # 보기 좋게 컬럼 순서 정렬
        cols = ['종목코드', '현재금액', '목표비중', '목표금액', '차액', '매매제안']
        return df_result[cols]

# ==========================================
# 테스트 실행 코드 (사용 예시)
# ==========================================
if __name__ == "__main__":
    rebalancer = PortfolioRebalancer()
    
    # [상황 설정]
    # 내 계좌: 총 1,000만원 (삼성 500, 하이닉스 400, 현대차 100)
    my_portfolio = {
        '005930': 5000000, # 삼성전자
        '000660': 4000000, # SK하이닉스
        '005380': 1000000  # 현대차
    }
    
    # [목표] 5 : 3 : 2 비율로 맞추고 싶다!
    my_target = {
        '005930': 0.5, # 50%
        '000660': 0.3, # 30%
        '005380': 0.2  # 20%
    }
    
    # 실행
    df = rebalancer.calculate_rebalancing(my_portfolio, my_target)
    
    # 결과 출력
    print(df.to_string(index=False))
    
    print("-" * 60)
    print("👉 해석:")
    for _, row in df.iterrows():
        action = row['매매제안']
        if action == "BUY":
            print(f"   [{row['종목코드']}] 비중이 부족합니다. {abs(row['차액']):,.0f}원 만큼 더 사세요 (매수)")
        elif action == "SELL":
            print(f"   [{row['종목코드']}] 비중이 너무 높습니다. {abs(row['차액']):,.0f}원 만큼 파세요 (매도)")
        else:
            print(f"   [{row['종목코드']}] 목표 비중과 일치합니다. (유지)")