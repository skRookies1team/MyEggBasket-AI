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
        if ratio_sum == 0:
            print("⚠️ 목표 비중 합계가 0입니다. 현금 보유를 권장합니다.")
            return pd.DataFrame()

        print(f"\n💰 [리밸런싱] 총 자산: {total_value:,.0f}원 (목표비중합: {ratio_sum:.2f})")
        
        result_list = []
        all_codes = set(current_holdings.keys()) | set(target_ratios.keys())
        
        for code in all_codes:
            current_amt = current_holdings.get(code, 0)
            raw_ratio = target_ratios.get(code, 0.0)
            
            # 비중 정규화 (전체 합을 1로 맞춤)
            target_ratio = raw_ratio / ratio_sum
            target_amt = total_value * target_ratio
            diff_amt = target_amt - current_amt
            
            action = "유지"
            if diff_amt > 1000: action = "매수"
            elif diff_amt < -1000: action = "매도"
            
            result_list.append({
                '종목코드': code,
                '현재금액': int(current_amt),
                '목표비중': f"{target_ratio*100:.1f}%",
                '목표금액': int(target_amt),
                '차액': int(diff_amt),
                '매매제안': action
            })
            
        df_result = pd.DataFrame(result_list)
        return df_result[['종목코드', '현재금액', '목표비중', '목표금액', '차액', '매매제안']]

    def run_ai_rebalancing(self, current_holdings, ai_scores_df):
        """
        [Case 2] AI 점수 기반 리밸런싱
        AI 점수가 높을수록 비중을 높게 가져갑니다.
        
        - current_holdings: {'005930': 5000000, ...}
        - ai_scores_df: analyzer.get_ai_scores()의 결과 (code, ai_score)
        """
        print("\n🤖 [AI 전략] AI 점수 기반 비중 산출 중...")
        
        if ai_scores_df is None or ai_scores_df.empty:
            print("❌ AI 점수 데이터가 없습니다.")
            return None

        # 보유 종목들의 AI 점수만 추출
        my_codes = list(current_holdings.keys())
        target_df = ai_scores_df[ai_scores_df['code'].isin(my_codes)].copy()
        
        # 만약 점수 데이터에 내 종목이 없으면 (데이터 부족 등) 기본값 처리
        existing_codes = target_df['code'].tolist()
        for code in my_codes:
            if code not in existing_codes:
                print(f"   ⚠️ {code} 종목의 AI 점수가 없어 50점으로 가정합니다.")
                # concat 대신 loc 사용
                new_row = {'code': code, 'ai_score': 50.0}
                target_df.loc[len(target_df)] = new_row

        # [핵심 로직] 점수를 비중으로 변환 (Score Weighting)
        # 점수 그대로를 비율로 사용 (예: 90점:40점 => 90:40 비율)
        # 조금 더 극적인 효과를 원하면 제곱(score^2)을 사용할 수도 있음
        
        target_ratios = {}
        for _, row in target_df.iterrows():
            score = row['ai_score']
            code = row['code']
            
            weight = score # 기본: 점수 그대로 비중 사용

            # 페널티 및 보너스 로직
            if score >= 80:
                weight = score * 1.2 # 강력 매수 (가중치 1.2배)
            elif score <= 40:
                weight = score * 0.5 # 매도 권장 (가중치 반토막)
            
            target_ratios[code] = weight

        return self.calculate_rebalancing(current_holdings, target_ratios)



# ==========================================
# 테스트 실행 코드 (사용 예시)
# ==========================================
if __name__ == "__main__":
    rebalancer = PortfolioRebalancer()
    
    # [상황 설정] 내 계좌 상황
    my_portfolio = {
        '005930': 5000000, # 삼성전자
        '000660': 4000000, # SK하이닉스
        '005380': 1000000  # 현대차
    }
    
    # [가상 데이터] AI가 분석한 점수라고 가정
    # 실제로는 analyzer.get_ai_scores() 결과를 넣으면 됩니다.
    fake_ai_scores = pd.DataFrame({
        'code': ['005930', '000660', '005380'],
        'ai_score': [80.0, 40.0, 90.0] 
        # 현대차(90) > 삼성전자(80) > 하이닉스(40)
    })
    
    # 실행
    df = rebalancer.run_ai_rebalancing(my_portfolio, fake_ai_scores)
    
    # 출력
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
            