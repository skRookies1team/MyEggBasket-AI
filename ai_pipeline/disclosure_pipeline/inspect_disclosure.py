import os
import sys
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ai_pipeline.boosting_model.disclosure_feature_transform import transform_disclosure_df

csv_path = os.path.join(os.path.dirname(__file__), 'data', 'integrated_financial_data.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"loaded {csv_path}: {len(df)} rows, {len(df.columns)} cols")
    print("columns:", df.columns.tolist()[:60])
    print("\nhead(3):")
    print(df.head(3).to_string(index=False))

    feats = transform_disclosure_df(df)
    if feats is None or feats.empty:
        print("\ntransformed features empty")
    else:
        print(f"\ntransformed disc features: shape={feats.shape}")
        print("disc columns sample:", list(feats.columns)[:60])
        print("\ndisc head(3):")
        print(feats.head(3).to_string())
else:
    print("file not found:", csv_path)
