@ -1,25 +0,0 @@
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

fe = FeatureEngineer(data_dir='data')
print('disclosure_df loaded?', fe.disclosure_df is not None)
if not fe.disclosure_df is None:
    print('disc index sample:', list(fe.disclosure_df.index[:10]))
    print('disc cols sample:', list(fe.disclosure_df.columns)[:20])

# pick one csv
csv = os.path.join(os.path.dirname(__file__), '..', 'data', '20251120.csv')
if os.path.exists(csv):
    X,y,codes = fe._process_single_file(csv)
    if X is None:
        print('processing returned None')
    else:
        disc_cols = [c for c in X.columns if str(c).startswith('disc_')]
        print('X cols count:', len(X.columns))
        print('disc cols in X count:', len(disc_cols))
        print('disc cols sample:', disc_cols[:50])
        if disc_cols:
            print(X[disc_cols].describe().T)
else:
    print('csv not found:', csv)