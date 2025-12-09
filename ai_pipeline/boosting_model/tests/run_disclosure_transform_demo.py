@ -1,33 +0,0 @@
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

try:
    fe = FeatureEngineer(data_dir='data')
    out_lines = []
    out_lines.append(f"disclosure_df loaded? {fe.disclosure_df is not None}")
    if fe.disclosure_df is not None:
        out_lines.append('disc index sample: '+ str(list(fe.disclosure_df.index[:10])))
        out_lines.append('disc columns sample: '+ str(list(fe.disclosure_df.columns)[:50]))

    res = fe.create_final_features()
    if res is None or res[0] is None:
        out_lines.append('create_final_features returned None or empty X')
    else:
        X, y, codes = res
        disc_cols = [c for c in X.columns if str(c).startswith('disc_')]
        out_lines.append(f'X shape: {X.shape}')
        out_lines.append(f'disc cols in final X (count): {len(disc_cols)}')
        out_lines.append('disc cols sample: '+ str(disc_cols[:50]))
        if disc_cols:
            out_lines.append('\nStatistics for disc cols:')
            out_lines.append(str(X[disc_cols].describe().T))

    # write to file for reliable retrieval
    with open(os.path.join(os.path.dirname(__file__), 'diagnose_output.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines))
except Exception as e:
    import traceback
    with open(os.path.join(os.path.dirname(__file__), 'diagnose_output.txt'), 'w', encoding='utf-8') as fh:
        fh.write('Exception during diagnosis:\n')
        traceback.print_exc(file=fh)