### 더미데이터로 AI 한 주기 돌리기

##### python 환경
 - Using python.exe (3.12.7)

##### AI learning 실행 명령어
```
poetry install
poetry run python src/ai_app/main.py

```

##### 실행 데이터
```

=============================================
         🚀 금융 AI 파이프라인 시작
=============================================
✅ NLP 모듈 초기화: snunlp/KR-FINBERT-SC 로드 중...
config.json: 100%|██████████████████████████████████████████████| 881/881 [00:00<?, ?B/s]
C:\Users\user\AppData\Local\pypoetry\Cache\virtualenvs\ai-app-tMKCnwgX-py3.12\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\user\.cache\huggingface\hub\models--snunlp--KR-FINBERT-SC. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
pytorch_model.bin: 100%|██████████████████████████████| 406M/406M [00:04<00:00, 88.4MB/s]
tokenizer_config.json: 100%|█████████████████████████████| 372/372 [00:00<00:00, 306kB/s]
vocab.txt: 143kB [00:00, 20.6MB/s]
tokenizer.json: 294kB [00:00, 48.9MB/s]
special_tokens_map.json: 100%|██████████████████████████████████| 112/112 [00:00<?, ?B/s]
Device set to use cpu
✅ NLP 모듈 로드 완료.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.

[단계 1. NLP 결과 (샘플)]
  sentiment_label  sentiment_score
0        positive        -0.999885
1        negative        -0.999695
2         neutral        -0.986806
✅ GCN 초기화: 입력=10, 임베딩 차원=32
C:\FinalProject_notOff\AI\AI-app\src\ai_app\pipeline_module.py:90: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:256.)
  edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
✅ 그래프 데이터 생성 완료. 노드 수: 100, 엣지 수: 1000
✅ GCN Node Embedding 추출 완료. 형태: (100, 32)

[단계 2. GCN Embedding 결과 (샘플)]
        gcn_emb_0  gcn_emb_1  gcn_emb_2  ...  gcn_emb_29  gcn_emb_30  gcn_emb_31
TKR000  -0.000776   0.269254   0.275683  ...    0.248878    0.180547   -0.046954
TKR001   0.008538   0.181657   0.256949  ...    0.187930    0.127864   -0.046506
TKR002   0.006933   0.225944   0.288115  ...    0.206155    0.137642   -0.063551

[3 rows x 32 columns]

[단계 3. XGBoost 실행]
✅ XGBoost 학습 데이터 크기: 8000
C:\Users\user\AppData\Local\pypoetry\Cache\virtualenvs\ai-app-tMKCnwgX-py3.12\Lib\site-packages\xgboost\training.py:199: UserWarning: [10:44:54] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:790:
Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
--- XGBoost 결과 ---
테스트 데이터 정확도 (Accuracy): 0.5180
----------------------
=============================================
         ✅ 파이프라인 전체 실행 완료
=============================================
```