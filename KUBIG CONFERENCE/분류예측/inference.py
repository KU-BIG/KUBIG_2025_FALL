"""
CTR 예측 모델 추론 코드
Team: 환승공모전
"""

# ====================================================
# Imports
# ====================================================
import numpy as np
import pandas as pd
import gc
import pickle
import json
from pathlib import Path
from scipy.special import expit

# ====================================================
# Configuration
# ====================================================
CONFIG = {
    'TEST_PATH': './data/test.parquet',
    'MODEL_DIR': './models',
    'OUTPUT_DIR': './outputs',
    'SUBMISSION_PATH': './submission.csv',
    'SEED': 42
}

# ====================================================
# Feature Engineering Functions 
# ====================================================
def extract_seq_features_simple(df, col='seq'):
    """시퀀스 특징 추출"""
    print(f"Extracting simple seq features from {len(df)} rows...")
    result_df = pd.DataFrame(index=df.index)
    chunk_size = 50000

    print("  Computing seq_length...")
    result_df['seq_length'] = df[col].str.count(',').astype('int16') + 1

    print("  Computing seq_last_item...")
    def get_last_item(s):
        idx = s.rfind(',')
        return s[idx+1:] if idx != -1 else s

    last_items = []
    for i in range(0, len(df), chunk_size):
        chunk = df[col].iloc[i:i+chunk_size].apply(get_last_item)
        last_items.append(chunk)
        gc.collect()

    result_df['seq_last_item'] = pd.concat(last_items).astype('category')
    del last_items
    gc.collect()

    print("  Computing seq_first_item...")
    def get_first_item(s):
        idx = s.find(',')
        return s[:idx] if idx != -1 else s

    first_items = []
    for i in range(0, len(df), chunk_size):
        chunk = df[col].iloc[i:i+chunk_size].apply(get_first_item)
        first_items.append(chunk)
        gc.collect()

    first_concat = pd.concat(first_items)
    result_df['seq_first_last_same'] = (first_concat.values == result_df['seq_last_item'].values).astype('int8')
    result_df['seq_first_item'] = first_concat.astype('category')
    del first_items, first_concat
    gc.collect()

    print(f"  Created {len(result_df.columns)} seq features")
    return result_df

# ====================================================
# Post-processing Classes
# ====================================================
class BetaCalibration:
    """Beta Calibration for probability calibration"""
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.c = 1.0
    
    def transform(self, probs):
        calibrated = expit(self.a * np.log(probs / (1 - probs + 1e-10)) + self.b) ** self.c
        return np.clip(calibrated, 1e-6, 1-1e-6)

# ====================================================
# Inference Functions
# ====================================================
def predict_lgbm(model_info, X_test):
    """LightGBM 모델 예측"""
    all_predictions = []
    all_weights = []
    
    for hp_idx, (models_hp, calibrators_hp, weight) in enumerate(
        zip(model_info['models'], model_info['calibrators'], model_info['weights'])):
        
        hp_predictions = []
        
        for bag_idx, (models_fold, (calib_method, calibrator)) in enumerate(
            zip(models_hp, calibrators_hp)):
            
            # Fold별 예측 평균
            fold_pred = np.zeros(len(X_test), dtype=np.float32)
            for model in models_fold:
                fold_pred += model.predict_proba(X_test)[:,1] / len(models_fold)
            
            # Calibration 적용
            if calib_method == 'isotonic':
                fold_pred_calibrated = calibrator.transform(fold_pred)
            elif calib_method == 'beta':
                calibrator_obj = BetaCalibration()
                calibrator_obj.a = calibrator.a
                calibrator_obj.b = calibrator.b
                calibrator_obj.c = calibrator.c
                fold_pred_calibrated = calibrator_obj.transform(fold_pred)
            elif calib_method == 'power':
                fold_pred_calibrated = fold_pred ** calibrator
            elif calib_method == 'clip':
                lower, upper = calibrator
                fold_pred_calibrated = np.clip(fold_pred, lower, upper)
            else:
                fold_pred_calibrated = fold_pred
            
            hp_predictions.append(fold_pred_calibrated)
        
        # Bag 평균
        hp_pred = np.mean(hp_predictions, axis=0)
        all_predictions.append(hp_pred)
        all_weights.append(weight)
    
    # 가중 평균
    weights = np.array(all_weights)
    predictions = np.vstack(all_predictions)
    final_pred = (weights[:,None] * predictions).sum(axis=0) / (weights.sum() + 1e-12)
    
    return final_pred

def predict_catboost(model_info, X_test):
    """CatBoost 모델 예측"""
    all_predictions = []
    all_weights = []
    
    for hp_idx, (models_hp, calibrators_hp, weight) in enumerate(
        zip(model_info['models'], model_info['calibrators'], model_info['weights'])):
        
        hp_predictions = []
        
        for bag_idx, (models_fold, (calib_method, calibrator)) in enumerate(
            zip(models_hp, calibrators_hp)):
            
            # Fold별 예측 평균
            fold_pred = np.zeros(len(X_test), dtype=np.float32)
            for model in models_fold:
                fold_pred += model.predict_proba(X_test)[:,1] / len(models_fold)
            
            # Calibration 적용
            if calib_method == 'isotonic':
                fold_pred_calibrated = calibrator.transform(fold_pred)
            elif calib_method == 'beta':
                calibrator_obj = BetaCalibration()
                calibrator_obj.a = calibrator.a
                calibrator_obj.b = calibrator.b
                calibrator_obj.c = calibrator.c
                fold_pred_calibrated = calibrator_obj.transform(fold_pred)
            elif calib_method == 'power':
                fold_pred_calibrated = fold_pred ** calibrator
            elif calib_method == 'clip':
                lower, upper = calibrator
                fold_pred_calibrated = np.clip(fold_pred, lower, upper)
            else:
                fold_pred_calibrated = fold_pred
            
            hp_predictions.append(fold_pred_calibrated)
        
        # Bag 평균
        hp_pred = np.mean(hp_predictions, axis=0)
        all_predictions.append(hp_pred)
        all_weights.append(weight)
    
    # 가중 평균
    weights = np.array(all_weights)
    predictions = np.vstack(all_predictions)
    final_pred = (weights[:,None] * predictions).sum(axis=0) / (weights.sum() + 1e-12)
    
    return final_pred

def wblend_prob(preds_list, scores_list):
    """확률 기반 가중 앙상블"""
    W = np.array(scores_list, dtype=np.float64)
    P = np.vstack(preds_list)
    return (W[:,None] * P).sum(axis=0) / (W.sum() + 1e-12)

# ====================================================
# Main Inference Function
# ====================================================
def main():
    print("="*60)
    print("CTR Prediction Model Inference")
    print("="*60)
    
    # 설정 파일 로드 (있는 경우)
    config_path = Path(CONFIG['OUTPUT_DIR']) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            train_config = json.load(f)
            print(f"Loaded training configuration from {config_path}")
    
    # 테스트 데이터 로드
    print("\n[1/5] Loading test data...")
    test = pd.read_parquet(CONFIG['TEST_PATH'], engine="pyarrow")
    print(f"Test shape: {test.shape}")
    
    # 시퀀스 특징 추출
    print("\n[2/5] Extracting sequence features...")
    seq_feats_test = extract_seq_features_simple(test)
    test = pd.concat([test, seq_feats_test], axis=1)
    del seq_feats_test
    gc.collect()
    
    test = test.drop(columns=['seq'])
    gc.collect()
    
    # 데이터 타입 처리
    print("\n[3/5] Processing data types...")
    for c in test.columns:
        if test[c].dtype == "object":
            test[c] = test[c].fillna('missing').astype("category")
    
    seq_cols = [c for c in test.columns if c.startswith('seq_')]
    for c in seq_cols:
        if test[c].dtype.name == 'category':
            test[c] = test[c].cat.add_categories(['missing']).fillna('missing')
    
    # 파생 특징 생성
    print("\n[4/5] Creating derived features...")
    feat_a_cols = [c for c in test.columns if c.startswith("feat_a_")]
    test["feat_a_missing"] = test[feat_a_cols].isna().any(axis=1).astype("int8")
    
    lh_cols = [c for c in test.columns if c.startswith("l_feat_")] + \
              [c for c in test.columns if c.startswith("history_b_")]
    test["flag_lh_missing_any"] = test[lh_cols].isna().any(axis=1).astype("int8")
    
    def has_leading_zero(x):
        s = str(x)
        return int(s.startswith("0"))
    test["flag_hour_leading_zero"] = test["hour"].astype(str).map(has_leading_zero).astype("int8")
    
    # 데이터 분할
    te_miss = test[test["feat_a_missing"]==1].copy()
    te_nmiss = test[test["feat_a_missing"]==0].copy()
    print(f"Missing: {len(te_miss)} samples")
    print(f"Not-missing: {len(te_nmiss)} samples")
    
    # 모델 로드 및 예측
    print("\n[5/5] Loading models and making predictions...")
    
    # 학습 결과 로드
    results_path = Path(CONFIG['OUTPUT_DIR']) / 'training_results.pkl'
    if results_path.exists():
        with open(results_path, 'rb') as f:
            training_results = pickle.load(f)
        
        # 학습 시 저장된 점수 사용
        miss_scores = [
            training_results['missing']['gbdt']['score'],
            training_results['missing']['dart']['score'],
            training_results['missing']['catboost']['score']
        ]
        nmiss_scores = [
            training_results['not_missing']['gbdt']['score'],
            training_results['not_missing']['dart']['score'],
            training_results['not_missing']['catboost']['score']
        ]
        
        print(f"\nMissing Subset Scores:")
        print(f"  GBDT: {miss_scores[0]:.6f}")
        print(f"  DART: {miss_scores[1]:.6f}")
        print(f"  CatBoost: {miss_scores[2]:.6f}")
        
        print(f"\nNot-Missing Subset Scores:")
        print(f"  GBDT: {nmiss_scores[0]:.6f}")
        print(f"  DART: {nmiss_scores[1]:.6f}")
        print(f"  CatBoost: {nmiss_scores[2]:.6f}")
    else:
        # 기본 가중치 (동일 가중)
        miss_scores = [1.0, 1.0, 1.0]
        nmiss_scores = [1.0, 1.0, 1.0]
    
    # Missing subset 예측
    print("\n--- Predicting for Missing subset ---")
    miss_predictions = []
    
    # LGBM GBDT
    print("Loading LGBM-GBDT (missing)...")
    with open(Path(CONFIG['MODEL_DIR']) / 'lgbm_gbdt_missing.pkl', 'rb') as f:
        model_info = pickle.load(f)
    features = model_info['features']
    pred_miss_gbdt = predict_lgbm(model_info, te_miss[features])
    miss_predictions.append(pred_miss_gbdt)
    
    # LGBM DART
    print("Loading LGBM-DART (missing)...")
    with open(Path(CONFIG['MODEL_DIR']) / 'lgbm_dart_missing.pkl', 'rb') as f:
        model_info = pickle.load(f)
    pred_miss_dart = predict_lgbm(model_info, te_miss[features])
    miss_predictions.append(pred_miss_dart)
    
    # CatBoost
    print("Loading CatBoost (missing)...")
    with open(Path(CONFIG['MODEL_DIR']) / 'catboost_missing.pkl', 'rb') as f:
        model_info = pickle.load(f)
    pred_miss_cat = predict_catboost(model_info, te_miss[features])
    miss_predictions.append(pred_miss_cat)
    
    # Missing subset 앙상블
    miss_final = wblend_prob(miss_predictions, miss_scores)
    
    # Not-Missing subset 예측
    print("\n--- Predicting for Not-Missing subset ---")
    nmiss_predictions = []
    
    # LGBM GBDT
    print("Loading LGBM-GBDT (not-missing)...")
    with open(Path(CONFIG['MODEL_DIR']) / 'lgbm_gbdt_not_missing.pkl', 'rb') as f:
        model_info = pickle.load(f)
    pred_nmiss_gbdt = predict_lgbm(model_info, te_nmiss[features])
    nmiss_predictions.append(pred_nmiss_gbdt)
    
    # LGBM DART
    print("Loading LGBM-DART (not-missing)...")
    with open(Path(CONFIG['MODEL_DIR']) / 'lgbm_dart_not_missing.pkl', 'rb') as f:
        model_info = pickle.load(f)
    pred_nmiss_dart = predict_lgbm(model_info, te_nmiss[features])
    nmiss_predictions.append(pred_nmiss_dart)
    
    # CatBoost
    print("Loading CatBoost (not-missing)...")
    with open(Path(CONFIG['MODEL_DIR']) / 'catboost_not_missing.pkl', 'rb') as f:
        model_info = pickle.load(f)
    pred_nmiss_cat = predict_catboost(model_info, te_nmiss[features])
    nmiss_predictions.append(pred_nmiss_cat)
    
    # Not-Missing subset 앙상블
    nmiss_final = wblend_prob(nmiss_predictions, nmiss_scores)
    
    # 최종 제출 파일 생성
    print("\n--- Creating submission file ---")
    submission = pd.concat([
        pd.DataFrame({"ID": te_miss["ID"], "clicked": miss_final}),
        pd.DataFrame({"ID": te_nmiss["ID"], "clicked": nmiss_final}),
    ], axis=0).set_index("ID").reindex(test["ID"]).reset_index()
    
    submission.to_csv(CONFIG['SUBMISSION_PATH'], index=False)
    print(f"[Saved] {CONFIG['SUBMISSION_PATH']}")
    
    # 예측 통계
    print("\n--- Prediction Statistics ---")
    print(f"Mean prediction: {submission['clicked'].mean():.6f}")
    print(f"Std prediction: {submission['clicked'].std():.6f}")
    print(f"Min prediction: {submission['clicked'].min():.6f}")
    print(f"Max prediction: {submission['clicked'].max():.6f}")
    
    print("\n" + "="*60)
    print("=== INFERENCE COMPLETE ===")
    print("="*60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(CONFIG['SEED'])
    main()