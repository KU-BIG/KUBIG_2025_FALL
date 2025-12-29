"""
CTR 예측 모델 학습 코드
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
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, log_loss
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.special import expit

# ====================================================
# Configuration
# ====================================================
CONFIG = {
    'TRAIN_PATH': './data/train.parquet',
    'TEST_PATH': './data/test.parquet',
    'OUTPUT_DIR': './outputs',
    'MODEL_DIR': './models',
    'N_SPLITS': 3,
    'BAG_ROUNDS': 10,
    'SEED': 42,
    'DEVICE': 'gpu'
}

# Create directories
Path(CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['MODEL_DIR']).mkdir(parents=True, exist_ok=True)

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
# Metrics
# ====================================================
def weighted_log_loss(y_true, y_pred):
    """가중 로그 손실 계산"""
    w0 = 0.5 / (y_true==0).mean()
    w1 = 0.5 / (y_true==1).mean()
    sw = np.where(y_true==1, w1, w0)
    return log_loss(y_true, y_pred, sample_weight=sw)

def leaderboard_score(y_true, y_pred):
    """리더보드 평가 지표"""
    ap = average_precision_score(y_true, y_pred)
    wll = weighted_log_loss(y_true, y_pred)
    return 0.5*ap + 0.5*(1/(1+wll))

# ====================================================
# Post-processing Classes
# ====================================================
class BetaCalibration:
    """Beta Calibration for probability calibration"""
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.c = 1.0

    def fit(self, probs, labels):
        def beta_objective(params):
            a, b, c = params
            calibrated = expit(a * np.log(probs / (1 - probs + 1e-10)) + b) ** c
            calibrated = np.clip(calibrated, 1e-6, 1-1e-6)
            return -leaderboard_score(labels, calibrated)

        result = minimize(beta_objective, x0=[1.0, 0.0, 1.0],
                         bounds=[(0.1, 10.0), (-5.0, 5.0), (0.1, 10.0)],
                         method='L-BFGS-B')
        self.a, self.b, self.c = result.x
        return self

    def transform(self, probs):
        calibrated = expit(self.a * np.log(probs / (1 - probs + 1e-10)) + self.b) ** self.c
        return np.clip(calibrated, 1e-6, 1-1e-6)

def optimize_post_processing(oof_probs, y_true, test_probs):
    """최적 post-processing 방법 선택"""
    methods = {}

    # 1) Isotonic Regression
    iso = IsotonicRegression(y_min=1e-6, y_max=1-1e-6, out_of_bounds="clip")
    iso.fit(oof_probs, y_true)
    oof_iso = iso.transform(oof_probs)
    test_iso = iso.transform(test_probs)
    methods['isotonic'] = (leaderboard_score(y_true, oof_iso), test_iso, iso)

    # 2) Beta Calibration
    beta = BetaCalibration()
    beta.fit(oof_probs, y_true)
    oof_beta = beta.transform(oof_probs)
    test_beta = beta.transform(test_probs)
    methods['beta'] = (leaderboard_score(y_true, oof_beta), test_beta, beta)

    # 3) Power Transform
    def power_objective(p):
        transformed = oof_probs ** p
        return -leaderboard_score(y_true, transformed)

    result = minimize(power_objective, x0=1.0, bounds=[(0.5, 2.0)], method='L-BFGS-B')
    power = result.x[0]
    oof_power = oof_probs ** power
    test_power = test_probs ** power
    methods['power'] = (leaderboard_score(y_true, oof_power), test_power, power)

    # 4) Clipping 최적화
    def clip_objective(params):
        lower, upper = params
        clipped = np.clip(oof_probs, lower, upper)
        return -leaderboard_score(y_true, clipped)

    result = minimize(clip_objective, x0=[1e-6, 1-1e-6],
                     bounds=[(1e-7, 0.01), (0.99, 1-1e-7)], method='L-BFGS-B')
    lower, upper = result.x
    test_clip = np.clip(test_probs, lower, upper)
    methods['clip'] = (leaderboard_score(y_true, np.clip(oof_probs, lower, upper)), test_clip, (lower, upper))

    # 최고 성능 방법 선택
    best_method = max(methods.items(), key=lambda x: x[1][0])
    print(f"    Best post-processing: {best_method[0]} (CV={best_method[1][0]:.6f})")

    return best_method[1][1], best_method[1][0], best_method[0], best_method[1][2]

# ====================================================
# Undersampling
# ====================================================
def undersample_1to1(df, target="clicked", seed=42):
    """1:1 비율로 언더샘플링"""
    pos = df[df[target]==1]
    neg = df[df[target]==0].sample(n=len(pos), random_state=seed)
    result = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=seed)
    gc.collect()
    return result

# ====================================================
# Model Training Functions
# ====================================================
def cv_train_predict_lgbm(df_train, df_test, features, model_type="gbdt",
                         target="clicked", n_splits=3, bag_rounds=10,
                         seed=42, hyperparams_list=None, save_models=True, subset_name=""):
    """LightGBM 학습 및 예측"""
    if hyperparams_list is None:
        hyperparams_list = [
            {'learning_rate': 0.08, 'num_leaves': 128, 'min_child_samples': 80},
            {'learning_rate': 0.06, 'num_leaves': 96, 'min_child_samples': 100},
            {'learning_rate': 0.10, 'num_leaves': 160, 'min_child_samples': 60},
        ]

    X_te = df_test[features]
    all_preds = []
    all_weights = []
    all_models = []
    all_calibrators = []

    for hp_idx, hp in enumerate(hyperparams_list):
        print(f"\n*** Hyperparameter Set {hp_idx+1}/{len(hyperparams_list)} ***")
        print(f"    {hp}")

        preds_test_bagged = []
        weights_bagged = []
        models_hp = []
        calibrators_hp = []

        for br in range(bag_rounds):
            print(f"\n--- Bag Round {br+1}/{bag_rounds} ---")
            dfb = undersample_1to1(df_train, target=target, seed=seed+br+hp_idx*100)
            X = dfb[features]
            y = dfb[target].values

            oof_probs = np.zeros(len(X), dtype=np.float32)
            te_pred_raw = np.zeros(len(X_te), dtype=np.float32)
            models_fold = []

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed+br+hp_idx*100)

            for fold, (tr, va) in enumerate(skf.split(X, y), 1):
                X_tr, y_tr = X.iloc[tr], y[tr]
                X_va, y_va = X.iloc[va], y[va]

                model = LGBMClassifier(
                    objective="binary",
                    boosting_type=model_type,
                    learning_rate=hp['learning_rate'],
                    num_leaves=hp['num_leaves'],
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=hp['min_child_samples'],
                    reg_lambda=1.0,
                    n_estimators=2000,
                    random_state=seed + br*100 + fold + hp_idx*1000,
                    device_type=CONFIG['DEVICE'],
                    verbose=-1
                )

                callbacks = [lgb.log_evaluation(100)]
                if model_type != "dart":
                    callbacks.insert(0, lgb.early_stopping(50))

                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    callbacks=callbacks
                )

                oof_probs[va] = model.predict_proba(X_va)[:,1]
                te_pred_raw += model.predict_proba(X_te)[:,1] / n_splits
                models_fold.append(model)

                fold_lb = leaderboard_score(y_va, oof_probs[va])
                print(f"[LGBM-{model_type} HP{hp_idx+1} | Bag {br+1} | Fold {fold}] LB = {fold_lb:.6f}")

            cv_lb_raw = leaderboard_score(y, oof_probs)
            print(f"[LGBM-{model_type} HP{hp_idx+1} | Bag {br+1}] CV LB (raw) = {cv_lb_raw:.6f}")

            # Post-processing
            te_pred_opt, cv_lb_opt, best_method, calibrator = optimize_post_processing(oof_probs, y, te_pred_raw)

            preds_test_bagged.append(te_pred_opt)
            weights_bagged.append(float(cv_lb_opt))
            models_hp.append(models_fold)
            calibrators_hp.append((best_method, calibrator))

            del dfb, X, y, oof_probs, te_pred_raw
            gc.collect()

        W = np.array(weights_bagged, dtype=np.float64)
        P = np.vstack(preds_test_bagged)
        hp_final_pred = (W[:,None] * P).sum(axis=0) / (W.sum() + 1e-12)

        all_preds.append(hp_final_pred)
        all_weights.append(float(W.mean()))
        all_models.append(models_hp)
        all_calibrators.append(calibrators_hp)

    # 여러 하이퍼파라미터 결과 앙상블
    W_hp = np.array(all_weights, dtype=np.float64)
    P_hp = np.vstack(all_preds)
    final_pred = (W_hp[:,None] * P_hp).sum(axis=0) / (W_hp.sum() + 1e-12)

    # 모델 저장
    if save_models:
        model_info = {
            'model_type': f'lgbm_{model_type}',
            'subset': subset_name,
            'models': all_models,
            'calibrators': all_calibrators,
            'weights': all_weights,
            'hyperparams_list': hyperparams_list,
            'features': features
        }
        save_path = Path(CONFIG['MODEL_DIR']) / f'lgbm_{model_type}_{subset_name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"Models saved to {save_path}")

    return final_pred, float(W_hp.mean())

def cv_train_predict_catboost(df_train, df_test, features, target="clicked",
                              n_splits=3, bag_rounds=10, seed=42,
                              hyperparams_list=None, save_models=True, subset_name=""):
    """CatBoost 학습 및 예측"""
    if hyperparams_list is None:
        hyperparams_list = [
            {'learning_rate': 0.08, 'depth': 8},
            {'learning_rate': 0.06, 'depth': 6},
        ]

    cat_features = [i for i, c in enumerate(features) if df_train[c].dtype.name == 'category']
    X_te = df_test[features]
    all_preds = []
    all_weights = []
    all_models = []
    all_calibrators = []

    for hp_idx, hp in enumerate(hyperparams_list):
        print(f"\n*** Hyperparameter Set {hp_idx+1}/{len(hyperparams_list)} ***")
        print(f"    {hp}")

        preds_test_bagged = []
        weights_bagged = []
        models_hp = []
        calibrators_hp = []

        for br in range(bag_rounds):
            print(f"\n--- Bag Round {br+1}/{bag_rounds} ---")
            dfb = undersample_1to1(df_train, target=target, seed=seed+br+hp_idx*100)
            X = dfb[features]
            y = dfb[target].values

            oof = np.zeros(len(X), dtype=np.float32)
            te_pred_raw = np.zeros(len(X_te), dtype=np.float32)
            models_fold = []

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed+br+hp_idx*100)

            for fold, (tr, va) in enumerate(skf.split(X, y), 1):
                X_tr, y_tr = X.iloc[tr], y[tr]
                X_va, y_va = X.iloc[va], y[va]

                train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
                val_pool = Pool(X_va, y_va, cat_features=cat_features)

                model = CatBoostClassifier(
                    iterations=2000,
                    learning_rate=hp['learning_rate'],
                    depth=hp['depth'],
                    loss_function='Logloss',
                    eval_metric='AUC',
                    random_seed=seed + br*100 + fold + hp_idx*1000,
                    task_type='GPU' if CONFIG['DEVICE'] == 'gpu' else 'CPU',
                    devices='0' if CONFIG['DEVICE'] == 'gpu' else None,
                    verbose=100,
                    early_stopping_rounds=50
                )

                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    use_best_model=True
                )

                oof[va] = model.predict_proba(X_va)[:,1]
                te_pred_raw += model.predict_proba(X_te)[:,1] / n_splits
                models_fold.append(model)

                fold_lb = leaderboard_score(y_va, oof[va])
                print(f"[CatBoost HP{hp_idx+1} | Bag {br+1} | Fold {fold}] LB = {fold_lb:.6f}")

            cv_lb_raw = leaderboard_score(y, oof)
            print(f"[CatBoost HP{hp_idx+1} | Bag {br+1}] CV LB (raw) = {cv_lb_raw:.6f}")

            # Post-processing
            te_pred_opt, cv_lb_opt, best_method, calibrator = optimize_post_processing(oof, y, te_pred_raw)

            preds_test_bagged.append(te_pred_opt)
            weights_bagged.append(float(cv_lb_opt))
            models_hp.append(models_fold)
            calibrators_hp.append((best_method, calibrator))

            del dfb, X, y, oof, te_pred_raw, train_pool, val_pool
            gc.collect()

        W = np.array(weights_bagged, dtype=np.float64)
        P = np.vstack(preds_test_bagged)
        hp_final_pred = (W[:,None] * P).sum(axis=0) / (W.sum() + 1e-12)

        all_preds.append(hp_final_pred)
        all_weights.append(float(W.mean()))
        all_models.append(models_hp)
        all_calibrators.append(calibrators_hp)

    # 여러 하이퍼파라미터 결과 앙상블
    W_hp = np.array(all_weights, dtype=np.float64)
    P_hp = np.vstack(all_preds)
    final_pred = (W_hp[:,None] * P_hp).sum(axis=0) / (W_hp.sum() + 1e-12)

    # 모델 저장
    if save_models:
        model_info = {
            'model_type': 'catboost',
            'subset': subset_name,
            'models': all_models,
            'calibrators': all_calibrators,
            'weights': all_weights,
            'hyperparams_list': hyperparams_list,
            'features': features,
            'cat_features': cat_features
        }
        save_path = Path(CONFIG['MODEL_DIR']) / f'catboost_{subset_name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"Models saved to {save_path}")

    return final_pred, float(W_hp.mean())

# ====================================================
# Main Training Function
# ====================================================
def main():
    print("="*60)
    print("CTR Prediction Model Training")
    print("="*60)
    
    # 데이터 로드
    print("\n[1/7] Loading data...")
    train = pd.read_parquet(CONFIG['TRAIN_PATH'], engine="pyarrow")
    test = pd.read_parquet(CONFIG['TEST_PATH'], engine="pyarrow")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # 시퀀스 특징 추출
    print("\n[2/7] Extracting sequence features...")
    seq_feats_train = extract_seq_features_simple(train)
    train = pd.concat([train, seq_feats_train], axis=1)
    del seq_feats_train
    gc.collect()

    seq_feats_test = extract_seq_features_simple(test)
    test = pd.concat([test, seq_feats_test], axis=1)
    del seq_feats_test
    gc.collect()

    train = train.drop(columns=['seq'])
    test = test.drop(columns=['seq'])
    gc.collect()

    # 데이터 타입 처리
    print("\n[3/7] Processing data types...")
    for c in train.columns:
        if train[c].dtype == "object":
            train[c] = train[c].fillna('missing').astype("category")
    for c in test.columns:
        if test[c].dtype == "object":
            test[c] = test[c].fillna('missing').astype("category")

    seq_cols = [c for c in train.columns if c.startswith('seq_')]
    for c in seq_cols:
        if train[c].dtype.name == 'category':
            train[c] = train[c].cat.add_categories(['missing']).fillna('missing')
            test[c] = test[c].cat.add_categories(['missing']).fillna('missing')

    # 파생 특징 생성
    print("\n[4/7] Creating derived features...")
    feat_a_cols = [c for c in train.columns if c.startswith("feat_a_")]
    train["feat_a_missing"] = train[feat_a_cols].isna().any(axis=1).astype("int8")
    test["feat_a_missing"] = test[feat_a_cols].isna().any(axis=1).astype("int8")

    lh_cols = [c for c in train.columns if c.startswith("l_feat_")] + \
              [c for c in train.columns if c.startswith("history_b_")]
    train["flag_lh_missing_any"] = train[lh_cols].isna().any(axis=1).astype("int8")
    test["flag_lh_missing_any"] = test[lh_cols].isna().any(axis=1).astype("int8")

    def has_leading_zero(x):
        s = str(x)
        return int(s.startswith("0"))
    train["flag_hour_leading_zero"] = train["hour"].astype(str).map(has_leading_zero).astype("int8")
    test["flag_hour_leading_zero"] = test["hour"].astype(str).map(has_leading_zero).astype("int8")

    # Feature 리스트 생성
    DROP_COLS = ["clicked", "ID"]
    FEATURES = [c for c in train.columns if c not in DROP_COLS]
    if "feat_a_missing" in FEATURES:
        FEATURES.remove("feat_a_missing")
    print(f"Number of features: {len(FEATURES)}")

    # 데이터 분할
    print("\n[5/7] Splitting data by feat_a_missing...")
    tr_miss = train[train["feat_a_missing"]==1].copy()
    tr_nmiss = train[train["feat_a_missing"]==0].copy()
    te_miss = test[test["feat_a_missing"]==1].copy()
    te_nmiss = test[test["feat_a_missing"]==0].copy()
    print(f"Missing: {len(tr_miss)} train, {len(te_miss)} test")
    print(f"Not-missing: {len(tr_nmiss)} train, {len(te_nmiss)} test")

    # 모델 학습 - Missing subset
    print("\n[6/7] Training models on Missing subset...")
    print("="*60)
    print("=== MISSING SUBSET ===")
    print("="*60)

    print("\n[1/3] LGBM-GBDT on Missing...")
    pred_miss_gbdt, lb_miss_gbdt = cv_train_predict_lgbm(
        tr_miss, te_miss, FEATURES, model_type="gbdt",
        n_splits=CONFIG['N_SPLITS'], bag_rounds=CONFIG['BAG_ROUNDS'],
        seed=CONFIG['SEED'], save_models=True, subset_name="missing"
    )

    print("\n[2/3] LGBM-DART on Missing...")
    pred_miss_dart, lb_miss_dart = cv_train_predict_lgbm(
        tr_miss, te_miss, FEATURES, model_type="dart",
        n_splits=CONFIG['N_SPLITS'], bag_rounds=CONFIG['BAG_ROUNDS'],
        seed=CONFIG['SEED'], save_models=True, subset_name="missing"
    )

    print("\n[3/3] CatBoost on Missing...")
    pred_miss_cat, lb_miss_cat = cv_train_predict_catboost(
        tr_miss, te_miss, FEATURES,
        n_splits=CONFIG['N_SPLITS'], bag_rounds=CONFIG['BAG_ROUNDS'],
        seed=CONFIG['SEED'], save_models=True, subset_name="missing"
    )

    # 모델 학습 - Not-missing subset
    print("\n[7/7] Training models on Not-Missing subset...")
    print("="*60)
    print("=== NOT-MISSING SUBSET ===")
    print("="*60)

    print("\n[1/3] LGBM-GBDT on Not-Missing...")
    pred_nmiss_gbdt, lb_nmiss_gbdt = cv_train_predict_lgbm(
        tr_nmiss, te_nmiss, FEATURES, model_type="gbdt",
        n_splits=CONFIG['N_SPLITS'], bag_rounds=CONFIG['BAG_ROUNDS'],
        seed=CONFIG['SEED'], save_models=True, subset_name="not_missing"
    )

    print("\n[2/3] LGBM-DART on Not-Missing...")
    pred_nmiss_dart, lb_nmiss_dart = cv_train_predict_lgbm(
        tr_nmiss, te_nmiss, FEATURES, model_type="dart",
        n_splits=CONFIG['N_SPLITS'], bag_rounds=CONFIG['BAG_ROUNDS'],
        seed=CONFIG['SEED'], save_models=True, subset_name="not_missing"
    )

    print("\n[3/3] CatBoost on Not-Missing...")
    pred_nmiss_cat, lb_nmiss_cat = cv_train_predict_catboost(
        tr_nmiss, te_nmiss, FEATURES,
        n_splits=CONFIG['N_SPLITS'], bag_rounds=CONFIG['BAG_ROUNDS'],
        seed=CONFIG['SEED'], save_models=True, subset_name="not_missing"
    )

    # 결과 저장
    results = {
        'missing': {
            'gbdt': {'pred': pred_miss_gbdt, 'score': lb_miss_gbdt},
            'dart': {'pred': pred_miss_dart, 'score': lb_miss_dart},
            'catboost': {'pred': pred_miss_cat, 'score': lb_miss_cat}
        },
        'not_missing': {
            'gbdt': {'pred': pred_nmiss_gbdt, 'score': lb_nmiss_gbdt},
            'dart': {'pred': pred_nmiss_dart, 'score': lb_nmiss_dart},
            'catboost': {'pred': pred_nmiss_cat, 'score': lb_nmiss_cat}
        },
        'test_ids': {
            'missing': te_miss["ID"].values,
            'not_missing': te_nmiss["ID"].values
        }
    }

    with open(Path(CONFIG['OUTPUT_DIR']) / 'training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Configuration 저장
    with open(Path(CONFIG['OUTPUT_DIR']) / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)

    print("\n" + "="*60)
    print("=== TRAINING COMPLETE ===")
    print("="*60)
    print(f"Models saved in: {CONFIG['MODEL_DIR']}")
    print(f"Results saved in: {CONFIG['OUTPUT_DIR']}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(CONFIG['SEED'])
    main()