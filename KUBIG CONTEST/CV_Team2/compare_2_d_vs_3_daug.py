#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two setups on RBC phenotypes (no model files saved):

A) 2D-only classifier  — features from (image, mask)
B) 3D-only with DISPR-style MINORITY AUGMENTATION (TRAIN-ONLY) — features from DISPR 3D predictions

- Group-aware evaluation by filename (StratifiedGroupKFold if possible).
- Prints macro-F1 and reports for both setups, plus a tiny summary table.

Inputs
------
1) wide_csv: columns include at least [filename, class_6, image_path, mask_path]
2) long_csv: columns include at least [filename, class_6, pred_path] (multiple rows per filename)

Usage
-----
python compare_2d_vs_3daug.py \
  --wide_csv /content/dataset_csvs/rbc_unified_wide.csv \
  --long_csv /content/dataset_csvs/rbc_unified_long.csv \
  --model rf --cv 5 --n_aug_per_sample 5 --minority_fraction 0.3

Dependencies
------------
- numpy, pandas, scikit-image, scipy, scikit-learn, tifffile
- trimesh is optional; if missing, 3D features degrade gracefully (hull/dihedral=0)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import tifffile as tiff
from skimage import filters
from skimage.measure import regionprops, marching_cubes
from skimage.morphology import binary_erosion
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage as ndi

try:
    import trimesh  # optional
except Exception:
    trimesh = None

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# =========================
# 2D IO + FEATURES
# =========================

def read_tif_2d(path: str|Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image for {path}, got {arr.shape}")
    arr = arr.astype(np.float32)
    vmax = float(arr.max())
    if vmax > 0:
        arr = arr / (255.0 if vmax > 1.0 else vmax)
    return np.nan_to_num(arr, 0.0)


def features_2d(img: np.ndarray, m2d: np.ndarray) -> Dict[str, float]:
    m2d = (np.squeeze(m2d) > 0).astype(np.uint8)
    if m2d.ndim != 2:
        raise ValueError("2D mask must be 2D")
    imgm = img.astype(np.float32) * m2d
    rp = regionprops(m2d.astype(int), intensity_image=img)
    if len(rp) == 0:
        return {k: 0.0 for k in [
            "area","perimeter","eccentricity","convex_area","solidity","mean_int","std_int",
            "glcm_contrast","glcm_corr","glcm_dissimilarity","boundary_length","boundary_roughness"
        ]}
    r = rp[0]
    area = float(r.area)
    perim = float(getattr(r, "perimeter", 0.0))
    ecc = float(getattr(r, "eccentricity", 0.0))
    cva = float(getattr(r, "convex_area", area))
    sol = float(area/(cva+1e-8))
    mean_int = float(getattr(r, "mean_intensity", float(imgm[m2d>0].mean() if area>0 else 0.0)))
    std_int = float(imgm[m2d>0].std() if area>0 else 0.0)

    # GLCM (16 levels, d=1, angle=0)
    imgn = imgm.copy()
    mx = float(imgn.max())
    if mx > 0:
        imgn = (imgn / mx * 15).astype(np.uint8)
    gl = graycomatrix(imgn, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    gl_contrast = float(graycoprops(gl, 'contrast')[0,0])
    gl_corr     = float(graycoprops(gl, 'correlation')[0,0])
    gl_diss     = float(graycoprops(gl, 'dissimilarity')[0,0])

    # Boundary
    er = binary_erosion(m2d)
    boundary = m2d ^ er
    boundary_len = float(boundary.sum())
    boundary_rough = float(boundary_len/(perim+1e-8)) if perim>0 else 0.0

    return {
        "area": area,
        "perimeter": perim,
        "eccentricity": ecc,
        "convex_area": cva,
        "solidity": sol,
        "mean_int": mean_int,
        "std_int": std_int,
        "glcm_contrast": gl_contrast,
        "glcm_corr": gl_corr,
        "glcm_dissimilarity": gl_diss,
        "boundary_length": boundary_len,
        "boundary_roughness": boundary_rough,
    }


def build_2d_table(wide_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wide_csv)
    req = {"filename","class_6","image_path","mask_path"}
    if not req.issubset(df.columns):
        raise ValueError(f"wide_csv must include {req}")
    rows = []
    for r in df.itertuples(index=False):
        try:
            img = read_tif_2d(r.image_path)
            msk = read_tif_2d(r.mask_path)
            feat = features_2d(img, msk)
            feat["filename"] = r.filename
            feat["class_6"] = r.class_6
            rows.append(feat)
        except Exception as e:
            print(f"[2D] skip {getattr(r,'filename','?')}: {e}")
    return pd.DataFrame(rows)

# =========================
# 3D IO + FEATURES
# =========================

def squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 3:
        return arr
    tmp = arr
    while tmp.ndim > 3 and 1 in tmp.shape:
        tmp = np.squeeze(tmp)
    if tmp.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X); got shape={arr.shape}")
    return tmp


def read_tif_3d(path: str|Path) -> np.ndarray:
    raw = tiff.imread(str(path))
    vol = squeeze_to_3d(raw)
    if vol.dtype not in (np.uint8, np.bool_):
        v = vol.astype(np.float32)
        v -= v.min()
        vmax = v.max() if v.max() > 0 else 1.0
        v /= vmax
        try:
            th = filters.threshold_otsu(v)
        except Exception:
            th = 0.5
        binv = (v >= th).astype(np.uint8)
    else:
        binv = (vol > 0).astype(np.uint8)
    return clean_volume(binv)


def clean_volume(vol01: np.ndarray) -> np.ndarray:
    if vol01.ndim != 3:
        raise ValueError("clean_volume expects 3D binary array")
    vol = (vol01 > 0).astype(np.uint8)
    lab, n = ndi.label(vol)
    if n > 1:
        sizes = ndi.sum(np.ones_like(vol), lab, index=np.arange(1, n+1))
        keep = 1 + int(np.argmax(sizes))
        vol = (lab == keep).astype(np.uint8)
    strc = np.ones((3,3,3), dtype=np.uint8)
    vol = ndi.binary_closing(vol, structure=strc)
    vol = ndi.binary_fill_holes(vol).astype(np.uint8)
    return vol


def mesh_from_binary(vol: np.ndarray, spacing=(1.0,1.0,1.0)) -> Tuple[np.ndarray, np.ndarray]:
    if vol.sum() < 10:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.int64)
    volf = ndi.gaussian_filter(vol.astype(np.float32), sigma=0.6)
    verts, faces, _, _ = marching_cubes(volf, level=0.5, spacing=spacing)
    return verts, faces


def tri_area(verts: np.ndarray, faces: np.ndarray) -> float:
    if len(verts)==0 or len(faces)==0:
        return 0.0
    tri = verts[faces]
    a = tri[:,1]-tri[:,0]
    b = tri[:,2]-tri[:,0]
    return float(0.5*np.linalg.norm(np.cross(a,b), axis=1).sum())


def convex_hull_area(verts: np.ndarray) -> float:
    if len(verts)==0 or trimesh is None:
        return 0.0
    try:
        tm = trimesh.Trimesh(vertices=verts, process=False)
        hull = tm.convex_hull
        return float(hull.area)
    except Exception:
        return 0.0


def inertia_eigs_from_points(points: np.ndarray) -> Tuple[float,float,float]:
    if points.shape[0] < 10:
        return (0.0,0.0,0.0)
    P = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(P.T)
    w,_ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 1e-12))
    return float(w[0]), float(w[1]), float(w[2])


def sphericity(volume_vox: float, surface_area: float) -> float:
    if volume_vox<=0 or surface_area<=0:
        return 0.0
    return (np.pi**(1/3.0))*((6.0*volume_vox)**(2/3.0))/surface_area


def features_3d(vol: np.ndarray) -> Dict[str, float]:
    vol = (vol>0).astype(np.uint8)
    V = float(vol.sum())
    verts, faces = mesh_from_binary(vol)
    A = tri_area(verts, faces)
    H = convex_hull_area(verts)

    coords = np.argwhere(vol>0)
    lam1, lam2, lam3 = inertia_eigs_from_points(coords)
    elong = np.sqrt((lam3+1e-12)/(lam1+1e-12)) if lam1>0 else 0.0
    flat  = np.sqrt((lam2+1e-12)/(lam1+1e-12)) if lam1>0 else 0.0

    # Without trimesh we can't get dihedral or radial spread reliably
    dihedral_mean = dihedral_std = dihedral_p95 = dihedral_sharp_frac = 0.0
    rad_std = 0.0
    surf_to_vol = float(A/(V+1e-8)) if V>0 else 0.0

    if trimesh is not None and len(verts)>0 and len(faces)>0:
        try:
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            ang = tm.face_adjacency_angles
            if ang is not None and len(ang)>0:
                a = np.abs(ang)
                dihedral_mean = float(a.mean())
                dihedral_std  = float(a.std())
                dihedral_p95  = float(np.percentile(a, 95.0))
                dihedral_sharp_frac = float((a > (np.pi/6)).mean())
            c = tm.vertices.mean(axis=0, keepdims=True)
            r = np.linalg.norm(tm.vertices - c, axis=1)
            rad_std = float(r.std())
            surf_to_vol = float(tm.area/(V+1e-8))
        except Exception:
            pass

    return {
        "vol_vox": V,
        "area_surf": A,
        "sphericity": sphericity(V, A),
        "hull_ratio": float(A/(H+1e-8)) if H>0 else 0.0,
        "inertia_eig1": lam1, "inertia_eig2": lam2, "inertia_eig3": lam3,
        "elongation": elong, "flatness": flat,
        "surf_to_vol": surf_to_vol,
        "rad_std": rad_std,
        "dihedral_mean": dihedral_mean,
        "dihedral_std": dihedral_std,
        "dihedral_p95": dihedral_p95,
        "dihedral_sharp_frac": dihedral_sharp_frac,
    }


def build_long3d_table(long_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(long_csv)
    req = {"filename","class_6","pred_path"}
    if not req.issubset(df.columns):
        raise ValueError(f"long_csv must include {req}")
    df = df[(df["pred_path"].astype(str)!="") & df["pred_path"].notna()].copy()
    rows = []
    for r in df.itertuples(index=False):
        try:
            vol = read_tif_3d(r.pred_path)
            feat = features_3d(vol)
            feat["filename"] = r.filename
            feat["class_6"] = r.class_6
            rows.append(feat)
        except Exception as e:
            print(f"[3D] skip {getattr(r,'pred_path','?')}: {e}")
    return pd.DataFrame(rows)


def aggregate_by_filename(tab_long: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in tab_long.columns if c not in ("filename","class_6")]
    agg = tab_long.groupby(["filename","class_6"], as_index=False)[num_cols].mean()
    return agg

# =========================
# MODELING
# =========================

def pick_model(name: str, class_weight: dict|str|None):
    n = name.lower()
    if n == "rf":
        return RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1, random_state=42,
                                      class_weight=class_weight)
    if n == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight=class_weight))
    if n == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return make_pipeline(StandardScaler(with_mean=False),
                             HistGradientBoostingClassifier(max_depth=None, learning_rate=0.06, max_iter=500, random_state=42))
    if n == "xgb":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=1200, max_depth=6, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, objective='multi:softprob', tree_method='hist',
                                 reg_lambda=1.0, random_state=42)
        except Exception:
            print("[WARN] xgboost not available; falling back to RF")
            return pick_model("rf", class_weight)
    return pick_model("rf", class_weight)


def compute_class_weights(y: np.ndarray) -> Dict[int,float]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    vc = pd.Series(y_enc).value_counts()
    n = len(y_enc)
    k = len(np.unique(y_enc))
    return {int(c): float(n/(k*max(vc.get(int(c),0),1))) for c in np.unique(y_enc)}

# =========================
# EVALUATION HELPERS
# =========================

def eval_group_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model_name: str, desc: str,
                  class_weight: bool=True, cv: int=5, seed: int=42) -> Tuple[float,float,str,np.ndarray]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    def fit_predict(tr_idx, te_idx):
        clf = pick_model(model_name, class_weight=compute_class_weights(y[tr_idx]) if class_weight and model_name.lower() in {"rf","lr"} else None)
        # sample_weight for HGB/XGB
        if model_name.lower() in {"hgb","xgb"}:
            cw = compute_class_weights(y[tr_idx])
            sw = np.array([cw[int(LabelEncoder().fit_transform(y[tr_idx])[i])] for i in range(len(tr_idx))])
            clf.fit(X.iloc[tr_idx], y_enc[tr_idx], sample_weight=sw)
        else:
            clf.fit(X.iloc[tr_idx], y_enc[tr_idx])
        yhat = clf.predict(X.iloc[te_idx])
        return yhat

    if cv and cv >= 2 and len(np.unique(groups)) >= cv:
        sgk = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
        f1s = []
        last_true = last_pred = None
        for tr_idx, te_idx in sgk.split(X, y_enc, groups):
            yhat = fit_predict(tr_idx, te_idx)
            yt = y_enc[te_idx]
            f1s.append(f1_score(yt, yhat, average='macro'))
            last_true, last_pred = yt, yhat
        rep = classification_report(le.inverse_transform(last_true), le.inverse_transform(last_pred), digits=4)
        cm = confusion_matrix(le.inverse_transform(last_true), le.inverse_transform(last_pred), labels=classes)
        print(f"\n=== [{desc}] GroupCV macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(rep)
        print("Confusion matrix:\n", cm)
        return float(np.mean(f1s)), float(np.std(f1s)), rep, cm
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(gss.split(X, y_enc, groups))
        yhat = fit_predict(tr_idx, te_idx)
        yt = y_enc[te_idx]
        f1m = f1_score(yt, yhat, average='macro')
        rep = classification_report(le.inverse_transform(yt), le.inverse_transform(yhat), digits=4)
        cm = confusion_matrix(le.inverse_transform(yt), le.inverse_transform(yhat), labels=classes)
        print(f"\n=== [{desc}] Holdout macro-F1: {f1m:.4f}")
        print(rep)
        print("Confusion matrix:\n", cm)
        return float(f1m), 0.0, rep, cm

# =========================
# PIPELINES
# =========================

def run_2d_only(wide_csv: Path, model: str, cv: int, seed: int) -> Tuple[float,float]:
    print("[INFO] Building 2D feature table…")
    tab2d = build_2d_table(wide_csv)
    tab2d = tab2d.dropna(subset=["class_6"]).reset_index(drop=True)
    X2d = tab2d.drop(columns=["filename","class_6"])  # DF
    y = tab2d["class_6"].astype(str).values
    groups = tab2d["filename"].astype(str).values
    return eval_group_cv(X2d, y, groups, model_name=model, desc="2D-only", cv=cv, seed=seed)[:2]


def run_3d_aug(wide_csv: Path, long_csv: Path, model: str, cv: int, n_aug_per_sample: int,
               minority_fraction: float, seed: int) -> Tuple[float,float]:
    print("[INFO] Extracting 3D features from long_csv predictions…")
    tab_long = build_long3d_table(long_csv)
    base_tab = aggregate_by_filename(tab_long)  # mean per filename

    # Align labels with wide_csv if present
    wide = pd.read_csv(wide_csv)
    if 'filename' in wide.columns:
        base_tab = base_tab.merge(wide[['filename','class_6']].drop_duplicates(), on='filename', how='left', suffixes=('', '_w'))
        base_tab['class_6'] = base_tab['class_6_w'].combine_first(base_tab['class_6'])
        base_tab = base_tab.drop(columns=[c for c in base_tab.columns if c.endswith('_w')])
    base_tab = base_tab.dropna(subset=['class_6']).reset_index(drop=True)

    # Prepare augmentation index by filename
    aug_by_file = {k: v for k, v in tab_long.groupby("filename")}

    # Encode labels (for minority selection only)
    le = LabelEncoder()
    y_all = le.fit_transform(base_tab['class_6'].astype(str).values)
    counts = pd.Series(y_all).value_counts().sort_values(ascending=True)
    cutoff_idx = max(1, int(np.ceil(len(counts) * minority_fraction)))
    minority_labels = set(counts.index[:cutoff_idx].tolist())

    # For CV we need to augment EACH TRAIN FOLD only
    def eval_with_aug():
        X = base_tab.drop(columns=["filename","class_6"])  # base features for eval
        y = base_tab['class_6'].astype(str).values
        groups = base_tab['filename'].astype(str).values

        le_fold = LabelEncoder().fit(y)
        y_enc_full = le_fold.transform(y)

        def fold_fit(tr_idx, te_idx):
            base_train = base_tab.iloc[tr_idx].reset_index(drop=True)
            base_test  = base_tab.iloc[te_idx].reset_index(drop=True)

            # Build augmented train rows
            aug_rows = []
            for r in base_train.itertuples(index=False):
                y_enc = le_fold.transform([r.class_6])[0]
                if y_enc in minority_labels:
                    g = aug_by_file.get(r.filename)
                    if g is not None and not g.empty:
                        take = min(n_aug_per_sample, len(g))
                        sel = g.sample(n=take, replace=False, random_state=seed)
                        for rr in sel.itertuples(index=False):
                            feat = {k: getattr(rr, k) for k in g.columns if k not in ("filename","class_6")}
                            feat["filename"] = r.filename
                            feat["class_6"] = r.class_6
                            aug_rows.append(feat)
            aug_df = pd.DataFrame(aug_rows)
            train_full = base_train if aug_df.empty else pd.concat([base_train, aug_df], axis=0, ignore_index=True)

            Xtr = train_full.drop(columns=["filename","class_6"]).values
            ytr = le_fold.transform(train_full['class_6'].astype(str).values)
            Xte = base_test.drop(columns=["filename","class_6"]).values
            yte = le_fold.transform(base_test['class_6'].astype(str).values)

            # Model + class weights
            cw = None
            if model.lower() in {"rf","lr"}:
                vc = pd.Series(ytr).value_counts()
                n = len(ytr); k = len(vc)
                cw = {int(c): float(n/(k*max(vc.get(int(c),0),1))) for c in vc.index}
            clf = pick_model(model, class_weight=cw)
            if model.lower() in {"hgb","xgb"}:
                sw = np.array([cw.get(int(c),1.0) for c in ytr]) if cw else None
                clf.fit(Xtr, ytr, sample_weight=sw)
            else:
                clf.fit(Xtr, ytr)

            yhat = clf.predict(Xte)
            return yte, yhat, le_fold.classes_

        if cv and cv >= 2 and len(np.unique(groups)) >= cv:
            sgk = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
            f1s = []
            last_true = last_pred = None
            last_classes = None
            for tr_idx, te_idx in sgk.split(X, y_enc_full, groups):
                yte, yhat, cls = fold_fit(tr_idx, te_idx)
                f1s.append(f1_score(yte, yhat, average='macro'))
                last_true, last_pred, last_classes = yte, yhat, cls
            rep = classification_report(last_true, last_pred, target_names=[str(c) for c in last_classes], digits=4)
            cm = confusion_matrix(last_true, last_pred)
            print(f"\n=== [3D-only + DISPR AUG] GroupCV macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(rep)
            print("Confusion matrix (encoded labels order):\n", cm)
            return float(np.mean(f1s)), float(np.std(f1s))
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr_idx, te_idx = next(gss.split(X.values, y_enc_full, base_tab['filename'].astype(str).values))
            yte, yhat, cls = fold_fit(tr_idx, te_idx)
            f1m = f1_score(yte, yhat, average='macro')
            rep = classification_report(yte, yhat, target_names=[str(c) for c in cls], digits=4)
            cm = confusion_matrix(yte, yhat)
            print(f"\n=== [3D-only + DISPR AUG] Holdout macro-F1: {f1m:.4f}")
            print(rep)
            print("Confusion matrix (encoded labels order):\n", cm)
            return float(f1m), 0.0

    return eval_with_aug()

# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide_csv', type=str, required=True)
    ap.add_argument('--long_csv', type=str, required=True)
    ap.add_argument('--model', type=str, default='rf', choices=['rf','lr','hgb','xgb'])
    ap.add_argument('--cv', type=int, default=5)
    ap.add_argument('--n_aug_per_sample', type=int, default=5)
    ap.add_argument('--minority_fraction', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # 2D-only
    f1_2d_mean, f1_2d_std = run_2d_only(Path(args.wide_csv), model=args.model, cv=args.cv, seed=args.seed)

    # 3D-only + DISPR AUG (train-only, minority classes)
    f1_3d_mean, f1_3d_std = run_3d_aug(Path(args.wide_csv), Path(args.long_csv), model=args.model,
                                       cv=args.cv, n_aug_per_sample=args.n_aug_per_sample,
                                       minority_fraction=args.minority_fraction, seed=args.seed)

    # Summary
    print("\n================ SUMMARY ================" )
    rows = [
        {"setup": "2D-only", "macroF1_mean": f1_2d_mean, "macroF1_std": f1_2d_std},
        {"setup": "3D-only + DISPR AUG (train-only)", "macroF1_mean": f1_3d_mean, "macroF1_std": f1_3d_std},
    ]
    summ = pd.DataFrame(rows)
    try:
        # Pretty print with 4 decimals
        with pd.option_context('display.float_format', '{:.4f}'.format):
            print(summ)
    except Exception:
        print(summ)

if __name__ == '__main__':
    main()
