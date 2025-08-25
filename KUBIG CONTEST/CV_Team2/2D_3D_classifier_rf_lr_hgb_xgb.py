#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAPR-like experiment script to compare 2D vs 3D vs 2D+3D-fused classification
on RBC phenotypes.

Key ideas (mirroring SHAPR-style downstream use of 3D):
- Robustly load predicted 3D volumes (handles (1,1,Z,Y,X) or similar stacks).
- Clean & stabilize the binary volume (largest CC, holes, closing).
- Extract richer 3D shape features from the mesh (area, sphericity, hull ratio,
  dihedral-angle roughness stats, principal axes, surface/volume, etc.).
- Aggregate across multiple 3D predictions per image (mean + std features).
- Compare 2D-only, 3D-only, and fused models with a group-aware split by filename.

Dependencies: numpy, pandas, scikit-image, scipy, scikit-learn, tifffile, trimesh
(Optionally: if 'xgboost' is available it will be used; otherwise falls back to RF.)
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff

from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes
from skimage import filters
from scipy import ndimage as ndi

import trimesh

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------
# IO helpers
# -------------------------------------------------------------

def read_tif_2d(path: str|Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    # squeeze (H,W,1) -> (H,W)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image for {path}, got shape={arr.shape}")
    return arr.astype(np.float32)


def squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
    """Make sure array is (Z,Y,X). We accept inputs like (1,1,Z,Y,X), (Z,Y,X,1), etc.
    The function squeezes singleton axes and validates dimensionality.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        # Some readers may return a single slice if series is misinterpreted.
        # We can't recover 3D from 2D; raise to catch data issues upstream.
        raise ValueError(f"Expected 3D (Z,Y,X); got 2D array with shape={arr.shape}.")
    if arr.ndim != 3:
        # Try to drop any 1-sized axes repeatedly
        tmp = arr
        while tmp.ndim > 3 and 1 in tmp.shape:
            tmp = np.squeeze(tmp)
        if tmp.ndim == 3:
            arr = tmp
        else:
            raise ValueError(f"Expected 3D (Z,Y,X); got shape={arr.shape}")
    # Now (Z,Y,X)
    return arr


def read_tif_3d(path: str|Path) -> np.ndarray:
    raw = tiff.imread(str(path))
    vol = squeeze_to_3d(raw)
    # Normalize & binarize robustly (Otsu if not already 0/1)
    if vol.dtype != np.uint8 and vol.dtype != np.bool_:
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
    # 3D cleanup
    return clean_volume(binv)


# -------------------------------------------------------------
# 2D features (same spirit as your previous script)
# -------------------------------------------------------------

def features_2d(img: np.ndarray, m2d: np.ndarray) -> Dict[str, float]:
    m2d = (np.squeeze(m2d) > 0).astype(np.uint8)
    if m2d.ndim != 2:
        raise ValueError("2D mask must be 2D.")
    imgm = img.astype(np.float32) * m2d
    rp = regionprops(m2d.astype(int), intensity_image=img)
    if len(rp)==0:
        return {k:0.0 for k in [
            "area","perimeter","eccentricity","convex_area","solidity","mean_int","std_int",
            "glcm_contrast","glcm_corr","glcm_dissimilarity","boundary_length","boundary_roughness"
        ]}
    r = rp[0]
    area = float(r.area)
    perim = float(getattr(r, "perimeter", 0.0))
    ecc = float(getattr(r, "eccentricity", 0.0))
    cva = float(getattr(r, "convex_area", area))
    sol = float(area/(cva+1e-8))
    mean_int = float(r.mean_intensity) if hasattr(r, "mean_intensity") else float(imgm[m2d>0].mean() if area>0 else 0.0)
    std_int = float(imgm[m2d>0].std() if area>0 else 0.0)

    # GLCM (16 levels, d=1, angle=0)
    imgn = imgm.copy()
    imx = float(imgn.max())
    if imx > 0:
        imgn = (imgn / imx * 15).astype(np.uint8)
    gl = graycomatrix(imgn, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    gl_contrast = float(graycoprops(gl, 'contrast')[0,0])
    gl_corr     = float(graycoprops(gl, 'correlation')[0,0])
    gl_diss     = float(graycoprops(gl, 'dissimilarity')[0,0])

    # Boundary features
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


# -------------------------------------------------------------
# 3D preprocessing & features (SHAPR-style downstream)
# -------------------------------------------------------------

def clean_volume(vol01: np.ndarray) -> np.ndarray:
    """Vol is 0/1 (Z,Y,X). Keep largest 3D component, fill holes, closing."""
    if vol01.ndim != 3:
        raise ValueError("clean_volume expects 3D binary array")
    vol = (vol01 > 0).astype(np.uint8)

    # Keep largest 3D component
    lab, n = ndi.label(vol)
    if n > 1:
        sizes = ndi.sum(np.ones_like(vol), lab, index=np.arange(1, n+1))
        keep_label = 1 + int(np.argmax(sizes))
        vol = (lab == keep_label).astype(np.uint8)

    # Morphological closing (3x3x3) & fill holes
    strc = np.ones((3,3,3), dtype=np.uint8)
    vol = ndi.binary_closing(vol, structure=strc)
    vol = ndi.binary_fill_holes(vol).astype(np.uint8)
    return vol


def mesh_from_binary(vol: np.ndarray, spacing=(1.0,1.0,1.0)) -> Tuple[np.ndarray, np.ndarray]:
    if vol.sum() < 10:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.int64)
    # Gentle smoothing to stabilize Marching Cubes
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
    if len(verts)==0:
        return 0.0
    tm = trimesh.Trimesh(vertices=verts, process=False)
    hull = tm.convex_hull
    return float(hull.area)


def inertia_eigs_from_points(points: np.ndarray) -> Tuple[float,float,float]:
    # points: (N,3)
    if points.shape[0] < 10:
        return (0.0,0.0,0.0)
    # center & covariance
    P = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(P.T)
    w,_ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 1e-12))
    return float(w[0]), float(w[1]), float(w[2])


def dihedral_stats(mesh: trimesh.Trimesh) -> Dict[str,float]:
    try:
        ang = mesh.face_adjacency_angles  # radians
        if ang is None or len(ang)==0:
            return {"dihedral_mean":0.0,"dihedral_std":0.0,"dihedral_p95":0.0,"dihedral_sharp_frac":0.0}
        a = np.abs(ang)
        return {
            "dihedral_mean": float(a.mean()),
            "dihedral_std":  float(a.std()),
            "dihedral_p95":  float(np.percentile(a, 95.0)),
            "dihedral_sharp_frac": float((a > (np.pi/6)).mean()),  # >30°
        }
    except Exception:
        return {"dihedral_mean":0.0,"dihedral_std":0.0,"dihedral_p95":0.0,"dihedral_sharp_frac":0.0}


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

    # Mesh object for additional descriptors
    mesh = None
    try:
        if len(verts)>0 and len(faces)>0:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception:
        mesh = None

    # Inertia eigenvalues from voxel coords (more stable than vertex only)
    coords = np.argwhere(vol>0)
    lam1, lam2, lam3 = inertia_eigs_from_points(coords)
    elong = np.sqrt((lam3+1e-12)/(lam1+1e-12)) if lam1>0 else 0.0
    flat  = np.sqrt((lam2+1e-12)/(lam1+1e-12)) if lam1>0 else 0.0

    # Dihedral angle-based roughness proxies
    dihed = {"dihedral_mean":0.0,"dihedral_std":0.0,"dihedral_p95":0.0,"dihedral_sharp_frac":0.0}
    surf2vol = 0.0
    rad_std = 0.0
    if mesh is not None and mesh.area > 0:
        dihed = dihedral_stats(mesh)
        surf2vol = float(mesh.area/(V+1e-8))
        # Radial distribution from centroid
        c = mesh.vertices.mean(axis=0, keepdims=True)
        r = np.linalg.norm(mesh.vertices - c, axis=1)
        rad_std = float(r.std())

    return {
        "vol_vox": V,
        "area_surf": A,
        "sphericity": sphericity(V, A),
        "hull_ratio": float(A/(H+1e-8)) if H>0 else 0.0,
        "inertia_eig1": lam1, "inertia_eig2": lam2, "inertia_eig3": lam3,
        "elongation": elong, "flatness": flat,
        "surf_to_vol": surf2vol,
        "rad_std": rad_std,
        **dihed,
    }


# -------------------------------------------------------------
# Aggregation utilities
# -------------------------------------------------------------

def aggregate_feat_rows(rows: List[Dict[str,float]]) -> Dict[str,float]:
    """Return mean and std for each numeric feature across ensemble rows."""
    if len(rows)==0:
        return {}
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = df.mean(axis=0, numeric_only=True)
    std  = df.std(axis=0, numeric_only=True)
    out = {}
    for k in mean.index:
        out[f"{k}_mean"] = float(mean[k])
    for k in std.index:
        out[f"{k}_std"] = float(std[k])
    return out


# -------------------------------------------------------------
# Dataset readers
# -------------------------------------------------------------

def build_2d_table(wide_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wide_csv)
    if "class_6" not in df.columns:
        raise ValueError("CSV is missing class_6 column")
    df = df.dropna(subset=["image_path","mask_path","class_6"]) 

    rows = []
    for r in df.itertuples(index=False):
        try:
            img = read_tif_2d(r.image_path)
            msk = read_tif_2d(r.mask_path)
            feat = features_2d(img, msk)
            feat["filename"] = getattr(r, "filename") if hasattr(r, "filename") else Path(r.image_path).name
            feat["class_6"] = r.class_6
            rows.append(feat)
        except Exception as e:
            print(f"[2D] skip {getattr(r, 'filename', 'unknown')}: {e}")
    out = pd.DataFrame(rows)
    return out


def build_3d_table(long_csv: Path, agg: str = "mean_std") -> pd.DataFrame:
    """From long CSV (with multiple pred_path per filename), compute 3D features.
       Returns one row per filename containing aggregated features (mean+std).
    """
    df = pd.read_csv(long_csv)
    if "pred_path" not in df.columns:
        raise ValueError("long_csv must contain pred_path column (3D predictions)")
    df = df[(df["pred_path"].astype(str)!="") & df["pred_path"].notna()]
    if df.empty:
        raise ValueError("No valid pred_path in long_csv. Ensure 3D predictions exist.")

    out_rows = []
    for name, g in df.groupby("filename"):
        # carry label
        label = g["class_6"].iloc[0] if "class_6" in g.columns else None
        per_pred_rows = []
        for p in g["pred_path"]:
            try:
                vol = read_tif_3d(p)
                per_pred_rows.append(features_3d(vol))
            except Exception as e:
                print(f"[3D] skip {p}: {e}")
        if len(per_pred_rows)==0:
            continue
        agg_row = aggregate_feat_rows(per_pred_rows)
        agg_row["filename"] = name
        if label is not None:
            agg_row["class_6"] = label
        out_rows.append(agg_row)
    return pd.DataFrame(out_rows)


# -------------------------------------------------------------
# Modeling & evaluation
# -------------------------------------------------------------

def pick_model(model_name: str, class_weight: str|dict|None = None):
    model_name = model_name.lower()
    if model_name == "lr":
        # Multinomial logistic with standardization
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight=class_weight)
        )
        return clf
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=1200, max_depth=14, n_jobs=-1, random_state=42,
            class_weight=class_weight
        )
        return clf
    if model_name == "hgb":
        # HistGradientBoosting (no class_weight) – we can upsample later if desired
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = make_pipeline(StandardScaler(with_mean=False), HistGradientBoostingClassifier(max_depth=None, learning_rate=0.06, max_iter=500, random_state=42))
        return clf
    # Try xgboost if available
    if model_name == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=1200, max_depth=6, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, objective='multi:softprob', tree_method='hist',
                reg_lambda=1.0, random_state=42
            )
            return clf
        except Exception:
            print("[WARN] xgboost not available, falling back to RF")
            return pick_model("rf", class_weight=class_weight)
    # default
    return pick_model("rf", class_weight=class_weight)


@dataclass
class SplitConfig:
    test_size: float = 0.2
    seed: int = 42
    cv_folds: int = 0   # 0 or 1 for single split; >=2 for StratifiedGroupKFold

from sklearn.preprocessing import LabelEncoder
def evaluate_table(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model_name: str, desc: str) -> None:
    # 1) 라벨을 0..K-1로 인코딩 (XGBoost 대비 + 일관성)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)                 # ex) ['SDE', ...] -> [0,1,2,3,4,5]
    classes_enc = np.unique(y_enc)              # 정수 라벨
    class_names = le.classes_                   # 문자열 라벨 (출력용)

    # 2) 클래스 가중치 계산 (정수 라벨 기준)
    weight_map = compute_class_weights(y_enc, classes_enc)  # dict: {0: w0, 1: w1, ...}

    def model_supports_class_weight(name: str) -> bool:
        return name.lower() in {"rf", "lr"}     # HGB/XGB는 class_weight 미지원(또는 부분지원)

    def fit_predict(train_idx, test_idx):
        # 모델 생성
        cw = weight_map if model_supports_class_weight(model_name) else None
        clf = pick_model(model_name, class_weight=cw)

        # sample_weight 준비 (HGB/XGB 등에 사용)
        sw = np.array([weight_map[int(c)] for c in y_enc[train_idx]])

        # 학습
        try:
            if model_name.lower() in {"hgb", "xgb"}:
                clf.fit(X.iloc[train_idx], y_enc[train_idx], sample_weight=sw)
            else:
                clf.fit(X.iloc[train_idx], y_enc[train_idx])
        except Exception as e:
            # 혹시 파이프라인/추정기별 인자 전달 문제 발생 시 안전하게 재시도
            clf = pick_model(model_name, class_weight=None)
            clf.fit(X.iloc[train_idx], y_enc[train_idx])

        # 예측 (정수 라벨)
        yhat_enc = clf.predict(X.iloc[test_idx])
        return yhat_enc

    # 3) CV (가능하면 그룹-계층 5-fold)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        from sklearn.model_selection import train_test_split
        tr_idx, te_idx = train_test_split(np.arange(len(y_enc)), test_size=0.2, stratify=y_enc, random_state=42)
        yhat_enc = fit_predict(tr_idx, te_idx)
        y_true = le.inverse_transform(y_enc[te_idx])
        y_pred = le.inverse_transform(yhat_enc)
        print(f"\n=== [{desc}] Results ===")
        print(f"macro F1 = {f1_score(y_true, y_pred, average='macro'):.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=class_names))
        return

    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        f1s = []
        last_te = None
        last_pred = None
        last_true = None
        for tr_idx, te_idx in cv.split(X, y_enc, groups):
            yhat_enc = fit_predict(tr_idx, te_idx)
            y_true = le.inverse_transform(y_enc[te_idx])
            y_pred = le.inverse_transform(yhat_enc)
            f1s.append(f1_score(y_true, y_pred, average='macro'))
            last_te, last_pred, last_true = te_idx, y_pred, y_true

        print(f"\n=== [{desc}] GroupCV Results ===")
        print(f"macro F1 (mean±sd): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(classification_report(last_true, last_pred, target_names=class_names, digits=4))
        print("Confusion matrix:\n", confusion_matrix(last_true, last_pred, labels=class_names))
    except Exception as e:
        print(f"[WARN] GroupCV failed ({e}); using GroupShuffleSplit holdout")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y_enc, groups))
        yhat_enc = fit_predict(tr_idx, te_idx)
        y_true = le.inverse_transform(y_enc[te_idx])
        y_pred = le.inverse_transform(yhat_enc)
        print(f"\n=== [{desc}] Results ===")
        print(f"macro F1 = {f1_score(y_true, y_pred, average='macro'):.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=class_names))


def report_block(title: str, y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray):
    print(f"\n=== [{title}] Results ===")
    print(f"macro F1 = {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=classes))


# ✅ 수정 (문자 라벨 OK, dict로 반환)
from typing import Any, Dict
import pandas as pd

def compute_class_weights(y: np.ndarray, classes: np.ndarray) -> Dict[Any, float]:
    """Balanced class weights: n_samples / (n_classes * count(cls))."""
    vc = pd.Series(y).value_counts()
    n = len(y)
    k = len(classes)
    return {cls: float(n / (k * max(vc.get(cls, 0), 1))) for cls in classes}

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide_csv", type=str, default="/content/dataset_csvs/rbc_unified_wide.csv")
    ap.add_argument("--long_csv", type=str, default="/content/dataset_csvs/rbc_unified_long.csv")
    ap.add_argument("--model", type=str, default="rf", choices=["rf","lr","hgb","xgb"], help="base classifier")
    ap.add_argument("--save_tables", action="store_true", help="save feature tables to CSV")
    args = ap.parse_args()

    # 1) Build feature tables
    print("[INFO] Building 2D feature table…")
    tab2d = build_2d_table(Path(args.wide_csv))
    print(f"[INFO] 2D table: {tab2d.shape}")

    print("[INFO] Building 3D feature table…")
    tab3d = build_3d_table(Path(args.long_csv))
    print(f"[INFO] 3D table: {tab3d.shape}")

    # Align by filename
    common_names = sorted(set(tab2d["filename"]) & set(tab3d["filename"]))
    tab2d_c = tab2d[tab2d["filename"].isin(common_names)].reset_index(drop=True)
    tab3d_c = tab3d[tab3d["filename"].isin(common_names)].reset_index(drop=True)

    # Ensure same order by filename
    tab2d_c = tab2d_c.sort_values("filename").reset_index(drop=True)
    tab3d_c = tab3d_c.sort_values("filename").reset_index(drop=True)

    # y labels & groups
    y2d = tab2d_c["class_6"].astype(str).values
    y3d = tab3d_c["class_6"].astype(str).values
    assert np.all(y2d == y3d), "Label mismatch after alignment"
    y = y2d
    groups = tab2d_c["filename"].values

    # X matrices
    X2d = tab2d_c.drop(columns=["filename","class_6"])  
    X3d = tab3d_c.drop(columns=["filename","class_6"])  
    Xfused = pd.concat([X2d.add_prefix("f2d_"), X3d.add_prefix("f3d_")], axis=1)

    if args.save_tables:
        Path("./feature_tables").mkdir(parents=True, exist_ok=True)
        tab2d.to_csv("./feature_tables/feat2d_all.csv", index=False)
        tab3d.to_csv("./feature_tables/feat3d_all.csv", index=False)
        Xfused.assign(class_6=y, filename=groups).to_csv("./feature_tables/feat_fused_aligned.csv", index=False)
        print("[INFO] Saved ./feature_tables/*.csv")

    # 2) Evaluate – Group-aware CV or holdout
    evaluate_table(X2d, y, groups, args.model, desc="2D-only")
    evaluate_table(X3d, y, groups, args.model, desc="3D-only")
    evaluate_table(Xfused, y, groups, args.model, desc="2D+3D Fused")


if __name__ == "__main__":
    main()
