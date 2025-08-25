# train_rbc_from_csv.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import tifffile as tiff

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes
import trimesh

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# IO helpers
# -----------------------
def read_tif_2d(path: str|Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    if arr.ndim == 3:  # (H,W,1)
        arr = arr[...,0]
    return arr.astype(np.float32)

def read_tif_3d(path: str|Path) -> np.ndarray:
    with tiff.TiffFile(str(path)) as tf:
        s = tf.series[0]
        axes = getattr(s, "axes", "") or ""
        a = s.asarray()

    a = np.asarray(a)
    a = np.squeeze(a)

    vol = None

    # Case 1) 이미 3D면 그대로 사용
    if a.ndim == 3:
        vol = a
    else:
        # Case 2) 축 문자열이 있고 Y/X가 보이면, Z 후보를 Y/X 앞의 마지막 유효축으로 간주
        if axes and ('Y' in axes and 'X' in axes):
            ix = axes.rfind('X')
            iy = axes.rfind('Y')
            if ix == -1 or iy == -1:
                pass
            else:
                # Y와 X 중 더 앞에 있는 쪽(min(iy,ix))의 "직전"에서, 길이가 1보다 큰 축을 Z로 선택
                zcand = None
                for i in range(min(iy, ix) - 1, -1, -1):
                    if a.shape[i] > 1:
                        zcand = i
                        break

                if zcand is not None:
                    # z,y,x는 slice(None), 나머지는 0으로 고정해서 3D 볼륨 추출
                    indexer = []
                    kept = []
                    for i in range(a.ndim):
                        if i in (zcand, iy, ix):
                            indexer.append(slice(None))
                            kept.append(i)
                        else:
                            indexer.append(0)
                    vol = a[tuple(indexer)]

                    # 현재 vol의 축 순서는 kept(오름차순)와 동일 → (Z,Y,X) 순서로 재배열
                    order = [kept.index(zcand), kept.index(iy), kept.index(ix)]
                    vol = np.transpose(vol, order)

        # Case 3) 축 정보가 부실(QQQYX 등)하거나 위에서 못 만들면: 마지막 3축을 (Z,Y,X)로 간주
        if vol is None:
            if a.ndim >= 3:
                # 앞쪽 축은 전부 0으로 고정하고, "마지막 3축"을 유지
                idx = ( [0] * (a.ndim - 3) ) + [slice(None), slice(None), slice(None)]
                vol = a[tuple(idx)]
            else:
                raise ValueError(f"Cannot form 3D volume from shape={a.shape}, axes='{axes}'")

    # 최종 검증
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X), got shape={vol.shape}, axes='{axes}'")

    # 이진화 (확률/연속값 가능성)
    if vol.dtype == np.bool_:
        vol = vol.astype(np.uint8)
    elif np.issubdtype(vol.dtype, np.integer):
        vol = (vol > 0).astype(np.uint8)
    else:
        vmax = float(vol.max()) if vol.size else 0.0
        thr = 0.5 * vmax if vmax > 0 else 0.0
        vol = (vol > thr).astype(np.uint8)

    return vol

# -----------------------
# 2D features
# -----------------------
def features_2d(img: np.ndarray, m2d: np.ndarray) -> Dict[str, float]:
    m2d = (m2d > 0).astype(np.uint8)
    imgm = img * m2d
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

    # GLCM (16레벨 양자화, d=1, angle=0)
    imgn = imgm.copy()
    imx = imgn.max()
    if imx > 0:
        imgn = (imgn / imx * 15).astype(np.uint8)
    gl = graycomatrix(imgn, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    gl_contrast = float(graycoprops(gl, 'contrast')[0,0])
    gl_corr     = float(graycoprops(gl, 'correlation')[0,0])
    gl_diss     = float(graycoprops(gl, 'dissimilarity')[0,0])

    # 경계 특성
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

# -----------------------
# 3D features
# -----------------------
def mesh_from_binary(vol: np.ndarray, spacing=(1.0,1.0,1.0)):
    if vol.sum() < 10:
        # 너무 작은 객체 보호
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.int32)
    verts, faces, *_ = marching_cubes(vol>0, level=0.5, spacing=spacing)
    return verts, faces

def surface_area_from_mesh(verts: np.ndarray, faces: np.ndarray) -> float:
    if len(verts)==0 or len(faces)==0:
        return 0.0
    tri = verts[faces]
    a = tri[:,1]-tri[:,0]
    b = tri[:,2]-tri[:,0]
    area = 0.5*np.linalg.norm(np.cross(a,b), axis=1).sum()
    return float(area)

def convex_hull_area(verts: np.ndarray) -> float:
    if len(verts)==0:
        return 0.0
    tm = trimesh.Trimesh(vertices=verts, process=False)
    hull = tm.convex_hull
    return float(hull.area)

def inertia_eigs(vol: np.ndarray) -> Tuple[float,float,float]:
    coords = np.argwhere(vol>0)
    if coords.shape[0] < 10:
        return (0.0,0.0,0.0)
    cov = np.cov(coords.T)
    w,_ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 1e-8))
    return float(w[0]), float(w[1]), float(w[2])

def sphericity(volume: float, surface_area: float) -> float:
    if volume<=0 or surface_area<=0:
        return 0.0
    return (np.pi**(1/3.0))*((6.0*volume)**(2/3.0))/surface_area

def roughness_ratio(surface_area: float, hull_area: float) -> float:
    if hull_area<=0:
        return 0.0
    return surface_area/hull_area

def features_3d(vol: np.ndarray) -> Dict[str, float]:
    vol = (vol>0).astype(np.uint8)
    V = float(vol.sum())
    verts, faces = mesh_from_binary(vol)
    A = surface_area_from_mesh(verts, faces)
    H = convex_hull_area(verts)
    lam1, lam2, lam3 = inertia_eigs(vol)
    elong = np.sqrt((lam3+1e-8)/(lam1+1e-8)) if lam1>0 else 0.0
    flat  = np.sqrt((lam2+1e-8)/(lam1+1e-8)) if lam1>0 else 0.0
    return {
        "vol_vox": V,
        "area_surf": A,
        "sphericity": sphericity(V, A),
        "roughness_ratio": roughness_ratio(A, H),
        "inertia_eig1": lam1, "inertia_eig2": lam2, "inertia_eig3": lam3,
        "elongation": elong, "flatness": flat,
    }
# -----------------------
#  2D / 3D and Fused feature
# -----------------------
def build_2d_table(wide_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wide_csv).dropna(subset=["image_path", "mask_path", "class_6"])
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
            print(f"[2D-TBL] skip {r.filename}: {e}")
    tab2d = pd.DataFrame(rows)
    # 2D 컬럼에 접두어(선택사항) – 충돌 방지용
    two_d_cols = [c for c in tab2d.columns if c not in ["filename","class_6"]]
    tab2d = tab2d[["filename","class_6"] + two_d_cols]
    tab2d.rename(columns={c: f"twoD_{c}" for c in two_d_cols}, inplace=True)
    return tab2d


def build_3d_table(long_csv: Path, agg="mean") -> pd.DataFrame:
    df = pd.read_csv(long_csv)
    df = df[(df["pred_path"].astype(str) != "") & df["pred_path"].notna()]
    groups = []
    for fname, g in df.groupby("filename"):
        rows = []
        for p in g["pred_path"]:
            try:
                vol = read_tif_3d(p)  # (Z,Y,X) 이진 볼륨
                rows.append(features_3d(vol))
            except Exception as e:
                print(f"[3D-TBL] skip {p}: {e}")
        if not rows:
            continue
        vec = aggregate_feats(pd.DataFrame(rows), how=agg)
        d = vec.to_dict()
        d["filename"] = fname
        groups.append(d)
    tab3d = pd.DataFrame(groups)
    # 3D 컬럼에 접두어(선택사항)
    three_d_cols = [c for c in tab3d.columns if c != "filename"]
    tab3d = tab3d[["filename"] + three_d_cols]
    tab3d.rename(columns={c: f"threeD_{c}" for c in three_d_cols}, inplace=True)
    return tab3d

# -----------------------
# Train/Eval – 2D
# -----------------------
def train_2d(wide_csv: Path, test_size=0.2, seed=42):
    df = pd.read_csv(wide_csv)
    # label 선택: class_6 (네 CSV에서 6-class로 묶음)
    if "class_6" not in df.columns:
        raise ValueError("CSV에 class_6 컬럼이 없습니다. 이전 단계 스크립트로 생성된 CSV를 사용하세요.")
    df = df.dropna(subset=["image_path","mask_path","class_6"])
    X_rows, y = [], []
    for r in df.itertuples(index=False):
        try:
            img = read_tif_2d(r.image_path)
            msk = read_tif_2d(r.mask_path)
            feat = features_2d(img, msk)
            X_rows.append(feat)
            y.append(r.class_6)
        except Exception as e:
            print(f"[2D] skip {r.filename}: {e}")
    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.array(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # 클래스 불균형 대응
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
    weight_map = {c:w for c,w in zip(classes, cw)}

    clf = RandomForestClassifier(
        n_estimators=1200, max_depth=12, random_state=seed, n_jobs=-1,
        class_weight=weight_map
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1m = f1_score(y_te, y_pred, average='macro')
    print("\n=== [2D] Results ===")
    print(f"macro F1 = {f1m:.4f}")
    print(classification_report(y_te, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred, labels=classes))
    return clf, classes

# -----------------------
# Train/Eval – 3D
# -----------------------
def aggregate_feats(feat_df: pd.DataFrame, how="mean"):
    if how=="mean":
        return feat_df.mean(axis=0, numeric_only=True)
    if how=="median":
        return feat_df.median(axis=0, numeric_only=True)
    raise ValueError("how must be 'mean' or 'median'.")

def train_3d(long_csv: Path, test_size=0.2, seed=42, agg="mean"):
    df = pd.read_csv(long_csv)
    # 예측 없는 행 제거
    df = df[(df["pred_path"].astype(str)!="") & df["pred_path"].notna()]
    if df.empty:
        raise ValueError("pred_path가 비어 있습니다. DISPR 예측이 있는 rbc_unified_long.csv를 사용하세요.")
    # 샘플 단위로 앙상블 집계
    groups = []
    labels = []
    filenames = []
    for name, g in df.groupby("filename"):
        y = g["class_6"].iloc[0]
        # 각 앙상블 볼륨에서 3D feature 추출 → 집계
        rows = []
        for p in g["pred_path"]:
            try:
                vol = read_tif_3d(p)
                rows.append(features_3d(vol))
            except Exception as e:
                print(f"[3D] skip {p}: {e}")
        if len(rows)==0:
            continue
        feat_df = pd.DataFrame(rows)
        vec = aggregate_feats(feat_df, how=agg)
        groups.append(vec)
        labels.append(y)
        filenames.append(name)

    X = pd.DataFrame(groups).fillna(0.0)
    y = np.array(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
    weight_map = {c:w for c,w in zip(classes, cw)}

    clf = RandomForestClassifier(
        n_estimators=1200, max_depth=14, random_state=seed, n_jobs=-1,
        class_weight=weight_map
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1m = f1_score(y_te, y_pred, average='macro')
    print("\n=== [3D] Results (agg:", agg, ") ===")
    print(f"macro F1 = {f1m:.4f}")
    print(classification_report(y_te, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred, labels=classes))
    return clf, classes
# -----------------------
# Train/Eval – 2d + 3D fused
# -----------------------
def train_fused(wide_csv: Path, long_csv: Path, test_size=0.2, seed=42, agg="mean"):
    # 개별 테이블 생성
    tab2d = build_2d_table(wide_csv)
    tab3d = build_3d_table(long_csv, agg=agg)

    # filename 기준 교집합 병합 (라벨은 2D 테이블의 class_6 사용)
    fused = pd.merge(tab2d, tab3d, on="filename", how="inner")
    if fused.empty:
        raise ValueError("2D/3D 공통 파일이 없습니다. CSV 경로/내용을 확인하세요.")

    y = fused["class_6"].values
    drop_cols = ["filename", "class_6"]
    X = fused.drop(columns=drop_cols).fillna(0.0)

    # 고정된 재현성 split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # 클래스 불균형 보정
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    weight_map = {c: w for c, w in zip(classes, cw)}

    clf = RandomForestClassifier(
        n_estimators=1400, max_depth=16, random_state=seed, n_jobs=-1,
        class_weight=weight_map
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1m = f1_score(y_te, y_pred, average="macro")

    print("\n=== [2D+3D Fused] Results (agg:", agg, ") ===")
    print(f"macro F1 = {f1m:.4f}")
    print(classification_report(y_te, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred, labels=classes))
    return clf, classes
# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide_csv", type=str, default="/content/dataset_csvs/rbc_unified_wide.csv")
    ap.add_argument("--long_csv", type=str, default="/content/dataset_csvs/rbc_unified_long.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agg", type=str, default="mean", choices=["mean","median"],
                    help="3D 앙상블 feature 집계 방식")
    args = ap.parse_args()

    print("[INFO] 2D training with:", args.wide_csv)
    clf2d, cls2d = train_2d(Path(args.wide_csv), test_size=args.test_size, seed=args.seed)

    print("\n[INFO] 3D training with:", args.long_csv)
    clf3d, cls3d = train_3d(Path(args.long_csv), test_size=args.test_size, seed=args.seed, agg=args.agg)

    print("\n[INFO] 2D+3D fused training with:", args.wide_csv, "and", args.long_csv)
    clff, clsf = train_fused(Path(args.wide_csv), Path(args.long_csv),
                             test_size=args.test_size, seed=args.seed, agg=args.agg)

if __name__ == "__main__":
    main()
