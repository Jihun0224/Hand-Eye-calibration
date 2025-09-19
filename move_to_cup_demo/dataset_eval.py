#!/usr/bin/env python3
import os, json, glob, math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd

# ====== 사용자 설정 ======
DATA_DIR = "calib_records"          # JSON들이 있는 폴더
OUT_CSV  = "handeye_dataset_eval.csv"
PATTERN  = {"type":"asymmetric", "cols":4, "rows":5, "spacing":0.02}  # 참고용(재투영 계산시 사용 가능)

# ====== 유틸 ======
def rodrigues_to_R(rvec):
    rvec = np.asarray(rvec, dtype=float).reshape(3,1)
    Rm,_ = cv2.Rodrigues(rvec)
    return Rm

def to_hmat(Rm, t):
    H = np.eye(4, dtype=float)
    H[:3,:3] = Rm
    H[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return H

def inv_h(H):
    Rm = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4)
    Hi[:3,:3] = Rm.T
    Hi[:3, 3] = -Rm.T @ t
    return Hi

def rot_angle(Rm):
    # 회전 각도(라디안)
    # angle = arccos((trace(R)-1)/2)
    tr = np.trace(Rm)
    val = (tr - 1.0) * 0.5
    val = min(1.0, max(-1.0, val))
    return math.acos(val)

def pose_delta(Ha, Hb):
    # a->b 상대변환
    return inv_h(Ha) @ Hb

def trans_norm(H):
    return float(np.linalg.norm(H[:3,3]))

def deg(rad): return rad*180.0/math.pi

def make_objp(ptype, cols, rows, spacing):
    if ptype == "asymmetric":
        objp = np.zeros((cols*rows,3), np.float64)
        k=0
        for j in range(rows):
            for i in range(cols):
                x = (2*i + (j%2)) * spacing
                y = j * spacing
                objp[k,0]=x; objp[k,1]=y; k+=1
        return objp
    else:
        grid = np.mgrid[0:cols,0:rows].T.reshape(-1,2).astype(np.float64)
        objp = np.zeros((cols*rows,3), np.float64)
        objp[:,:2] = grid * spacing
        return objp

# ====== JSON 로드 ======
records = []
for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
    try:
        with open(p,"r") as f:
            d = json.load(f)
        bd = d.get("board_detection", {})
        rvec = bd.get("rvec", None)
        tvec = bd.get("tvec", None)
        Hgb  = d.get("gripper_pose_base", {}).get("H", None)

        if rvec is None or tvec is None or Hgb is None:
            continue

        H_cam_board = to_hmat(rodrigues_to_R(rvec), tvec)   # {}^{cam}H_{board}
        H_base_grip = np.asarray(Hgb, dtype=float)          # {}^{base}H_{grip}

        diag = bd.get("diagnostics", {})
        rec = {
            "path": p,
            "image": d.get("image"),
            "mean_reproj_px": diag.get("mean_reproj_px"),
            "mean_depth_resid_m": diag.get("mean_depth_resid_m"),
            "n_missing_depth": diag.get("n_missing_depth"),
            "n_depth_far": diag.get("n_depth_far"),
            "keep_sample": diag.get("keep_sample"),
            "board_z_cam": float(tvec[2]),
            "H_cam_board": H_cam_board,
            "H_base_grip": H_base_grip,
        }
        records.append(rec)
    except Exception as e:
        print("[WARN] load failed:", p, e)

if len(records)==0:
    raise SystemExit("No valid JSON found in %s" % DATA_DIR)

# ====== 개별 샘플 통계 ======
for rec in records:
    Ra = rec["H_cam_board"][:3,:3]; Rb = rec["H_base_grip"][:3,:3]
    rec["rot_cam_board_deg"] = deg(rot_angle(Ra))
    rec["rot_base_grip_deg"] = deg(rot_angle(Rb))  # 절대 회전량 자체는 크게 의미 없지만 참고용
    rec["trans_base_grip_m"] = trans_norm(rec["H_base_grip"])

df = pd.DataFrame([{
    "image": r["image"],
    "path": r["path"],
    "keep": bool(r["keep_sample"]),
    "reproj_px": r["mean_reproj_px"],
    "depth_resid_m": r["mean_depth_resid_m"],
    "missing_depth": r["n_missing_depth"],
    "depth_far": r["n_depth_far"],
    "board_z_cam_m": r["board_z_cam"],
    "trans_grip_m": r["trans_base_grip_m"],
} for r in records])

# ====== 품질 기준에 따른 자동 라벨 ======
REPROJ_T = 1.0      # px
DEPTH_R_T = 0.02    # m
MISS_T = 2
Z_RANGE = (0.35, 1.5)  # 보드-카메라 거리 합리 범위

reasons = []
keep_mask = []
for r in records:
    ok = True; why=[]
    rp = r["mean_reproj_px"]; dr = r["mean_depth_resid_m"]
    md = r["n_missing_depth"]; z  = r["board_z_cam"]
    if rp is not None and rp > REPROJ_T: ok=False; why.append(f"reproj>{REPROJ_T}")
    if dr is not None and dr > DEPTH_R_T: ok=False; why.append(f"depth_resid>{DEPTH_R_T}")
    if md is not None and md > MISS_T: ok=False; why.append(f"missing_depth>{MISS_T}")
    if z is not None and (z<Z_RANGE[0] or z>Z_RANGE[1]): ok=False; why.append("board_z_out_of_range")
    keep_mask.append(ok)
    reasons.append(";".join(why))

df["auto_keep"] = keep_mask
df["reject_reason"] = reasons

# ====== 포즈 다양성(상대변환) 검사 ======
# A_ij = (cam_board_i)^-1 * cam_board_j
# B_ij = (base_grip_i)^-1 * base_grip_j
rotA, transA, rotB, transB = [], [], [], []
idx_pairs = []
use_idx = [i for i,k in enumerate(keep_mask) if k]  # 자동 keep 샘플만으로 우선 평가
if len(use_idx) < 2:
    use_idx = list(range(len(records)))

for ai in range(len(use_idx)):
    for aj in range(ai+1, len(use_idx)):
        i = use_idx[ai]; j = use_idx[aj]
        Ha = records[i]["H_cam_board"]; Hb = records[j]["H_cam_board"]
        Ga = records[i]["H_base_grip"]; Gb = records[j]["H_base_grip"]
        Aij = pose_delta(Ha,Hb); Bij = pose_delta(Ga,Gb)
        rotA.append(deg(rot_angle(Aij[:3,:3])))
        rotB.append(deg(rot_angle(Bij[:3,:3])))
        transA.append(trans_norm(Aij))
        transB.append(trans_norm(Bij))
        idx_pairs.append((i,j))

rotA = np.array(rotA); rotB = np.array(rotB)
transA = np.array(transA); transB = np.array(transB)

def summarize(name, arr_deg, arr_t):
    if arr_deg.size==0:
        print(f"[{name}] no pairs")
        return
    print(f"[{name}] rot(deg) min/med/max = {arr_deg.min():.2f} / {np.median(arr_deg):.2f} / {arr_deg.max():.2f}")
    print(f"[{name}] trans(m)  min/med/max = {arr_t.min():.3f} / {np.median(arr_t):.3f} / {arr_t.max():.3f}")
    small_rot = np.sum(arr_deg < 5.0)
    print(f"[{name}] pairs with rot<5deg : {small_rot}/{arr_deg.size}")

print("=== DATASET QUALITY SUMMARY ===")
print(f"- total json: {len(records)}")
print(f"- keep (json diag): {sum(1 for r in records if r['keep_sample'])} (if present)")
print(f"- auto_keep: {df['auto_keep'].sum()} / {len(df)}")
print(f"- reproj px mean/med/max: {df['reproj_px'].mean():.3f}/{df['reproj_px'].median():.3f}/{df['reproj_px'].max():.3f}")
print(f"- depth resid m mean/med/max: {df['depth_resid_m'].mean():.4f}/{df['depth_resid_m'].median():.4f}/{df['depth_resid_m'].max():.4f}")
print(f"- board_z_cam range: {df['board_z_cam_m'].min():.3f} ~ {df['board_z_cam_m'].max():.3f}")

summarize("A(cam-board relative)", rotA, transA)
summarize("B(base-grip relative)", rotB, transB)

# 퇴화 경고
warnings = []
if np.median(rotA) < 8.0 or np.median(rotB) < 8.0:
    warnings.append("Relative rotation median < 8deg (pose diversity may be insufficient).")
if (rotA < 3.0).sum() > 0 or (rotB < 3.0).sum() > 0:
    warnings.append("Some pairs have <3deg rotation; consider removing near-static shots.")
if df["auto_keep"].sum() < max(6, len(df)//3):
    warnings.append("Too few good samples; consider collecting more or relaxing thresholds slightly.")

for w in warnings:
    print("[WARN]", w)

# CSV 저장
df.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
