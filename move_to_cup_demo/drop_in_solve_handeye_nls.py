#!/usr/bin/env python3
import os, glob, json, math
import numpy as np
import cv2
from scipy.optimize import least_squares

# ---------- 경로 ----------
DATA_DIR = "calib_records"
OUT_JSON = "handeye_result.json"
OUT_NPY  = "handeye_result.npz"

# ---------- 패턴 (비대칭 원격자) ----------
PATTERN = dict(type="asymmetric", cols=4, rows=5, spacing=0.02)

# ---------- 품질 필터 ----------
REPROJ_T = 1.0
DEPTH_R_T = 0.02
MISS_T   = 2
Z_RANGE  = (0.35, 1.5)

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

def rodrigues_to_R(rvec):
    rvec = np.asarray(rvec, dtype=float).reshape(3,1)
    Rm,_ = cv2.Rodrigues(rvec)
    return Rm

def h_from_Rt(Rm, t):
    H = np.eye(4)
    H[:3,:3] = Rm
    H[:3, 3] = np.asarray(t).reshape(3)
    return H

def inv_h(H):
    R = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4); Hi[:3,:3] = R.T; Hi[:3,3] = -R.T@t
    return Hi

def load_records(folder):
    recs = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(p,"r") as f: d=json.load(f)
            bd = d.get("board_detection", {})
            rvec = bd.get("rvec"); tvec = bd.get("tvec")
            Hgb  = d.get("gripper_pose_base", {}).get("H")
            if rvec is None or tvec is None or Hgb is None: continue
            H_cam_board = h_from_Rt(rodrigues_to_R(rvec), tvec)     # {}^{cam}H_{board}
            H_base_grip = np.asarray(Hgb, dtype=float)              # {}^{base}H_{grip}
            diag = bd.get("diagnostics", {})
            recs.append(dict(
                path=p, Hcb=H_cam_board, Hbg=H_base_grip,
                reproj=diag.get("mean_reproj_px"),
                dres =diag.get("mean_depth_resid_m"),
                miss =diag.get("n_missing_depth"),
                z    =float(tvec[2])
            ))
        except Exception as e:
            print("[WARN] skip", p, e)
    return recs

def auto_keep(r):
    ok=True
    if r["reproj"] is not None and r["reproj"]>REPROJ_T: ok=False
    if r["dres"]  is not None and r["dres"] >DEPTH_R_T: ok=False
    if r["miss"]  is not None and r["miss"] >MISS_T:    ok=False
    if r["z"]     is not None and (r["z"]<Z_RANGE[0] or r["z"]>Z_RANGE[1]): ok=False
    return ok

# === 데이터 로드 ===
records = load_records(DATA_DIR)
assert len(records)>=2, "not enough records"

sel = [r for r in records if auto_keep(r)]
if len(sel)<8:
    print(f"[INFO] auto_keep {len(sel)} < 8, using all {len(records)}")
    sel = records
print(f"[INFO] using {len(sel)} / {len(records)} samples")

# === 보드 3D 포인트 ===
objp = make_objp(PATTERN["type"], PATTERN["cols"], PATTERN["rows"], PATTERN["spacing"])
objp_h = np.hstack([objp, np.ones((objp.shape[0],1))])  # Nx4

# === 파라메터화 (rvec+translation) ===
def h_from_x(x6):
    r = x6[:3].reshape(3,1); t = x6[3:].reshape(3,1)
    Rm,_ = cv2.Rodrigues(r)
    return h_from_Rt(Rm, t)

def pack6_from_H(H):
    Rm = H[:3,:3]; t = H[:3,3]
    rvec,_ = cv2.Rodrigues(Rm)
    return np.hstack([rvec.reshape(-1), t.reshape(-1)])

# === 초기값 ===
# 대충: H_base_cam ~= [I | 평균(base*grip*board*cam) 차이] 수준으로 잡아도,
# 비선형 최소제곱이 수렴한다. 여긴 보수적으로 0로 둔다.
x_base_cam_0 = np.zeros(6)
# H_grip_board는 첫 샘플로 근사: Hbg^-1 * H_base_cam^-1 * Hcb  (지금은 H_base_cam=I)
Hgb0 = inv_h(sel[0]["Hbg"]) @ sel[0]["Hcb"]
x_grip_board_0 = pack6_from_H(Hgb0)
x0 = np.hstack([x_base_cam_0, x_grip_board_0])

# === 잔차 함수 ===
def residuals(x):
    x_bc = x[:6]; x_gb = x[6:]
    Hbc = h_from_x(x_bc)      # {}^{base}H_{cam}
    Hgb = h_from_x(x_gb)      # {}^{grip}H_{board}
    res = []
    for r in sel:
        Hcb = r["Hcb"]; Hbg = r["Hbg"]
        # 경로1: board -> cam -> base
        P1 = (Hbc @ Hcb @ objp_h.T).T[:, :3]
        # 경로2: board -> grip -> base
        P2 = (Hbg @ Hgb @ objp_h.T).T[:, :3]
        res.append((P1 - P2).reshape(-1))
    return np.concatenate(res)

# === 최적화 ===
opt = least_squares(residuals, x0, method="lm")  # LM가 빠르고 안정적
x_opt = opt.x
H_base_cam = h_from_x(x_opt[:6])
H_grip_board = h_from_x(x_opt[6:])

# === 최종 RMSE ===
def eval_rmse(Hbc, Hgb):
    errs=[]
    for r in sel:
        Hcb=r["Hcb"]; Hbg=r["Hbg"]
        P1=(Hbc @ Hcb @ objp_h.T).T[:,:3]
        P2=(Hbg @ Hgb @ objp_h.T).T[:,:3]
        e=np.linalg.norm(P1-P2, axis=1)
        errs.append(e)
    e=np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2))), float(np.median(e)), float(np.max(e))

rmse, med, mx = eval_rmse(H_base_cam, H_grip_board)

np.set_printoptions(suppress=True, precision=6)
print("\n=== NLS SOLUTION ===")
print("H_base_cam =\n", H_base_cam)
print("H_grip_board =\n", H_grip_board)
print("\n=== CONSISTENCY (board points in base) ===")
print(f"RMSE = {rmse*1000:.2f} mm,  median = {med*1000:.2f} mm,  max = {mx*1000:.2f} mm")

# === 저장 ===
out = {
    "H_base_cam": H_base_cam.tolist(),
    "H_cam_base": inv_h(H_base_cam).tolist(),
    "H_grip_board": H_grip_board.tolist(),
    "stats": {
        "rmse_m": rmse, "median_m": med, "max_m": mx,
        "n_samples": len(sel), "n_total": len(records),
        "opt_cost": float(opt.cost), "opt_nfev": int(opt.nfev)
    },
    "pattern": PATTERN,
    "notes": "Nonlinear least squares over board 3D points. Use H_base_cam: p_base = H_base_cam @ p_cam_h"
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
np.savez(OUT_NPY, H_base_cam=H_base_cam, H_cam_base=inv_h(H_base_cam), H_grip_board=H_grip_board)
print(f"Saved: {OUT_JSON}, {OUT_NPY}")
