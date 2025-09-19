#!/usr/bin/env python3
import os, glob, json, math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ---------- 사용자 경로/패턴 ----------
DATA_DIR = "calib_records"
OUT_JSON = "handeye_result.json"
OUT_NPY  = "handeye_result.npz"

# 패턴 정의 (비대칭 원격자)
PATTERN = dict(type="asymmetric", cols=4, rows=5, spacing=0.02)  # [m]

# ---------- 품질 필터(평가 스크립트와 동일/약간 보수적) ----------
REPROJ_T = 1.0     # px
DEPTH_R_T = 0.02   # m
MISS_T   = 2
Z_RANGE  = (0.35, 1.5)  # m

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

def to_h(Rm, t):
    H = np.eye(4, dtype=float)
    H[:3,:3] = np.asarray(Rm, dtype=float)
    H[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return H

def inv_h(H):
    Rm = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4)
    Hi[:3,:3] = Rm.T
    Hi[:3, 3] = -Rm.T @ t
    return Hi

def load_records(folder):
    recs = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(p, "r") as f: d = json.load(f)
            bd = d.get("board_detection", {})
            rvec = bd.get("rvec", None); tvec = bd.get("tvec", None)
            Hgb  = d.get("gripper_pose_base", {}).get("H", None)
            if rvec is None or tvec is None or Hgb is None:
                continue
            H_cam_board = to_h(rodrigues_to_R(rvec), tvec)        # {}^{cam}H_{board}
            H_base_grip = np.asarray(Hgb, dtype=float)            # {}^{base}H_{grip}
            diag = bd.get("diagnostics", {})
            recs.append(dict(
                path=p, image=d.get("image"),
                H_cam_board=H_cam_board, H_base_grip=H_base_grip,
                reproj=diag.get("mean_reproj_px"),
                dres =diag.get("mean_depth_resid_m"),
                miss =diag.get("n_missing_depth"),
                z    =float(tvec[2]),
            ))
        except Exception as e:
            print("[WARN] skip", p, e)
    return recs

def auto_keep(r):
    ok = True
    if r["reproj"] is not None and r["reproj"] > REPROJ_T: ok=False
    if r["dres"]  is not None and r["dres"]  > DEPTH_R_T: ok=False
    if r["miss"]  is not None and r["miss"]  > MISS_T:    ok=False
    if r["z"]     is not None and (r["z"]<Z_RANGE[0] or r["z"]>Z_RANGE[1]): ok=False
    return ok

# ---------- 1) 데이터 로드 & 필터 ----------
records = load_records(DATA_DIR)
if len(records) < 2:
    raise SystemExit("Not enough records")

sel = [r for r in records if auto_keep(r)]
if len(sel) < 8:   # 너무 적으면 전체 사용
    print(f"[INFO] auto_keep {len(sel)} < 8, using all {len(records)}")
    sel = records
print(f"[INFO] using {len(sel)} / {len(records)} samples")

# ---------- 2) OpenCV 용 리스트 구성 ----------
# sel 에서 행렬 리스트 뽑기
H_cam_board_list = [s["H_cam_board"] for s in sel]   # {}^{cam}H_{board}
H_base_grip_list = [s["H_base_grip"] for s in sel]   # {}^{base}H_{grip}

# OpenCV 4.9 시그니처: R_world2cam, t_world2cam, R_base2gripper, t_base2gripper
# 여기서 world == board 로 둔다.
R_t2c = [np.asarray(H[:3, :3], dtype=np.float64) for H in H_cam_board_list]               # R_world2cam
t_t2c = [np.asarray(H[:3,  3], dtype=np.float64).reshape(3,1) for H in H_cam_board_list]  # t_world2cam

# 우리 데이터는 {}^{base}H_{grip} 이므로 base→grip으로 역변환해서 넘긴다.
R_b2g, t_b2g = [], []
for Hbg in H_base_grip_list:   # {}^{base}H_{grip}
    Rbg = np.asarray(Hbg[:3, :3], dtype=np.float64)
    tbg = np.asarray(Hbg[:3,  3], dtype=np.float64).reshape(3,1)
    Rb2g = Rbg.T
    tb2g = -Rb2g @ tbg
    R_b2g.append(Rb2g)   # R_base2gripper
    t_b2g.append(tb2g)   # t_base2gripper

# 모양/개수 sanity check
assert len(R_t2c)==len(t_t2c)==len(R_b2g)==len(t_b2g)>1, "lists must have same length > 1"
for arr in (R_t2c+R_b2g):
    assert arr.shape==(3,3)
for arr in (t_t2c+t_b2g):
    assert arr.shape==(3,1)

# ---------- 3) 동시 Hand-Eye (호환 래퍼) ----------
def call_rwhe(R_t2c, t_t2c, R_b2g, t_b2g, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH):
    """
    Tries multiple OpenCV 4.x signatures:
    - kwargs:  world2cam/base2gripper   -> returns 4 or 5 items
    - positional: world2cam/base2gripper
    - legacy positional: gripper2base / target2cam (invert base2gripper)
    Returns: (R_cam2base, t_cam2base, R_grip2target, t_grip2target)
    """
    # ensure shapes/dtypes
    R_t2c = [np.asarray(R, dtype=np.float64).reshape(3,3) for R in R_t2c]
    t_t2c = [np.asarray(t, dtype=np.float64).reshape(3,1) for t in t_t2c]
    R_b2g = [np.asarray(R, dtype=np.float64).reshape(3,3) for R in R_b2g]
    t_b2g = [np.asarray(t, dtype=np.float64).reshape(3,1) for t in t_b2g]

    # 1) kwargs: world2cam/base2gripper
    try:
        out = cv2.calibrateRobotWorldHandEye(
            R_world2cam=R_t2c, t_world2cam=t_t2c,
            R_base2gripper=R_b2g, t_base2gripper=t_b2g,
            method=method
        )
        if len(out) == 4:
            R_cam2base, t_cam2base, R_grip2target, t_grip2target = out
        elif len(out) == 5:
            _retval, R_cam2base, t_cam2base, R_grip2target, t_grip2target = out
        else:
            raise RuntimeError("Unexpected return arity (kwargs world2cam/base2gripper)")
        print("[INFO] RWHE path: kwargs world2cam/base2gripper")
        return R_cam2base, t_cam2base, R_grip2target, t_grip2target
    except Exception as e:
        pass

    # 2) positional: world2cam/base2gripper
    try:
        out = cv2.calibrateRobotWorldHandEye(R_t2c, t_t2c, R_b2g, t_b2g, method)
        if len(out) == 4:
            R_cam2base, t_cam2base, R_grip2target, t_grip2target = out
        elif len(out) == 5:
            _retval, R_cam2base, t_cam2base, R_grip2target, t_grip2target = out
        else:
            raise RuntimeError("Unexpected return arity (positional world2cam/base2gripper)")
        print("[INFO] RWHE path: positional world2cam/base2gripper")
        return R_cam2base, t_cam2base, R_grip2target, t_grip2target
    except Exception as e:
        pass

    # 3) legacy positional: gripper2base / target2cam
    #    -> we must invert base2gripper to get gripper2base
    R_g2b, t_g2b = [], []
    for Rb2g, tb2g in zip(R_b2g, t_b2g):
        Rg2b = Rb2g.T
        tg2b = -Rg2b @ tb2g
        R_g2b.append(Rg2b); t_g2b.append(tg2b)
    try:
        out = cv2.calibrateRobotWorldHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method)
        if len(out) == 4:
            R_cam2base, t_cam2base, R_grip2target, t_grip2target = out
        elif len(out) == 5:
            _retval, R_cam2base, t_cam2base, R_grip2target, t_grip2target = out
        else:
            raise RuntimeError("Unexpected return arity (legacy gripper2base/target2cam)")
        print("[INFO] RWHE path: legacy positional gripper2base/target2cam")
        return R_cam2base, t_cam2base, R_grip2target, t_grip2target
    except Exception as e:
        raise SystemExit(f"[FATAL] calibrateRobotWorldHandEye failed on all paths: {e}")

# 호출
R_cam2base, t_cam2base, R_grip2target, t_grip2target = call_rwhe(R_t2c, t_t2c, R_b2g, t_b2g)

H_cam_base  = to_h(R_cam2base,    t_cam2base)      # {}^{cam}H_{base}
H_base_cam  = inv_h(H_cam_base)                    # 최종: {}^{base}H_{cam}
H_grip_board= to_h(R_grip2target, t_grip2target)   # {}^{grip}H_{board}
# ---- 방향 가설 자동 점검: 4가지 조합 테스트 ----
def compute_rmse(H_base_cam_used, H_grip_board_used, sel, objp_h):
    errs_all = []
    for s in sel:
        Hcb = s["H_cam_board"]   # {}^{cam}H_{board}
        Hbg = s["H_base_grip"]   # {}^{base}H_{grip}
        # 경로1: board -> cam -> base
        P1 = (H_base_cam_used @ Hcb @ objp_h.T).T[:, :3]
        # 경로2: board -> grip -> base
        P2 = (Hbg @ H_grip_board_used @ objp_h.T).T[:, :3]
        e = np.linalg.norm(P1 - P2, axis=1)
        errs_all.append(e)
    errs_all = np.concatenate(errs_all)
    rmse = float(np.sqrt(np.mean(errs_all**2)))
    med  = float(np.median(errs_all))
    mx   = float(np.max(errs_all))
    return rmse, med, mx

objp = make_objp(PATTERN["type"], PATTERN["cols"], PATTERN["rows"], PATTERN["spacing"])
objp_h = np.hstack([objp, np.ones((objp.shape[0],1))])

# 4가지 조합:
#  (A) H_base_cam,        H_grip_board
#  (B) H_base_cam,        inv(H_grip_board)
#  (C) inv(H_base_cam)=H_cam_base, H_grip_board
#  (D) inv(H_base_cam)=H_cam_base, inv(H_grip_board)
H_cam_board_inv = None  # just placeholder, not used
H_grip_board_inv = inv_h(H_grip_board)
H_cam_base_inv = inv_h(H_base_cam)  # == H_cam_base

combos = {
    "A(base_cam, grip_board)"     : (H_base_cam,        H_grip_board),
    "B(base_cam, inv_grip_board)" : (H_base_cam,        H_grip_board_inv),
    "C(cam_base, grip_board)"     : (H_cam_base_inv,    H_grip_board),
    "D(cam_base, inv_grip_board)" : (H_cam_base_inv,    H_grip_board_inv),
}

results = {}
for name, (Hbc_use, Hgb_use) in combos.items():
    rmse, med, mx = compute_rmse(Hbc_use, Hgb_use, sel, objp_h)
    results[name] = (rmse, med, mx)

print("\n=== ORIENTATION HYPOTHESES ===")
best_name = None; best_rmse = 1e9
for name, (rmse, med, mx) in results.items():
    print(f"{name:28s}  RMSE={rmse*1000:.2f} mm  median={med*1000:.2f} mm  max={mx*1000:.2f} mm")
    if rmse < best_rmse:
        best_rmse = rmse; best_name = name

print(f"\n[SELECT] using combo: {best_name}")

# 선택 반영
H_base_cam_used, H_grip_board_used = combos[best_name]

# 이후 일관된 표준 출력/저장용으로 값 업데이트
FINAL_H_base_cam  = H_base_cam_used
FINAL_H_cam_base  = inv_h(FINAL_H_base_cam)
FINAL_H_grip_board= H_grip_board_used

# 최종 일관성 재출력
rmse, med, mx = compute_rmse(FINAL_H_base_cam, FINAL_H_grip_board, sel, objp_h)
print("\n=== CONSISTENCY (board points in base) ===")
print(f"RMSE = {rmse*1000:.2f} mm,  median = {med*1000:.2f} mm,  max = {mx*1000:.2f} mm")

# 저장도 FINAL_* 기준으로
out = {
    "H_base_cam": FINAL_H_base_cam.tolist(),
    "H_cam_base": FINAL_H_cam_base.tolist(),
    "H_grip_board": FINAL_H_grip_board.tolist(),
    "stats": {
        "rmse_m": rmse, "median_m": med, "max_m": mx,
        "n_samples": len(sel), "n_total": len(records),
        "selected_combo": best_name
    },
    "pattern": PATTERN,
    "notes": "Use H_base_cam to map camera 3D -> base 3D: p_base = H_base_cam @ p_cam_h"
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
np.savez(OUT_NPY, H_base_cam=FINAL_H_base_cam, H_cam_base=FINAL_H_cam_base, H_grip_board=FINAL_H_grip_board)
print(f"Saved: {OUT_JSON}, {OUT_NPY}")


print("\n=== SOLUTION ===")
np.set_printoptions(suppress=True, precision=6)
print("H_base_cam =\n", H_base_cam)
print("H_grip_board =\n", H_grip_board)

# ---------- 4) 검증(보드 점을 두 경로로 base로 옮겨 비교) ----------
objp = make_objp(PATTERN["type"], PATTERN["cols"], PATTERN["rows"], PATTERN["spacing"])
objp_h = np.hstack([objp, np.ones((objp.shape[0],1))])   # Nx4

errs_all = []
for s in sel:
    Hcb = s["H_cam_board"]
    Hbg = s["H_base_grip"]
    # 경로1: cam->base
    P1 = (H_base_cam @ Hcb @ objp_h.T).T[:, :3]
    # 경로2: grip->base
    P2 = (Hbg @ H_grip_board @ objp_h.T).T[:, :3]
    e  = np.linalg.norm(P1 - P2, axis=1)  # per-point [m]
    errs_all.append(e)

errs_all = np.concatenate(errs_all)
rmse = float(np.sqrt(np.mean(errs_all**2)))
med  = float(np.median(errs_all))
mx   = float(np.max(errs_all))

print("\n=== CONSISTENCY (board points in base) ===")
print(f"RMSE = {rmse*1000:.2f} mm,  median = {med*1000:.2f} mm,  max = {mx*1000:.2f} mm")

# ---------- 5) 저장 ----------
out = {
    "H_base_cam": H_base_cam.tolist(),
    "H_cam_base": H_cam_base.tolist(),
    "H_grip_board": H_grip_board.tolist(),
    "stats": {
        "rmse_m": rmse, "median_m": med, "max_m": mx,
        "n_samples": len(sel), "n_total": len(records)
    },
    "pattern": PATTERN,
    "notes": "Use H_base_cam to map camera 3D -> base 3D: p_base = H_base_cam @ p_cam_h"
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
np.savez(OUT_NPY, H_base_cam=H_base_cam, H_cam_base=H_cam_base, H_grip_board=H_grip_board)
print(f"Saved: {OUT_JSON}, {OUT_NPY}")
