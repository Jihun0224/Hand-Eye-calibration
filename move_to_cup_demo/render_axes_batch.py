#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, json, math, argparse
import numpy as np, cv2, pandas as pd

# ---------- math utils ----------
def inv_h(H):
    R = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4); Hi[:3,:3]=R.T; Hi[:3,3]=-R.T@t
    return Hi

def to_rvec_tvec_from_H_camX(H_cam_X):
    Rm = H_cam_X[:3,:3].astype(np.float64)
    tv = H_cam_X[:3, 3].reshape(3,1).astype(np.float64)
    rvec,_ = cv2.Rodrigues(Rm)
    return rvec, tv

def draw_axes(img, K, D, rvec, tvec, axis_len=0.1, thickness=2, label=None):
    K = np.asarray(K, np.float64); D = np.asarray(D, np.float64)
    origin = np.zeros((3,1), np.float64)
    axes = np.float64([[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]]).reshape(-1,1,3)
    pts = np.vstack([origin.reshape(1,1,3), axes])
    proj,_ = cv2.projectPoints(pts, rvec, tvec, K, D)
    p0 = tuple(np.int32(proj[0].ravel()))
    px = tuple(np.int32(proj[1].ravel()))
    py = tuple(np.int32(proj[2].ravel()))
    pz = tuple(np.int32(proj[3].ravel()))
    out = img
    cv2.line(out, p0, px, (0,0,255), thickness)
    cv2.line(out, p0, py, (0,255,0), thickness)
    cv2.line(out, p0, pz, (255,0,0), thickness)
    if label:
        cv2.putText(out, label, (p0[0]+6, p0[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return out

def make_objp_asym(cols, rows, spacing):
    objp = np.zeros((cols*rows,3), np.float64)
    k=0
    for j in range(rows):
        for i in range(cols):
            x=(2*i + (j%2))*spacing; y=j*spacing
            objp[k,0]=x; objp[k,1]=y; k+=1
    return objp

def rot_angle_deg(R):
    tr = np.trace(R); c = max(-1.0, min(1.0, (tr-1)/2))
    return math.degrees(math.acos(c))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Batch overlay axes and summarize errors")
    ap.add_argument("--records_dir", default="calib_records")
    ap.add_argument("--handeye", default="handeye_result.json")
    ap.add_argument("--K", default="K_color_1280x720.npy")
    ap.add_argument("--D", default="D_color_1280x720.npy")
    ap.add_argument("--out_dir", default="out_overlays")
    ap.add_argument("--axis_board_m", type=float, default=0.05)
    ap.add_argument("--axis_grip_m",  type=float, default=0.10)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--spacing", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    K = np.load(args.K); D = np.load(args.D)
    with open(args.handeye, "r") as f: he = json.load(f)
    H_base_cam = np.asarray(he["H_base_cam"], float)
    H_cam_base = inv_h(H_base_cam)
    H_grip_board = np.asarray(he["H_grip_board"], float)

    objp = make_objp_asym(args.cols, args.rows, args.spacing)
    objp_h = np.hstack([objp, np.ones((objp.shape[0],1))])

    rows_csv = []
    jsons = sorted(glob.glob(os.path.join(args.records_dir, "*.json")))
    legend = np.zeros((90,260,3), np.uint8); legend[:] = (0,0,0)
    cv2.putText(legend, "X:red  Y:green  Z:blue", (18,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(legend, "board/grip axes", (18,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

    for jp in jsons:
        with open(jp,"r") as f: rec = json.load(f)
        img_name = rec.get("image"); 
        if not img_name: continue
        ip = img_name if os.path.isabs(img_name) else os.path.join(args.records_dir, img_name)
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None: 
            print("[skip] cannot read", ip); 
            continue

        # board pose in cam
        bd = rec["board_detection"]; 
        rvec = np.asarray(bd["rvec"], float).reshape(3,1)
        tvec = np.asarray(bd["tvec"], float).reshape(3,1)
        Rcb,_ = cv2.Rodrigues(rvec)
        H_cam_board = np.eye(4); H_cam_board[:3,:3]=Rcb; H_cam_board[:3,3]=tvec.reshape(3)

        # grip pose in base
        H_base_grip = np.asarray(rec["gripper_pose_base"]["H"], float)

        # overlay
        out = img.copy()
        out = draw_axes(out, K, D, rvec, tvec, axis_len=args.axis_board_m, thickness=3, label="board")
        H_cam_grip = H_cam_base @ H_base_grip
        r_g, t_g = to_rvec_tvec_from_H_camX(H_cam_grip)
        out = draw_axes(out, K, D, r_g, t_g, axis_len=args.axis_grip_m, thickness=2, label="grip")
        out[10:10+legend.shape[0], 10:10+legend.shape[1]] = cv2.addWeighted(
            out[10:10+legend.shape[0], 10:10+legend.shape[1]], 1.0, legend, 0.9, 0)

        # consistency error per sample
        H1 = H_base_cam @ H_cam_board
        H2 = H_base_grip @ H_grip_board
        P1 = (H1 @ objp_h.T).T[:, :3]
        P2 = (H2 @ objp_h.T).T[:, :3]
        diff = P1 - P2
        err_mm = np.linalg.norm(diff, axis=1) * 1000.0
        mean_mm = float(np.mean(err_mm)); med_mm=float(np.median(err_mm)); max_mm=float(np.max(err_mm))

        # rotation delta (deg)
        R1 = H1[:3,:3]; R2 = H2[:3,:3]
        Rdelta = R1 @ R2.T
        ang_deg = rot_angle_deg(Rdelta)

        # save
        base = os.path.splitext(os.path.basename(jp))[0]
        op = os.path.join(args.out_dir, f"{base}_axes.png")
        cv2.imwrite(op, out)

        rows_csv.append({
            "json": os.path.basename(jp),
            "image": os.path.basename(ip),
            "mean_err_mm": round(mean_mm,3),
            "median_err_mm": round(med_mm,3),
            "max_err_mm": round(max_mm,3),
            "rot_delta_deg": round(ang_deg,3),
            "overlay": op
        })
        print(f"[ok] {base}: mean {mean_mm:.2f} mm, rotΔ {ang_deg:.2f}° -> {op}")

    df = pd.DataFrame(rows_csv)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print("Saved summary:", csv_path)

if __name__ == "__main__":
    main()
