#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import cv2

# --------- math utils ----------
def inv_h(H):
    R = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4, dtype=float)
    Hi[:3,:3] = R.T
    Hi[:3, 3] = -R.T @ t
    return Hi

def to_rvec_tvec_from_H_camX(H_cam_X):
    R = H_cam_X[:3,:3].astype(np.float64)
    t = H_cam_X[:3, 3].reshape(3,1).astype(np.float64)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t

def draw_axes(img_bgr, K, D, rvec, tvec, axis_len=0.1, thickness=3, org_label=None):
    """
    이미지에 좌표축 투영 (X=red, Y=green, Z=blue).
    axis_len: meters (카메라 좌표계 기준)
    """
    K = np.asarray(K, dtype=np.float64); D = np.asarray(D, dtype=np.float64)
    origin = np.zeros((3,1), np.float64)
    axes = np.float64([
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ]).reshape(-1,1,3)  # (3,1,3)

    pts = np.vstack([origin.reshape(1,1,3), axes])  # (4,1,3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    p0 = tuple(np.int32(proj[0].ravel()))
    px = tuple(np.int32(proj[1].ravel()))
    py = tuple(np.int32(proj[2].ravel()))
    pz = tuple(np.int32(proj[3].ravel()))

    out = img_bgr
    cv2.line(out, p0, px, (0,0,255), thickness)   # X - red
    cv2.line(out, p0, py, (0,255,0), thickness)   # Y - green
    cv2.line(out, p0, pz, (255,0,0), thickness)   # Z - blue
    if org_label is not None:
        cv2.putText(out, org_label, (p0[0]+5, p0[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return out

# --------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Overlay axes for board/gripper/base on a sample image.")
    ap.add_argument("--sample_json", required=True, help="Path to a single calib JSON (e.g., calib_records/he_record_0003.json)")
    ap.add_argument("--image_root", default="calib_records", help="Directory that contains the PNG image referenced by JSON")
    ap.add_argument("--K", default="K_color_1280x720.npy", help="Path to intrinsic matrix .npy")
    ap.add_argument("--D", default="D_color_1280x720.npy", help="Path to distortion .npy")
    ap.add_argument("--handeye", default="handeye_result.json", help="Path to NLS result JSON (contains H_base_cam, H_grip_board)")
    ap.add_argument("--out", default="axes_overlay.png", help="Output image path")
    ap.add_argument("--len_board_m", type=float, default=0.05, help="Axis length for board frame (m)")
    ap.add_argument("--len_grip_m",  type=float, default=0.10, help="Axis length for gripper frame (m)")
    ap.add_argument("--len_base_m",  type=float, default=0.15, help="Axis length for base frame (m)")
    ap.add_argument("--draw_base", action="store_true", help="Try drawing base frame (may be off-image)")
    args = ap.parse_args()

    # --- load camera intrinsics ---
    K = np.load(args.K)
    D = np.load(args.D)

    # --- load hand-eye results ---
    with open(args.handeye, "r") as f:
        he = json.load(f)
    H_base_cam = np.asarray(he["H_base_cam"], dtype=float)
    H_cam_base = inv_h(H_base_cam)
    H_grip_board = np.asarray(he["H_grip_board"], dtype=float)

    # --- load sample json & image ---
    with open(args.sample_json, "r") as f:
        rec = json.load(f)

    img_name = rec.get("image")
    if img_name is None:
        raise SystemExit("image field missing in sample JSON")

    img_path = img_name
    if not os.path.isabs(img_path):
        img_path = os.path.join(args.image_root, img_name)
    if not os.path.exists(img_path):
        raise SystemExit(f"image not found: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"failed to read image: {img_path}")

    # --- board pose in camera (from solvePnP) ---
    bd = rec["board_detection"]
    rvec = np.asarray(bd["rvec"], dtype=float).reshape(3,1)
    tvec = np.asarray(bd["tvec"], dtype=float).reshape(3,1)
    # {}^{cam}H_{board} for possible use:
    Rcb, _ = cv2.Rodrigues(rvec)
    H_cam_board = np.eye(4, dtype=float)
    H_cam_board[:3,:3] = Rcb
    H_cam_board[:3, 3] = tvec.reshape(3)

    # --- gripper pose in base (FK) ---
    H_base_grip = np.asarray(rec["gripper_pose_base"]["H"], dtype=float)

    # --- frames to draw (in camera coordinates) ---
    # board: already in cam (rvec,tvec)
    out = draw_axes(img, K, D, rvec, tvec, axis_len=args.len_board_m, thickness=3, org_label="board")

    # grip: need {}^{cam}H_{grip} = {}^{cam}H_{base} * {}^{base}H_{grip}
    H_cam_grip = H_cam_base @ H_base_grip
    r_g, t_g = to_rvec_tvec_from_H_camX(H_cam_grip)
    out = draw_axes(out, K, D, r_g, t_g, axis_len=args.len_grip_m, thickness=2, org_label="grip")

    # (optional) base: {}^{cam}H_{base} = H_cam_base
    if args.draw_base:
        r_b, t_b = to_rvec_tvec_from_H_camX(H_cam_base)
        out = draw_axes(out, K, D, r_b, t_b, axis_len=args.len_base_m, thickness=2, org_label="base")

    # --- Legend hint ---
    cv2.rectangle(out, (10,10), (260,90), (0,0,0), -1)
    cv2.putText(out, "X:red  Y:green  Z:blue", (18,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, "board/grip[/base] axes", (18,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

    # --- save ---
    cv2.imwrite(args.out, out)
    print(f"[OK] saved: {args.out}")
    print(f"    image: {img_path}")
    print(f"    sample: {args.sample_json}")
    print(f"    handeye: {args.handeye}")
    print(f"    K/D: {args.K} / {args.D}")

if __name__ == "__main__":
    main()
