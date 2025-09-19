#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import rby1_sdk as rb

# ---------- 경로 ----------
K_FILE = "K_color_1280x720.npy"
D_FILE = "D_color_1280x720.npy"
HAND_EYE_JSON = "handeye_result_fixed.json"  # save_torso5_cam_transform.py로 만든 파일 (H_link_torso5_cam 포함)
RS_PRESET = "rs_locked_preset.json"
YOLO_WEIGHTS = "yolov8m.pt"
OUT_DIR = "cup_eval_out"

# ---------- 로봇 인덱스 / 주소 ----------
ADDRESS     = "192.168.30.1:50051"
SIM_ADDRESS = "172.17.0.1:50051"
USE_SIM     = False  # 시뮬 테스트면 True
MODEL   = "a"
POWER   = ".*"
BASE_INDEX, LINK_TORSO_5_INDEX, EE_RIGHT_INDEX, EE_LEFT_INDEX = 0, 1, 2, 3

# ---------- 파라미터 ----------
TARGET_CLASS = "cup"
CONF_THRESH  = 0.25
RETRY_MAX    = 10
STACK_N      = 7          # 같은 포즈에서 n프레임 쌓아 z 중앙값
PATCH_INIT   = 3
PATCH_MAX    = 21
MOVE_TIME    = 5.0
STEP_MAX     = 0.20       # 다단계 이동: 한 스텝 최대 이동 (m)
Z_STEP_MAX   = 0.10       # 다단계 이동: z 최대 스텝 (m)

# ---------- 유틸 ----------
def inv_h(H):
    Rm = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4); Hi[:3,:3] = Rm.T; Hi[:3,3] = -Rm.T @ t
    return Hi

def load_KD():
    return np.load(K_FILE).astype(np.float64), np.load(D_FILE).astype(np.float64)

def load_handeye_fixed():
    with open(HAND_EYE_JSON, "r") as f:
        j = json.load(f)
    if "H_link_torso5_cam" not in j:
        raise RuntimeError("handeye_result_fixed.json에 'H_link_torso5_cam'이 필요합니다.")
    H_linktorso5_cam = np.asarray(j["H_link_torso5_cam"], float)
    return H_linktorso5_cam

def undistort_pixel_to_cam3d(u, v, z, K, D):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    norm = cv2.undistortPoints(pts, K, D, P=None)  # normalized
    x, y = float(norm[0,0,0]), float(norm[0,0,1])
    return np.array([x*z, y*z, z], dtype=float)

def median_depth_patch(depth_np, u, v, depth_scale, init_r=3, max_r=21, max_depth_m=5.0):
    h, w = depth_np.shape[:2]
    u0 = int(round(u)); v0 = int(round(v))
    for r in range(init_r, max_r+1, 2):
        x1 = max(0, u0-r); x2 = min(w-1, u0+r)
        y1 = max(0, v0-r); y2 = min(h-1, v0+r)
        patch = depth_np[y1:y2+1, x1:x2+1].astype(np.float32)
        valid = patch > 0
        if valid.sum()==0: continue
        z = float(np.median(patch[valid])) * depth_scale
        if 0.0 < z <= max_depth_m: return z
    return None

def get_depth_stable(pipe, align, cx, cy, depth_scale, stack_n=7):
    zs = []
    for _ in range(stack_n):
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        d = aligned.get_depth_frame()
        if not d: continue
        depth_np = np.asanyarray(d.get_data())
        z = median_depth_patch(depth_np, cx, cy, depth_scale, init_r=PATCH_INIT, max_r=PATCH_MAX)
        if z is not None: zs.append(z)
        time.sleep(0.003)
    if len(zs)==0: return None
    return float(np.median(np.array(zs)))

def center_from_xyxy(x1,y1,x2,y2):
    return int(round((x1+x2)/2.0)), int(round((y1+y2)/2.0))

def _distance(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.linalg.norm(a - b))

# ---------- 로봇 래퍼 (좌/우 공용) ----------
class RBY1:
    def __init__(self):
        addr = SIM_ADDRESS if USE_SIM else ADDRESS
        self.robot = rb.create_robot(addr, MODEL)
        self.robot.connect()
        if not self.robot.is_power_on(".*"):
            assert self.robot.power_on(".*"), "power on failed"
        self.robot.servo_on(".*")
        self.robot.reset_fault_control_manager()
        self.robot.enable_control_manager()
        self.model = self.robot.model()
        self.dyn = self.robot.get_dynamics()
        self.state = self.dyn.make_state(["base","link_torso_5","ee_right","ee_left"], self.model.robot_joint_names)

    def _ee_index_and_name(self, side:str):
        s = side.lower()
        if s in ("left","l"):
            return EE_LEFT_INDEX, "ee_left"
        elif s in ("right","r"):
            return EE_RIGHT_INDEX, "ee_right"
        raise ValueError("side must be 'left' or 'right'")

    def current_H_base(self, target_index):
        self.state.set_q(self.robot.get_state().position)
        self.dyn.compute_forward_kinematics(self.state)
        return np.array(self.dyn.compute_transformation(self.state, BASE_INDEX, target_index), dtype=float)

    def current_H_base_torso5(self):
        return self.current_H_base(LINK_TORSO_5_INDEX)

    def current_H_base_ee(self, side:str):
        idx, _ = self._ee_index_and_name(side)
        return self.current_H_base(idx)

    def current_pose_in_torso5(self, side:str):
        H_base_torso5 = self.current_H_base_torso5()
        H_base_ee     = self.current_H_base_ee(side)
        H_torso5_ee   = inv_h(H_base_torso5) @ H_base_ee
        p_torso5 = H_torso5_ee[:3,3].copy()
        R_torso5 = H_torso5_ee[:3,:3].copy()
        return p_torso5, R_torso5, H_torso5_ee

    def _send_pose_in_torso5(self, side:str, p_torso5_target, minimum_time):
        """현재 EE 회전 유지, 위치만 p_torso5_target으로 교체해 add_target 전송"""
        ee_idx, ee_name = self._ee_index_and_name(side)
        # 현재 변환들
        H_base_torso5 = self.current_H_base_torso5()
        H_base_ee     = self.current_H_base_ee(side)
        # base 기준 desired (회전 유지, 위치만 교체)
        H_des = H_base_ee.copy()
        p_base_target = (H_base_torso5 @ np.r_[p_torso5_target, 1.0])[:3]
        H_des[:3,3] = p_base_target
        # torso5 기준으로 변환
        T_torso5_ee_des = inv_h(H_base_torso5) @ H_des

        # 빌더 분기 명확화
        body = rb.BodyComponentBasedCommandBuilder()
        cart = (rb.CartesianCommandBuilder()
                .set_command_header(rb.CommandHeaderBuilder().set_control_hold_time(100))
                .set_minimum_time(minimum_time)
                .add_target("link_torso_5", ee_name, T_torso5_ee_des, 1.0, 1.0, 1.0))

        if ee_name == "ee_right":
            body = body.set_right_arm_command(cart)
        else:
            body = body.set_left_arm_command(cart)

        cmd = rb.RobotCommandBuilder().set_command(
            rb.ComponentBasedCommandBuilder().set_body_command(body)
        )
        fb = self.robot.send_command(cmd).get()  # 동기 대기
        ok = (fb.finish_code == rb.RobotCommandFeedback.FinishCode.Ok)
        info = {"finish_code": str(fb.finish_code), "T_torso5_ee_des": T_torso5_ee_des.tolist()}
        return ok, info

    def move_ee_to_torso5_point_multi(self, side:str, p_torso5_target, 
                                      minimum_time=MOVE_TIME, 
                                      step_max=STEP_MAX, z_step_max=Z_STEP_MAX):

        p_curr, R_curr, _ = self.current_pose_in_torso5(side)
        p_tgt = np.asarray(p_torso5_target, float).reshape(3,)
        delta = p_tgt - p_curr

        n_steps_pos = int(np.ceil(np.linalg.norm(delta) / step_max)) if step_max > 0 else 1
        n_steps_z   = int(np.ceil(abs(delta[2]) / z_step_max))        if z_step_max > 0 else 1
        n_steps = max(1, n_steps_pos, n_steps_z)

        waypoints = [p_curr + (i+1)/n_steps * delta for i in range(n_steps)]

        for i, p_wp in enumerate(waypoints, 1):
            ok, info = self._send_pose_in_torso5(side, p_wp, minimum_time)
            if not ok:
                print(f"[FAIL] step {i}/{n_steps} failed:", info)
                return False, {"failed_step": i, **info}
            time.sleep(0.05)
        return True, {"steps": n_steps, "final_target": p_tgt.tolist()}

# ---------- 한 번에: 좌/우 선택해 컵으로 이동 (방법 B만 사용) ----------
def move_to_cup(side="left"):
    os.makedirs(OUT_DIR, exist_ok=True)
    K, D = load_KD()
    H_linktorso5_cam = load_handeye_fixed()  # 방법 B 전용

    # YOLO
    yolo = YOLO(YOLO_WEIGHTS)

    # RealSense
    pipe = rs.pipeline()
    align = rs.align(rs.stream.color)
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    prof = pipe.start(cfg)

    # Preset 적용 (깊이 안정화)
    try:
        depth_sensor = prof.get_device().first_depth_sensor()
        depth_scale  = depth_sensor.get_depth_scale()
        if os.path.isfile(RS_PRESET):
            preset = json.load(open(RS_PRESET,"r"))
            for k,v in preset.items():
                opt = getattr(rs.option, k, None)
                if opt is not None and depth_sensor.supports(opt):
                    depth_sensor.set_option(opt, v)
    except Exception as e:
        print("[WARN] preset apply failed:", e)
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    print("[INFO] depth scale:", depth_scale)

    # 로봇
    rby = RBY1()

    # 탐지+깊이 안정화 with retry
    tried = 0; cup_box=None; cx=cy=None; z=None; color_bgr=None
    while tried < RETRY_MAX and z is None:
        tried += 1
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth: continue
        color_bgr = np.asanyarray(color.get_data())

        res = yolo(color_bgr[..., ::-1], verbose=False, conf=CONF_THRESH)[0]
        names = res.names
        if res.boxes is None or len(res.boxes)==0:
            print(f"[{tried}] no detections"); continue
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        clses = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        cups = [(float(c), b.tolist()) for b,cid,c in zip(boxes, clses, confs) if names[int(cid)].lower()==TARGET_CLASS]
        if not cups:
            print(f"[{tried}] no '{TARGET_CLASS}'"); continue
        cups.sort(key=lambda x:x[0], reverse=True)
        cup_box = cups[0][1]
        x1,y1,x2,y2 = cup_box
        cx, cy = center_from_xyxy(x1,y1,x2,y2)

        # 같은 포즈에서 n프레임 stack → z 중앙값
        z = get_depth_stable(pipe, align, cx, cy, depth_scale, stack_n=STACK_N)
        if z is None:
            print(f"[{tried}] depth missing; retrying...")
            continue

    try:
        pipe.stop()
    except: pass

    if z is None:
        print("[FAIL] depth not found after retries")
        return False

    # 픽셀→카메라3D
    p_cam = undistort_pixel_to_cam3d(cx, cy, z, K, D)
    # 카메라3D → torso5 (방법 B: 고정 H_link_torso5_cam)
    p_torso5 = (H_linktorso5_cam @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]

    # 현재 EE 포즈(토르소5 기준)와 델타
    p_curr_torso5, R_curr_torso5, _ = rby.current_pose_in_torso5(side)
    delta_torso5 = p_torso5 - p_curr_torso5

    # 정보 출력
    print("\n=== CUP TARGET (side: {}) ===".format(side))
    print(f"bbox: {cup_box}, center=({cx},{cy}), z={z:.4f} m")
    print("p_cam           :", np.round(p_cam, 6))
    print("p_torso5(target):", np.round(p_torso5, 6))
    print("EE_torso5(current) pos:", np.round(p_curr_torso5, 6))
    print("delta_torso5 (target - current):", np.round(delta_torso5, 6))

    # 실제 이동: 다단계 분할 이동(긴 거리/상하 큰 변화 대비)
    ok, info = rby.move_ee_to_torso5_point_multi(side, p_torso5, 
                                                 minimum_time=MOVE_TIME,
                                                 step_max=STEP_MAX, z_step_max=Z_STEP_MAX)
    print("move:", ok, info if not ok else "sent")
    return ok

# ---------- 실행 예시 ----------
if __name__ == "__main__":
    # "left" 또는 "right"
    move_to_cup(side="left")
