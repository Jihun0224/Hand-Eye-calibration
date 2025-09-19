#!/usr/bin/env python3
# save_torso5_cam_transform.py
import json, numpy as np, rby1_sdk as rb
from scipy.spatial.transform import Rotation as R

ADDRESS = "192.168.30.1:50051"
MODEL = "a"
POWER = ".*"
BASE_INDEX, LINK_TORSO_5_INDEX = 0, 1
HAND_EYE_JSON_IN  = "handeye_result.json"
HAND_EYE_JSON_OUT = "handeye_result_fixed.json"

def inv_h(H):
    Rm = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4); Hi[:3,:3] = Rm.T; Hi[:3,3] = -Rm.T @ t
    return Hi

def main():
    with open(HAND_EYE_JSON_IN, "r") as f:
        he = json.load(f)
    H_base_cam = np.asarray(he["H_base_cam"], float)

    robot = rb.create_robot(ADDRESS, MODEL)
    robot.connect()
    if not robot.is_power_on(POWER):
        assert robot.power_on(POWER), "power on failed"
    robot.servo_on(".*"); robot.reset_fault_control_manager(); robot.enable_control_manager()

    model = robot.model()
    dyn = robot.get_dynamics()
    st  = dyn.make_state(["base","link_torso_5"], model.robot_joint_names)
    st.set_q(robot.get_state().position)
    dyn.compute_forward_kinematics(st)

    # {}^{base}H_{link_torso_5}
    H_base_torso5 = np.array(dyn.compute_transformation(st, BASE_INDEX, LINK_TORSO_5_INDEX), dtype=float)
    # {}^{link_torso_5}H_{cam} = ({}^{base}H_{link_torso_5})^{-1}  {}^{base}H_{cam}
    H_linktorso5_cam = inv_h(H_base_torso5) @ H_base_cam

    he_out = he.copy()
    he_out["H_link_torso5_cam"] = H_linktorso5_cam.tolist()
    he_out["notes"] = he_out.get("notes","") + " | Added H_link_torso5_cam at current torso pose."

    with open(HAND_EYE_JSON_OUT, "w") as f:
        json.dump(he_out, f, indent=2)
    print("Saved:", HAND_EYE_JSON_OUT)

if __name__ == "__main__":
    main()
