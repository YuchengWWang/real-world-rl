import time
import numpy as np
from typing import Tuple, List
import math
from scipy.spatial.transform import Rotation as R
from piper_sdk.piper_sdk import C_PiperInterface_V2

class Affine3d:
    def __init__(self, O_T_EE):
        self.T = np.asarray(O_T_EE, dtype=float).reshape(4, 4, order='F')
    @property
    def translation(self):
        return self.T[:3, 3]
    @property
    def linear(self):
        return self.T[:3, :3]
    def as_quat_xyzw(self):
        return R.from_matrix(self.linear).as_quat()

COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
        "joint1_nullspace_stiffness": 20,  # 关节1的零空间刚度
        "nullspace_stiffness_": 20
    }

_CAN_TO_IDX = {
    0x251: 0,  # Joint 1
    0x252: 1,  # Joint 2
    0x253: 2,  # Joint 3
    0x254: 3,  # Joint 4
    0x255: 4,  # Joint 5
    0x256: 5,  # Joint 6
}

cartesian_stiffness_vector = np.diag([COMPLIANCE_PARAM["translational_stiffness"]] * 3 + [COMPLIANCE_PARAM["rotational_stiffness"]] * 3)
cartesian_damping_vector = np.diag([COMPLIANCE_PARAM["translational_damping"]] * 3 + [COMPLIANCE_PARAM["rotational_damping"]] * 3)
Ki_ = np.diag([COMPLIANCE_PARAM["translational_Ki"]] * 3 + [COMPLIANCE_PARAM["rotational_Ki"]] * 3) 

deta_position_d_target = [0.0, 0.0, 0.0]
# delta_orientation_d_target = [1.0, 0.0, 0.0, 0.0]

class DynamicModel:
    @staticmethod
    def get_full_state(piper):
        """
        获取机械臂的全状态，包括关节位置、速度、零雅可比矩阵、末端执行器位姿、科氏力、重力、关节力矩和雅可比矩阵的伪逆。
        """

        def get_joint_states(piper) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            读取关节位置/速度/力矩.
            
            Returns:
                q   : (6,) 关节位置，单位 rad
                qd  : (6,) 关节角速度，单位 rad/s
                tau : (6,) 关节力矩，单位 N·m
            """
            # ---------- 1) 位置：来自 ArmMsgFeedBackJointStates（单位 0.001 度）----------
            js = piper.GetArmJointMsgs()  # -> ArmMsgFeedBackJointStates
            # 0.001 deg -> deg -> rad
            q_deg = np.array([
                js.joint_1, js.joint_2, js.joint_3,
                js.joint_4, js.joint_5, js.joint_6
            ], dtype=float) * 0.001
            q = np.deg2rad(q_deg)  # 转成 rad

            # ---------- 2) 高速反馈：速度/电流/力矩 ----------
            hs_msgs: List = piper.GetArmHighSpdInfoMsgs()  # -> list[ArmMsgFeedbackHighSpd]
            # 先按 can_id 排序，确保顺序为关节1..6
            hs_msgs = sorted(hs_msgs, key=lambda m: _CAN_TO_IDX.get(m.can_id, 999))

            # 角速度：0.001 rad/s -> rad/s
            qd = np.array([m.motor_speed for m in hs_msgs], dtype=float) * 0.001

            # 力矩：使用类自带的 cal_effort() 换算，然后 0.001 N·m -> N·m
            # （cal_effort 会根据 can_id 选不同系数）
            tau_001 = np.array([m.cal_effort() for m in hs_msgs], dtype=float)
            tau = tau_001 * 0.001  # 转成 N·m

            return q, qd, tau



        curr_full_state = {
            "q": piper.GetArmJointPositionMsgs(),  # 关节位置
            "dq": piper.GetArmJointVelocityMsgs(),  # 关节速度
            "zero_jacobian": piper.GetArmZeroJacobianMsgs(),  # ee to base雅可比矩阵
            "O_T_EE": piper.GetArmEndPoseMsgs(),  # 末端执行器位姿
            "coriolis": piper.GetArmCoriolisMsgs(),  # 科氏力
            "gravity": piper.GetArmGravityMsgs(),  # 重力
            "tau_J_d": piper.GetArmJointTorqueMsgs(),  # 上一个周期的关节力矩
        }
        return curr_full_state




if __name__ == "__main__":
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    model = DynamicModel(piper)

    curr_full_state = model.get_full_state(piper)

    curr_end_effector_pose = piper.GetArmEndPoseMsgs() # [6]
    # position_d_target = curr_end_effector_pose[:3] + deta_position_d_target
    # orientation_d_target = curr_end_effector_pose[3:6]


    # initialize and start
    initial_state = curr_full_state
    q_initial = initial_state["q"] # [7,1]
    initial_transform = Affine3d(initial_state["O_T_EE"]) # [4,4]这里需要一个将4*4的变换矩阵得到3维空间平移translation和四元数linear的函数
    
    position_d_ = initial_transform.translation # [3,1]
    orientation_d_ = initial_transform.as_quat_xyzw() # [4,1]
    # position_d_target_ = initial_transform.translation # [3,1]
    # orientation_d_target_ = initial_transform.linear # [4,1]

    q_d_nullspace_ = q_initial # [7,1]

    def update(model, piper):
        curr_full_state = model.get_full_state(piper)
        jacobian = curr_full_state["zero_jacobian"]
        coriolis = curr_full_state["coriolis"]
        gravity = curr_full_state["gravity"]
        q = curr_full_state["q"]
        dq = curr_full_state["dq"]
        tau_J_d_last = curr_full_state["tau_J_d"]
        transform = Affine3d(curr_full_state["O_T_EE"])
        position = transform.translation
        orientation = transform.as_quat_xyzw()

        error_ = np.array([0.0] * 6)  # [6,1]
        error_i = np.array([0.0] * 6)  # [6,1]
        error_[:3] = position - position_d_

        if np.dot(orientation_d_, orientation) < 0.0:
            orientation = -orientation
        
        R_cur = R.from_quat(orientation)
        R_des = R.from_quat(orientation_d_)
        R_err = R_cur.inv() * R_des
        error_[3:] = R_err.as_quat()[:3]

        boundary = np.array([[COMPLIANCE_PARAM["translational_clip_neg_x"], COMPLIANCE_PARAM["translational_clip_neg_y"], COMPLIANCE_PARAM["translational_clip_neg_z"],\
                             COMPLIANCE_PARAM["rotational_clip_neg_x"], COMPLIANCE_PARAM["rotational_clip_neg_y"], COMPLIANCE_PARAM["rotational_clip_neg_z"]],
                             [COMPLIANCE_PARAM["translational_clip_x"], COMPLIANCE_PARAM["translational_clip_y"], COMPLIANCE_PARAM["translational_clip_z"],\
                             COMPLIANCE_PARAM["rotational_clip_x"], COMPLIANCE_PARAM["rotational_clip_y"], COMPLIANCE_PARAM["rotational_clip_z"]]])
        error_ = np.clip(error_, boundary[0], boundary[1])

        error_i[:3] = np.clip(error_i[:3] + error_[:3], -0.1, 0.1)
        error_i[3:] = np.clip(error_i[3:] + error_[3:], -0.3, 0.3)

        tau_task = np.array([0.0] * 7)  # [7,1]
        tau_nullspace = np.array([0.0] * 7)  # [7,1]

        jacobian_transpose_pinv = np.linalg.pinv(jacobian.T)  # [7,6]

        tau_task = jacobian.T @ (-cartesian_stiffness_vector @ error_ - cartesian_damping_vector @ (jacobian @ dq) - Ki_ @ error_i)

        qe = q_d_nullspace_ - q
        qe[1] = qe[1] * COMPLIANCE_PARAM["joint1_nullspace_stiffness"]
        dqe = dq
        dqe[1] = dqe[1] * 2.0 * np.sqrt(COMPLIANCE_PARAM["joint1_nullspace_stiffness"])
        tau_nullspace = (np.eye(7) - jacobian.T @ jacobian_transpose_pinv) @ (COMPLIANCE_PARAM["nullspace_stiffness_"] * qe - 2.0 * np.sqrt(COMPLIANCE_PARAM["nullspace_damping_"]) * dqe)

        tau_d = tau_task + tau_nullspace + coriolis + gravity  # [7,1]
        tau_rate = tau_d - tau_J_d_last
        tau_rate = np.clip(tau_rate, -1.0, 1.0)
        tau_d = tau_J_d_last + tau_rate








    while True:
        
        print(piper.GetArmHighSpdInfoMsgs())
        time.sleep(0.1)