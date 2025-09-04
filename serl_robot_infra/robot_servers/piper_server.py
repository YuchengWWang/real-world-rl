from flask import Flask, request, jsonify
import numpy as np
import subprocess
from absl import app, flags
import time
from typing import Tuple, List
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import sys, os
root = os.path.abspath(os.path.join(__file__, "..", "..", "piper_sdk"))
sys.path.append(root)
from piper_sdk import *


FLAGS = flags.FLAGS
flags.DEFINE_string("flask_url", 
    "100.72.16.108",
    "URL for the flask server to run on."
)
flags.DEFINE_integer("port", 
    "5001",
    ""
)
flags.DEFINE_string("can_name", 
    "can_right",
    ""
)


urdf_path = "piper_sdk/urdf/piper_description_d405.urdf"
k1 = 4.
k2 = 0.87


class DynamicModel:
    def __init__(self, urdf_path, package_dirs="piper_sdk/urdf"):
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        self.model, self.data = self.robot.model, self.robot.data
        self.base_frame_id = self.robot.model.getFrameId("base_link")
        self.ee_frame_id = self.robot.model.getFrameId("eeflink") 
        # print(self.model)
    
    def get_J_C_g_T_e(self, q, dq, pose_desire):
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        
        pos = pose_desire[:3]   # [0.1, 0.0, 0.4]
        rpy = pose_desire[3:]
        rot = R.from_euler('xyz', rpy).as_matrix()
        oMdes = pin.SE3(rot, pos)

        oMe = self.data.oMf[self.ee_frame_id]
        pose_error = pin.log(oMdes * oMe.inverse()).vector
        
        
        T_base_ee = self.data.oMf[self.ee_frame_id].homogeneous
        # T_base_ee = self.data.oMf[self.ee_frame_id].homogeneous
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # pin.computeJointJacobians(self.model, self.data, q)
        # J = pin.getFrameJacobian(self.model, self.data, frame_id=self.ee_frame_id, reference_frame=pin.LOCAL_WORLD_ALIGNED)
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        coriolis = C @ dq

        # print(g)

        # pin.updateFramePlacements(self.model, self.data)
        # oMe = self.data.oMf[self.ee_frame_id]

        return J, g, coriolis, T_base_ee, pose_error

    def get_J_g(self, q, pose_desire):
        pos = pose_desire[:3]   # [0.1, 0.0, 0.4]
        rpy = pose_desire[3:]
        rot = R.from_euler('xyz', rpy).as_matrix()
        oMdes = pin.SE3(rot, pos)

    def get_J_C_g_p(self, q, dq):
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)        
        
        pos = self.data.oMf[self.ee_frame_id].translation
        rpy = R.from_matrix(self.data.oMf[self.ee_frame_id].rotation).as_euler('xyz')
        pose = np.concatenate([pos, rpy])
        # T_base_ee = self.data.oMf[self.ee_frame_id].homogeneous
        
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        coriolis = C @ dq

        return J, g, coriolis, pose
    
    def get_e(self, pose_desire):
        pos = pose_desire[:3]   
        rpy = pose_desire[3:]
        rot = R.from_euler('xyz', rpy).as_matrix()
        oMdes = pin.SE3(rot, pos)
        oMe = self.data.oMf[self.ee_frame_id]
        pose_error = pin.log(oMdes * oMe.inverse()).vector

        return pose_error


class piper_server:
    def __init__(self, piper_interface, reset_joint_target):
        self.reset_joint_target = reset_joint_target
        self.piper = piper_interface
        self.pino_model = DynamicModel(urdf_path)
        self.takeover = False

        self.curr_joint_q = np.zeros([6])
        self.curr_joint_dq = np.zeros([6])
        self.curr_joint_tau = np.zeros([6])
        self.curr_J = np.zeros([6,6])
        self.curr_g = np.zeros([6,1])
        self.curr_coriolis = np.zeros([6,6])
        self.curr_bTe = np.zeros([4,4])
        self.curr_pose_error = np.zeros([6,1])
        self.curr_pose = np.zeros([6,1])
        self.curr_vel = np.zeros([6,1])
        self.curr_gripper_pos = np.zeros([1])

        self.pico_action = {"pose": [0., 0., 0., 0., .0, 0.],
                            "gripper": [0.],
                            "activate_button": [0.]}
        self.curr_action = {"pose": [0., 0., 0., 0., .0, 0.],
                            "gripper": [0.],}

        K_cartesian = np.diag([1.2, 1., 1., 0.1, 0.1, 0.1])
        self.K_cartesian = K_cartesian * 90
        D_cartesian = np.diag([1.5, 1.3, 1.3, 0.1, 0.1, 0.1])
        self.D_cartesian = D_cartesian * 5
        self.kp_lift = 1 * np.array([0, 0, 0, 1, 0, 0])
        self.kd_lift = 0.2 * np.array([0.4, 0.8, 1., 1.1, 1.5, 1.0])
        self.k_joint = 1. *np.array([2., 5., 4, 1, 0.5, 0.5])

        self.tau_apply_last = np.array([0., 0., 0., 0., 0., 0.])

        self.set_curr_joint_q_qd_tau()
        self.set_curr_J_C_g_p()
        self.set_curr_eef_vel_gripper()

    def stop_control(self):
        self.piper.MotionCtrl_1(0x01,0,0)

    def reset(self):
        self.piper.MotionCtrl_1(0x02,0,0)#恢复
        self.piper.MotionCtrl_2(0, 0, 0, 0x00)#位置速度模式
        time.sleep(0.5)
        self.piper.MotionCtrl_1(0x02,0,0)#恢复
        self.piper.MotionCtrl_2(0, 0, 0, 0x00)#位置速度模式
    
    def enable(self):
        while( not self.piper.EnablePiper()):
            time.sleep(0.01)
        print("enable success!")
    
    def disable(self):
        while(self.piper.DisablePiper()):
            time.sleep(0.01)
        print("disable success!")

    def go_zero(self, position=[0., 0., 0., 0., 0., 0., 0.]):
        while( not self.piper.EnablePiper()):
            time.sleep(0.01)
        factor = 57295.7795 #1000*180/3.1415926
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        joint_6 = round(position[6]*1000*1000)
        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    def PDmove_joint(self, move_joint_target):
        while( not self.piper.EnablePiper()):
            time.sleep(0.01)
        factor = 57295.7795 #1000*180/3.1415926
        position = move_joint_target
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        joint_6 = round(position[6]*1000*1000)
        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    def set_curr_joint_q_qd_tau(self):

            # js = self.piper.GetArmJointMsgs()  # -> ArmMsgFeedBackJointStates
            # js = js.joint_state.__dict__
            # q_deg = np.array([
            #     js["joint_1"], js["joint_2"], js["joint_3"],
            #     js["joint_4"], js["joint_5"], js["joint_6"]
            # ], dtype=float) * 0.001
            # q = np.deg2rad(q_deg)
            # self.curr_joint_q = q
            
            hs_msgs = self.piper.GetArmHighSpdInfoMsgs()  # -> list[ArmMsgFeedbackHighSpd]
            hs_msgs = hs_msgs.__dict__
            
            dq = np.array([
                hs_msgs["motor_1"].motor_speed, hs_msgs["motor_2"].motor_speed, hs_msgs["motor_3"].motor_speed,
                hs_msgs["motor_4"].motor_speed, hs_msgs["motor_5"].motor_speed, hs_msgs["motor_6"].motor_speed
            ], dtype=float) * 0.001
            self.curr_joint_dq = dq

            tau = np.array([
                hs_msgs["motor_1"].effort, hs_msgs["motor_2"].effort, hs_msgs["motor_3"].effort,
                hs_msgs["motor_4"].effort, hs_msgs["motor_5"].effort, hs_msgs["motor_6"].effort
            ], dtype=float) * 0.001
            self.curr_joint_tau = tau

            q = np.array([
                hs_msgs["motor_1"].pos, hs_msgs["motor_2"].pos - 6, hs_msgs["motor_3"].pos + 7,
                hs_msgs["motor_4"].pos, hs_msgs["motor_5"].pos + 331, hs_msgs["motor_6"].pos
            ], dtype=float) * 0.001
            self.curr_joint_q = q

            # alpha = 1
            # self.curr_joint_q = alpha * q + (1 - alpha) * self.curr_joint_q
            # self.curr_joint_dq = alpha * dq + (1 - alpha) * self.curr_joint_dq

    def set_is_taken(self, is_taken):
        self.takeover = is_taken

    def set_curr_J_C_g_p(self):
        # print(self.curr_joint_q)
        self.curr_J, self.curr_g, self.curr_coriolis, self.curr_pose = self.pino_model.get_J_C_g_p(self.curr_joint_q, self.curr_joint_dq)

    def set_curr_eef_vel_gripper(self):
        self.curr_vel = self.curr_J @ self.curr_joint_dq
        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_msg = gripper_msg.__dict__
        self.curr_gripper_pos = np.clip(np.array(gripper_msg["gripper_state"].grippers_angle, dtype=float) / 789600, a_min=0, a_max=1)

    def set_curr_pose_error(self, pose_desire):
        self.curr_pose_error = self.pino_model.get_e(pose_desire)

    def set_mit(self):
        self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)

    def MITmove_joint(self, id, q_d, dq_d, Kp, Kd, tau_ff):
        self.piper.JointMitCtrl(id, q_d, dq_d, Kp, Kd, tau_ff)

    def MITmove_all_joint(self, pose_desire):
        self.set_curr_joint_q_qd_tau()
        self.set_curr_J_C_g_p()
        self.set_curr_eef_vel_gripper()
        self.set_curr_pose_error(pose_desire)
        
        tau_task = self.curr_J.T @ (self.K_cartesian @ self.curr_pose_error + self.D_cartesian @ (- self.curr_vel))
        tau_task = tau_task * self.k_joint

        tau_ori = tau_task + self.curr_g + self.curr_coriolis
        ori = tau_ori
        ori[:3] = ori[:3] / k1
        ori[3:] = ori[3:] / k2
        tau_new = ori

        tau_diff = tau_new - self.tau_apply_last
        tau_diff[:3] = np.clip(tau_diff[:3], -1. / k1, 1. / k1)
        tau_diff[3:] = np.clip(tau_diff[3:], -1. / k2, 1. / k2)
        tau_apply = tau_diff + self.tau_apply_last
        tau_apply[:3] = np.clip(tau_apply[:3], -8 / k1, 8 / k1)
        tau_apply[3:] = np.clip(tau_apply[3:], -5 / k2, 5 / k2)
        
    
        self.set_mit()
        self.MITmove_joint(1, 0, 0, self.kp_lift[0], self.kd_lift[0], tau_apply[0])
        self.MITmove_joint(2, 0, 0, self.kp_lift[1], self.kd_lift[1], tau_apply[1])
        self.MITmove_joint(3, 0, 0, self.kp_lift[2], self.kd_lift[2], tau_apply[2])
        self.MITmove_joint(4, 0, 0, self.kp_lift[3], self.kd_lift[3], tau_apply[3])
        self.MITmove_joint(5, 0, 0, self.kp_lift[4], self.kd_lift[4], tau_apply[4])
        self.MITmove_joint(6, 0, 0, self.kp_lift[5], self.kd_lift[5], tau_apply[5])

        self.tau_apply_last = tau_apply

    def get_compensate(self):
        self.set_curr_joint_q_qd_tau()
        self.set_curr_J_C_g_p()

        tau_ori = self.curr_g + self.curr_coriolis
        ori = tau_ori
        ori[:3] = ori[:3] / k1
        ori[3:] = ori[3:] / k2
        tau_new = ori

        tau_diff = tau_new - self.tau_apply_last
        tau_diff[:3] = np.clip(tau_diff[:3], -1. / k1, 1. / k1)
        tau_diff[3:] = np.clip(tau_diff[3:], -1. / k2, 1. / k2)
        tau_apply = tau_diff + self.tau_apply_last
        tau_apply[:3] = np.clip(tau_apply[:3], -8 / k1, 8 / k1)
        tau_apply[3:] = np.clip(tau_apply[3:], -5 / k2, 5 / k2)

        self.tau_apply_last = tau_apply

        return tau_apply

    def set_p_ref_init(self):
        self.set_curr_joint_q_qd_tau()
        self.set_curr_J_C_g_p()
        self.p_ref_init = self.curr_pose


def main(_):
    webapp = Flask(__name__)
    piper = C_PiperInterface(can_name=FLAGS.can_name,
                                judge_flag=False,
                                can_auto_init=True,
                                dh_is_offset=1,
                                start_sdk_joint_limit=False,
                                start_sdk_gripper_limit=False,
                                logger_level=LogLevel.WARNING,
                                log_to_file=False,
                                log_file_path=None)
    piper.ConnectPort()
    robot_server = piper_server(piper_interface=piper, reset_joint_target=[0., 0., 0., 0., 0., 0., 0.])


    
    @webapp.route("/getstate", methods=["GET", "POST"])
    def get_state():
        robot_server.set_curr_joint_q_qd_tau()
        robot_server.set_curr_J_C_g_p()
        robot_server.set_curr_eef_vel_gripper()
        
        return jsonify({"q": robot_server.curr_joint_q.tolist(), 
                        "dq": robot_server.curr_joint_dq.tolist(),
                        # "tau": robot_server.curr_joint_tau.tolist(),
                        "J": robot_server.curr_J.tolist(),
                        "pose": robot_server.curr_pose.tolist(),
                        "vel": robot_server.curr_vel.tolist(),
                        "gripper_pos": robot_server.curr_gripper_pos.tolist(),
                        })
    
    @webapp.route("/getcompensate", methods=["GET", "POST"])
    def get_compensate():
        tau_comp = robot_server.get_compensate()
        return jsonify({"tau_comp": tau_comp.tolist()})

    @webapp.route("/reset", methods=["GET", "POST"])
    def reset():
        robot_server.reset()
        return "reset arm"
    
    @webapp.route("/go_zero", methods=["GET", "POST"])
    def go_zero():
        robot_server.go_zero()
        return "go zero"
    
    @webapp.route("/enable", methods=["GET", "POST"])
    def enable():
        robot_server.enable()
        return "enable arm"
    
    @webapp.route("/disable", methods=["GET", "POST"])
    def disable():
        robot_server.disable()
        return "disable arm"
    
    @webapp.route("/PDmove", methods=["GET", "POST"])
    def PDmove():
        pos = np.array(request.json["arr"])
        robot_server.PDmove_joint(pos)
        return "PD moving to: "+str(pos)
    
    @webapp.route("/stop", methods=["GET", "POST"])
    def stop():
        robot_server.stop_control()
        return "stop!"
    
    @webapp.route("/setmit", methods=["GET", "POST"])
    def set_mit():
        robot_server.set_mit()
        return "set to MIT mode"
    
    @webapp.route("/MITmove", methods=["GET", "POST"])
    def MIT_move():
        pose_desire = np.array(request.json["pose_desire"])
        robot_server.MITmove_all_joint(pose_desire=pose_desire)

        return "mit moving to"+str(pose_desire)

    @webapp.route("/takeover", methods=["GET", "POST"])
    def set_takeover():
        robot_server.set_is_taken(True)
        return "takeover"
    
    @webapp.route("/automatic", methods=["GET", "POST"])
    def set_automatic():
        robot_server.set_is_taken(False)
        return "automatic"
    
    @webapp.route("/get_takeover", methods=["GET", "POST"])
    def get_takeover():
        return jsonify({"is_taken": robot_server.takeover})
    
    @webapp.route("/set_pico_action", methods=["GET", "POST"])
    def set_pico_action():
        robot_server.pico_action = request.json
        return "set pico action"

    @webapp.route("/get_pico_action", methods=["GET", "POST"])
    def get_pico_action():
        return jsonify(robot_server.pico_action)
    
    @webapp.route("/set_action", methods=["GET", "POST"])
    def set_action():
        robot_server.curr_action = request.json
        return "set action"
    
    @webapp.route("/get_action", methods=["GET", "POST"])
    def get_action():
        return jsonify(robot_server.curr_action)
    
    @webapp.route("/get_p_ref_init", methods=["GET", "POST"])
    def get_p_ref_init():
        robot_server.set_p_ref_init()
        return jsonify({"p_ref_init":robot_server.p_ref_init.tolist()}) # 6D pose list
    

    webapp.run(host=FLAGS.flask_url, port=FLAGS.port)


if __name__ == "__main__":
    app.debug = True
    app.run(main)
    # test_main()





