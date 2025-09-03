import os
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import requests
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

from piper_sdk.piper_sdk import *
import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


k1 = 4.
k2 = 0.87

class PiperInterface:
    """
    Base class for a single robot arm.

    Args:
        config (Dict[str, sAny]): Configuration dictionary for the robot arm

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the robot arm
        num_joints (int): Number of joints in the arm
    """

    def __init__(
        self,
        can_port: str = "can_right",
        url: str = "http://100.72.16.108:5001/",
        dt: float = 0.01,
    ):
        self.dt = dt
        self.arm = C_PiperInterface_V2(can_port)
        self.arm.ConnectPort()
        self.url = url


    def go_home(self) -> bool:
        """
        Move the robot arm to a pre-defined home pose.

        Returns:
            bool: True if the action was successful, False otherwise
        """
        while( not self.arm.EnablePiper()):
            time.sleep(0.01)
        
        # self.arm.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.set_joint_positions(position=[0., 0., 0., 0., 0., 0.])
        self.arm.GripperCtrl(0,1000,0x00, 0xAE)
        time.sleep(1)
        self.arm.GripperCtrl(0,1000,0x02, 0)
        self.arm.GripperCtrl(0,1000,0x01, 0)
        time.sleep(1)
        print("testing gripper ......")
        self.arm.GripperCtrl(50*1000,1000,0x01, 0)
        time.sleep(1)
        self.arm.GripperCtrl(0,1000,0x01, 0)

        return True
    
    def return_home(self):
        """
        Move the robot arm to a pre-defined home pose.

        Returns:
            bool: True if the action was successful, False otherwise
        """
        while( not self.arm.EnablePiper()):
            time.sleep(0.01)
        
        # self.arm.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.set_joint_positions(position=[0., 0., 0., 0., 0., 0.])
        time.sleep(2)
        self.arm.GripperCtrl(0,1000,0x01, 0)

        # return True

    def protect_mode(self) -> bool:
        self.arm.MotionCtrl_1(0x01,0,0)
        return True

    def MITmove_joint(self, id, q_d, dq_d, Kp, Kd, tau_ff):
        self.arm.JointMitCtrl(id, q_d, dq_d, Kp, Kd, tau_ff)

    def set_joint_positions(
        self,
        position: Union[float, List[float], np.ndarray],
        **kwargs,  # Shape: (num_joints,)
    ) -> bool:
        
        g_comp = True
        tau_comp = np.array([0., 0., 0., 0., 0., 0.])

        if g_comp:
            ps = requests.post(self.url + "getcompensate").json()
            tau_comp = np.array(ps["tau_comp"])
            # print(tau_comp)

        kp = 3. * np.array([1.5/k1, 1.5/k1, 1.5/k1, 1./k2, 2./k2, 1./k2])
        kd = 0.2 * np.array([1./k1, 1./k1, 1./k1, 1./k2, 1./k2, 1./k2])
        # print(position)
        while( not self.arm.EnablePiper()):
            time.sleep(0.01)

        q_d = position
        # q_d[0] *= -1
        # q_d[1] *= -1
        q_d[4] -= 0.331
        # q_d[5] *= -1


        self.arm.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
        self.MITmove_joint(1, q_d[0], 0, kp[0], kd[0], tau_comp[0])
        self.MITmove_joint(2, q_d[1], 0, kp[1], kd[1], tau_comp[1])
        self.MITmove_joint(3, q_d[2], 0, kp[2], kd[2], tau_comp[2])
        self.MITmove_joint(4, q_d[3], 0, kp[3], kd[3], tau_comp[3])
        self.MITmove_joint(5, q_d[4], 0, kp[4], kd[4], tau_comp[4])
        self.MITmove_joint(6, q_d[5], 0, kp[5], kd[5], tau_comp[5])

    def set_catch_pos(self, pos: float):
        while( not self.arm.EnablePiper()):
            time.sleep(0.01)
        factor = 57295.7795 #1000*180/3.1415926
        joint_6 = round(pos*1000*1000)
        self.arm.GripperCtrl(abs(joint_6), 2000, 0x01, 0)

    def get_joint_positions(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        """
        Get the current joint position(s) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get positions for. Shape: (num_joints,) or single string. If None,
                            return positions for all joints.

        """

        hs_msgs = self.arm.GetArmHighSpdInfoMsgs()
        hs_msgs = hs_msgs.__dict__

        q = np.array(
            [hs_msgs["motor_1"].pos, hs_msgs["motor_2"].pos - 6, hs_msgs["motor_3"].pos + 7,
              hs_msgs["motor_4"].pos, hs_msgs["motor_5"].pos + 331, hs_msgs["motor_6"].pos
              ], dtype=float
        ) * 0.001

        return q

    def get_joint_velocities(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        """
        Get the current joint velocity(ies) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get velocities for. Shape: (num_joints,) or single string. If None,
                            return velocities for all joints.

        """
        hs_msgs = self.arm.GetArmHighSpdInfoMsgs()
        hs_msgs = hs_msgs.__dict__

        dq = np.arry(
            [hs_msgs["motor_1"].motor_speed, hs_msgs["motor_2"].motor_speed, hs_msgs["motor_3"].motor_speed,
              hs_msgs["motor_4"].motor_speed, hs_msgs["motor_5"].motor_speed, hs_msgs["motor_6"].motor_speed
              ], dtype=float
        ) * 0.001
        return dq

    def stop_control(self):
        self.return_home()
        time.sleep(5)
        # self.arm.MotionCtrl_1(0x01,0,0)

    def get_ee_pose_xyzrpy(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        xyzrpy = self.arm.GetArmEndPoseMsgs()

        return xyzrpy

    def __del__(self):
        print("PiperInterface is being deleted")
