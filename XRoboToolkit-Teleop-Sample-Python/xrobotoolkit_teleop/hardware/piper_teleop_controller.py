import os
import time
from typing import Dict
import threading
import requests

import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import (
    HardwareTeleopController,
)

from xrobotoolkit_teleop.hardware.interface.piper import PiperInterface
from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_PIPER_URDF_PATH = os.path.join(ASSET_PATH, "piper/piper.urdf")
DEFAULT_DUAL_PIPER_URDF_PATH = os.path.join(ASSET_PATH, "piper/dual_piper.urdf")
DEFAULT_SCALE_FACTOR = 1.0
CONTROLLER_DEADZONE = 0.1

# Default camera configuration
# DEFAULT_RIGHT_WRIST_CAM_SERIAL = "218622272014"
# DEFAULT_LEFT_WRIST_CAM_SERIAL = "218622272499"
# DEFAULT_BASE_CAM_SERIAL = "215222077461"

CAM_SERIAL_DICT = {
    # "left_wrist": DEFAULT_LEFT_WRIST_CAM_SERIAL,
    # "right_wrist": DEFAULT_RIGHT_WRIST_CAM_SERIAL,
    # "base": DEFAULT_BASE_CAM_SERIAL,
}

DEFAULT_CAN_PORTS = {
    "left_arm": "can_left",
    "right_arm": "can_right",
}


DEFAULT_PIPER_MANIPULATOR_CONFIG = {
    "right_arm": {  # Using "right_arm" for consistency with base controller
        "link_name": "eeflink",  # URDF link name for the end-effector
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["joint7"],
            "open_pos": [0.1],
            "close_pos": [0.0],
        },
    },
}

DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG = {
    "right_arm": {  # Using "right_arm" for consistency with base controller
        "link_name": "right_eeflink",  # URDF link name for the end-effector
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["right_joint7"],
            "open_pos": [0.1],
            "close_pos": [0.0],
        },
    },
    "left_arm": {  # Using "left_arm" for consistency with base controller
        "link_name": "left_eeflink",  # URDF link name for the end-effector
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": ["left_joint7"],
            "open_pos": [0.1],
            "close_pos": [0.0],
        },
    },
}

url_left = "http://100.72.16.108:5001/"
url_right = "http://100.72.16.108:5000/"


class PiperTeleopController(HardwareTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
        manipulator_config: dict = DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
        can_ports: Dict[str, str] = DEFAULT_CAN_PORTS,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = True,
        log_dir: str = "logs/piper",
        log_freq: float = 50,
        enable_camera: bool = True,
        camera_serial_dict: Dict[str, str] = CAM_SERIAL_DICT,
        camera_width: int = 424,
        camera_height: int = 240,
        camera_fps: int = 60,
        enable_camera_depth: bool = False,
        enable_camera_compression: bool = True,
        camera_jpg_quality: int = 85,
        _is_xr_exe: bool = False
    ):
        self.can_ports = can_ports
        self.camera_serial_dict = camera_serial_dict
        self.camera_serial_to_name = {serial: name for name, serial in camera_serial_dict.items()}
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.enable_camera_depth = enable_camera_depth
        self.enable_camera_compression = enable_camera_compression
        self.camera_jpg_quality = camera_jpg_quality
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
            _is_xr_pub=False,
            _is_xr_exe=_is_xr_exe
        )

    def _placo_setup(self):
        super()._placo_setup() # 根据urdf文件在pinocchio建模机械臂
        self.placo_arm_joint_slice: Dict[str, slice] = {}
        for arm_name, config in self.manipulator_config.items(): # 重命名左右臂（加上左右区分），取出关节名称索引
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("eeflink", "")
            arm_joint_names = [f"{arm_prefix}joint{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _robot_setup(self):
        """Initializes the Piper hardware interfaces for both arms."""
        self.arm_controllers: Dict[str, PiperInterface] = {}
        for arm_name, can_port in self.can_ports.items():
            print(f"Setting up Piper {arm_name} on CAN port: {can_port}")
            if arm_name == "right_arm":
                arm = PiperInterface(can_port=can_port, url=url_right, dt=self.dt)
            elif arm_name == "left_arm":
                arm = PiperInterface(can_port=can_port, url=url_left, dt=self.dt)
            self.arm_controllers[arm_name] = arm

        print("Going to home position...")
        for arm in self.arm_controllers.values():
            arm.go_home()

        time.sleep(1)  # Wait for the arms to reach home
        print("Arms are at home.")

    def _initialize_camera(self):
        if self.enable_camera:
            print("Initializing camera...")
            try:
                self.camera_interface = RealSenseCameraInterface(
                    width=self.camera_width,
                    height=self.camera_height,
                    fps=self.camera_fps,
                    serial_numbers=list(self.camera_serial_dict.values()),
                    enable_depth=self.enable_camera_depth,
                    enable_compression=self.enable_camera_compression,
                    jpg_quality=self.camera_jpg_quality,
                )
                self.camera_interface.start()
                print("Camera initialized successfully.")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                self.camera_interface = None

    def _update_robot_state(self):
        """Reads current joint states from the arms and updates Placo."""
        for arm_name, controller in self.arm_controllers.items():
            q_slice = self.placo_arm_joint_slice[arm_name]
            self.placo_robot.state.q[q_slice] = controller.get_joint_positions()[:6] # 这个controller是硬件接口，读编码器返回值，赋给placo_robot里的pinocchio模型作fk

    def _control_thread(self, stop_event: threading.Event):
        """Dedicated thread for sending commands to hardware."""
        while not stop_event.is_set():
            start_time = time.time()
            for arm_name, controller in self.arm_controllers.items():
                self._send_command()
                # exe不需要知道是否activate, 只需要持续发送电机指令，takeover信号由pub提供给server，gym去时刻读取并确定delta来源
                # if self.active.get(arm_name, False):
                #     self._send_takeover()
                # else:
                #     self._send_automatic()

            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._shutdown_robot()
        print("Control loop has stopped.")

    
    
    def _send_command(self):
        """Sends the solved joint targets to the hardware controllers."""
        for arm_name, controller in self.arm_controllers.items():
            if self.active.get(arm_name, False):
                q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()
                controller.set_joint_positions(q_des)

                if "gripper_config" in self.manipulator_config[arm_name]:
                    gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                    joint_name = gripper_config["joint_names"][0]
                    gripper_target = self.gripper_pos_target[arm_name][joint_name]
                    controller.set_catch_pos(gripper_target)

    def _get_robot_state_for_logging(self) -> Dict:
        """Returns a dictionary of robot-specific data for logging."""
        return {
            "qpos": {arm: c.get_joint_positions() for arm, c in self.arm_controllers.items()},
            "qvel": {arm: c.get_joint_velocities() for arm, c in self.arm_controllers.items()},
            "qpos_des": {
                arm: self.placo_robot.state.q[self.placo_arm_joint_slice[arm]].copy()
                for arm in self.arm_controllers
            },
            "gripper_target": {
                arm: (
                    self.gripper_pos_target[arm].copy()
                    if "gripper_config" in self.manipulator_config[arm]
                    else None
                )
                for arm in self.arm_controllers
            },
        }

    def _get_camera_frame_for_logging(self) -> Dict:
        """Returns a dictionary of camera frames for logging with camera names as keys."""
        if not self.camera_interface:
            return {}

        # Use compressed frames for logging to reduce file size
        if self.camera_interface.enable_compression:
            frames_by_serial = self.camera_interface.get_compressed_frames()
        else:
            # Fallback to regular frames for compatibility
            frames_by_serial = self.camera_interface.get_frames()

        if not frames_by_serial:
            return {}

        # Convert from serial number keys to camera name keys
        frames_by_name = {}
        for serial, frames in frames_by_serial.items():
            camera_name = self.camera_serial_to_name.get(
                serial, serial
            )  # Fallback to serial if name not found
            frames_by_name[camera_name] = frames

        return frames_by_name

    def _shutdown_robot(self):
        """Performs graceful shutdown of the robot hardware."""
        for _, controller in self.arm_controllers.items():
            controller.stop_control()
        time.sleep(1)
