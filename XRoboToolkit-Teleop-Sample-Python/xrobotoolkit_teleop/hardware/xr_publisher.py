from typing import Dict


import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import (
    HardwareTeleopController,
)

from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD


class XRPublisher(HardwareTeleopController):
    def __init__(
        self,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        visualize_placo: bool = False,
        control_rate_hz: int = 10,
    ):

        super().__init__(
            R_headset_world=R_headset_world,
            floating_base=False,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            robot_urdf_path="",
            manipulator_config={},
            scale_factor=1.0,
            enable_log_data=False,
            log_dir="",
            log_freq=0,
            enable_camera=True,
            camera_fps=0,
            _is_xr_pub=True
        )

    def _placo_setup(self):
        pass

    def _robot_setup(self):
        pass

    def _initialize_camera(self):
        pass

    def _update_robot_state(self):
        pass
    
    def _send_command(self):
        pass
    
    def _get_robot_state_for_logging(self) -> Dict:
        pass

    def _get_camera_frame_for_logging(self) -> Dict:
        pass

    def _shutdown_robot(self):
        pass

