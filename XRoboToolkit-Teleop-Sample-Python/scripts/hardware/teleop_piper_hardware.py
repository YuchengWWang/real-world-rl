import tyro
from xrobotoolkit_teleop.hardware.piper_teleop_controller import (
    DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
    DEFAULT_DUAL_PIPER_URDF_PATH,
    PiperTeleopController
)


def main(
    robot_urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
    scale_factor: float = 1.5,
    enable_camera: bool = False,
    enable_log_data: bool = False,
    visualize_placo: bool = False,
    control_rate_hz: int = 30,
    log_dir: str = "logs/piper",
    enable_camera_compression: bool = True,
    camera_jpg_quality: int = 85,
    _is_xr_exe = True
):
    """
    Main function to run the Piper teleoperation.
    """
    controller = PiperTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
        scale_factor=scale_factor,
        enable_camera=enable_camera,
        enable_log_data=enable_log_data,
        can_ports={"right_arm": "can_right",
                   "left_arm": "can_left"},
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        log_dir=log_dir,
        enable_camera_compression=enable_camera_compression,
        camera_jpg_quality=camera_jpg_quality,
        _is_xr_exe = _is_xr_exe
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
