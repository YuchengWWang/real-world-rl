import tyro
from xrobotoolkit_teleop.hardware.xr_publisher import (
    XRPublisher
)


def main(
    control_rate_hz: int = 30,

):
    """
    Main function to run the Piper teleoperation.
    """
    controller = XRPublisher(
        control_rate_hz=control_rate_hz,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
