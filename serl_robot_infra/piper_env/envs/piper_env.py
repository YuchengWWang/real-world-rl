"""Gym Interface for Piper"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from piper_env.camera.video_capture import VideoCapture
from piper_env.camera.rs_capture import RSCapture
from piper_env.utils.rotations import euler_2_quat, quat_2_euler

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


class PiperEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._RESET_POSE = config.RESET_POSE
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = 1.0

        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self._update_currpos()
        self.randomreset = config.RANDOM_RESET
        self.hz = hz

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []
        
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        # "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        # "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in config.REALSENSE_CAMERAS}
                ),
            }
        )

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()
        
        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized Piper")
        
    
    def _update_currpos(self):
        
        ps = requests.post(self.url + "getstate").json()

        self.currpose = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])
        self.currjacobian = np.reshape(np.array(ps["J"]))
        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])
        self.curr_gripper_pos = np.array(ps["gripper_pos"])
        # self.currtorque = np.array(ps["tau"])
        # self.currforce = np.array(ps["force"])

    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap
    
    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = pose[3:]

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign_x = np.sign(euler[0])
        euler[0] = sign_x * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        sign_z = np.sign(euler[2])
        euler[2] = sign_z * (
            np.clip(
                np.abs(euler[2]),
                self.rpy_bounding_box.low[2],
                self.rpy_bounding_box.high[2],
            )
        )

        euler[1] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1], self.rpy_bounding_box.high[1]
        )
        pose[3:] = euler

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        # ps = requests.post(self.url + "get_takeover").json()
        # is_taken = ps["is_taken"]

        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high) # -1, 1
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_euler(self.currpose[3:])
        ).as_euler('xyz')

        gripper_action = action[6] * self.action_scale[2] # 夹爪用绝对位置[-1, 1] -> [0, 0.05]
        self.action_dict_to_send = {"pose": [0., 0., 0., 0., .0, 0.],
                                    "gripper": [0.],}
        self._send_gripper_command(gripper_action) # 这里没有真正发送action，只是更新self.action_dict_to_send
        self._send_pos_command(self.clip_safety_box(self.nextpos)) # 这里才调用request请求发送action

        self.curr_path_length += 1
        # dt = time.time() - start_time
        # time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        #这里获取obs的时候的request是需要等待返回的，所以才是影响频率的主要因素，sleep是sleep掉整个流程多余的时间应该加在这后面才对
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        reward = 0   # 留在wrapper里算
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}


    def _send_gripper_command(self, pos, mode="binary"):
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.085) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                self.action_dict_to_send["gripper"] = [0.0]
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.085) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                self.action_dict_to_send["gripper"] = [0.1]
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else: 
                return
        
        elif mode == "continuous":
            data = {"pos": pos.tolist(),
                    "tau": tau.tolist()}
            requests.post(self.url + "move_gripper", json=data)

    def _send_pos_command(self, nextpos: np.ndarray, mode="joint_impedance"):
        """关节空间阻抗，发的是6维nextpos，通过/set_action发到piper server上，再由xr_exe去执行"""
        if mode == "joint_impedance":
            self.action_dict_to_send["pose"] = nextpos.tolist()
            # 发送请求如果不设置timeout会阻塞至一个RTT时间，影响频率，这里不需要等待返回可以直接把timeout设很小
            try:
                requests.post(
                    self.url + "set_action",
                    json=self.action_dict_to_send,
                    timeout=(0.01, 0.01)  #connect/read各10ms，整个命令最多阻塞～10ms
                )
            except requests.exceptions.RequestException:
                #忽略所有异常，发出去就行了
                pass     
    
    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpose,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            # "tcp_force": self.currforce,
            # "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
    
    def get_im(self) -> Dict[str, np.ndarray]:
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images
    
    def reset(self, mode="cartesian", hard_reset=False, **kwargs):
        if hard_reset:
            self.hard_reset()
        self.last_gripper_act = time.time()
        if mode == "cartesian":
            if self.save_video:
                self.save_video_recording()

            self.go_to_reset()
            self.curr_path_length = 0

            self._update_currpos()
            obs = self._get_obs()
            self.terminate = False
            return obs, {"succeed": False}
    
    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://100.16.72.108:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def hard_reset(self):
        reset_t = time.time()
        while time.time() - reset_t < 4:
            requests.post(self.url + "stop")
        while time.time() - reset_t < 5:
            requests.post(self.url + "disable")
        while time.time() - reset_t < 7:
            requests.post(self.url + "reset")
        while time.time() - reset_t < 10:
            requests.post(self.url + "enable")
        
        print("Hard reset finished")
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        # if goal.shape == (6,):
        #     goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpose, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()
    
    def go_to_reset(self):
        self._update_currpos()
        self._send_pos_command(self.currpose)
        reset_pose = self.resetpos.copy()
        self.interpolate_move(reset_pose, timeout=1) # 慢慢回零，这样阻抗的K和D可以只按小步走设计