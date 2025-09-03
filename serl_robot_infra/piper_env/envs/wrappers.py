import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
import requests
from scipy.spatial.transform import Rotation as R
from typing import List

url_left = "http://100.72.16.108:5001/"
url_right = "http://100.72.16.108:5000/"

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew  # 走完了或者terminated了是done, 成功了也是done
        info['succeed'] = bool(rew)
        if self.target_hz is not None:  # 控制频率的写法
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
class DualPicoIntervention(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action: np.ndarray) -> np.ndarray:
        intervened = False
        
        ps_left = requests.post(url_left + "get_pico_action").json()
        ps_right = requests.post(url_right + "get_pico_action").json()
        left_pose = np.array(ps_left["pose"])
        left_gripper_pos = np.array(ps_left["gripper_pos"])
        right_pose = np.array(ps_right["pose"])
        right_gripper_pos = np.array(ps_right["gripper_pos"])
        expert_a = np.concatenate((left_pose, left_gripper_pos, right_pose, right_gripper_pos), axis=0) # 14D action

        ps = requests.post(url_left + "get_takeover").json()
        is_taken = ps["is_taken"]
        if is_taken:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

class DualGripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_left_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_left_gripper_pos = obs["state"][0, 0]
        self.last_right_gripper_pos = obs["state"][0, 19]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        info["grasp_penalty"] = 0.0

        if (action[6] < -0.5 and self.last_left_gripper_pos > 0.085) or (
            action[6] > 0.5 and self.last_left_gripper_pos < 0.085
        ):
            info["grasp_penalty"] += self.penalty
            print("left grasp penalty")
            
        if (action[13] < -0.5 and self.last_right_gripper_pos > 0.085) or (
            action[13] > 0.5 and self.last_right_gripper_pos < 0.085
        ):
            info["grasp_penalty"] += self.penalty
            print("right grasp penalty")

        # 如果索引是这样取说明gym的flatten_space函数是按照字母顺序排的，也就是把gripper_pose放到tcp_pose前面了
        self.last_left_gripper_pos = observation["state"][0, 0]
        self.last_right_gripper_pos = observation["state"][0, 13]
        
        return observation, reward, terminated, truncated, info