"""Gym Interface for Piper"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict

import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..","..","serl_launcher")))
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..","..","serl_robot_infra")))
# print(os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "serl_robot_infra")))


from piper_env.envs.wrappers import (
    DualPicoIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    DualGripperPenaltyWrapper
)
from piper_env.envs.piper_env import PiperEnv
from piper_env.envs.relative_env import DualRelativeFrame
from piper_env.envs.dual_piper_env import DualPiperEnv
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig

class EnvConfig:
    SERVER_URL = "http://100.72.16.108:5001/"
    REALSENSE_CAMERAS = {}
    IMAGE_CROP = {}
    RESET_POSE = np.array([0.25705429, -0.01145554, 0.35798343, 0, 0, 0])
    ABS_POSE_LIMIT_LOW = np.array([0., -0.03, 0.3, np.pi-0.01, -0.3, 0]) # 他这样设置基本上就只给绕y轴转20度
    ABS_POSE_LIMIT_HIGH = np.array([0.67, -0.01, 0.42, np.pi+0.01, 0.0, 0.01]) # 只能落在
    RANDOM_RESET = False
    ACTION_SCALE = (0.04, 0.2, 1) # 对应xyz、 rpy和gripper的scale
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100 #TBD
    # 现在piper_server的参数直接在server脚本里改就行了没必要在这里改
    # COMPLIANCE_PARAM = { 
    #     "translational_stiffness": 90, 
    #     "translational_damping": 6.5, 
    #     "rotational_stiffness": 9, 
    #     "rotational_damping": 0.5, 
    #     # "translational_Ki": 0,
    #     "translational_clip_x": 0.0075,
    #     "translational_clip_y": 0.0016,
    #     "translational_clip_z": 0.0055,
    #     "translational_clip_neg_x": 0.002,
    #     "translational_clip_neg_y": 0.0016,
    #     "translational_clip_neg_z": 0.005,
    #     "rotational_clip_x": 0.01,
    #     "rotational_clip_y": 0.025,
    #     "rotational_clip_z": 0.005,
    #     "rotational_clip_neg_x": 0.01,
    #     "rotational_clip_neg_y": 0.025,
    #     "rotational_clip_neg_z": 0.005,
    #     # "rotational_Ki": 0,
    # }
    # LOAD_PARAM: Dict[str, float] = {
    #     "mass": 0.0,
    #     "F_x_center_load": [0.0, 0.0, 0.0],
    #     "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # }

class LeftEnvConfig(EnvConfig):
    SERVER_URL = "http://100.72.16.108:5000/"
    REALSENSE_CAMERAS = {
        "wrist_left": {
            "serial_number": "130322273438",
            "dim": (1280, 720),
            "exposure": 10000,
        }
    }
    DISPLAY_IMAGE = False  # 两个分环境不display只在dualenv里display

class RightEnvConfig(EnvConfig):
    SERVER_URL = "http://100.72.16.108:5001/"
    REALSENSE_CAMERAS = {
        "wrist_right": {
            "serial_number": "130322270883",
            "dim": (1280, 720),
            "exposure": 10000,
        },
        # "head": {
        #     "serial_number": "13032227177",
        #     "dim": (1280, 720),
        #     "exposure": 10000,
        # }
    }
    DISPLAY_IMAGE = False

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left/wrist_left", "right/wrist_right"]
    classifier_keys = ["right/wrist_right"] #训练分类器用的camera
    proprio_keys = ["left/tcp_pose", "left/tcp_vel", "left/gripper_pose",
                    "right/tcp_pose", "right/tcp_vel", "right/gripper_pose"]
    buffer_period = 1000 # 每buffer_period个transition存一次进pickle file
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "dual-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, gvl=False):
        left_env = PiperEnv( 
            fake_env=fake_env,
            save_video=save_video,
            config=LeftEnvConfig,
        )

        right_env = PiperEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=RightEnvConfig,
        )
    
        env = DualPiperEnv(left_env, right_env)
        if not fake_env:
            env = DualPicoIntervention(env) # 加中断信息以及轨迹替换
        env = DualRelativeFrame(env) 
        # env = DualQuat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys) # flatten图片外的其他proprio信息
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier and self.classifier_keys is not None:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                p = sigmoid(classifier(obs))
                return int(p[0] > 0.75 and obs['state'][0, 0] > 0.5)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        elif gvl:
            pass
        
        # 这个wrapper是用来算gripper的penalty的, 每一次开合都会惩罚一次，经过这个wrapper
        # 返回的reward还是binary的但在info里会多一项“grasp_penalty”，在后面如果setup_mode
        # 是learned-gripper就会在reward里加上这个penalty
        env = DualGripperPenaltyWrapper(env, penalty=-0.02)

        return env

