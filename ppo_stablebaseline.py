from stable_baselines3.common.env_checker import check_env

import torch
from datetime import datetime
import os
import time
import numpy as np


from utils.utils import (
    ReplayBuffer,
    parse_arguments_from_ini,
    read_file_if_modified,
)


from algorithms.sac import SAC

import wandb
import os.path as op


# import rclpy
from envs.prex_ultrasonic_sensor import PrexWorld

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# rclpy.init()


# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="prex_ultrasonic-sac",
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.001,
#         "architecture": "fc",
#         "dataset": "coppelia-prex",
#         "epochs": 0,
#     },
# )


# file_config_path = op.join(__file__[: -len("train.py")], "config.ini")
# args = parse_arguments_from_ini(file_config_path)
# last_mod_time = os.path.getmtime(file_config_path)

device = "cpu"#"cuda"
env = PrexWorld(
    max_episode_length=150,
    max_linear_speed=4,
    max_angular_speed=2,
    topic_pub="/prex/action",
    type_ros2_msg="String",
    dt=0.005,
    verbose=False,
    time_factor=10,
    clipping_limit=100,
    radius_target=0.20,
    max_speed_bonus=5.0,
    repeating_action=3,
    # square
    perimeter=(2, 2),
)

check_env(env, warn=True)

# Instantiate the env
vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(grid_size=10))

# Train the agent
model = PPO("MlpPolicy", vec_env, verbose=1).learn(5000)

# Test the trained agent
# using the vecenv
obs, _ = vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, terminated, truncated, info = vec_env.step(action)
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break