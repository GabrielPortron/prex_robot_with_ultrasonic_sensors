import torch
from datetime import datetime
import os
import numpy as np


from utils.utils import (
    ReplayBuffer,
    parse_arguments_from_ini,
    read_file_if_modified,
)

from algorithms.ppo import PPO

import wandb
import os.path as op


import rclpy
from envs.prex_ultrasonic_sensor import PrexWorld

rclpy.init()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="prex_ultrasonic-sac",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "fc",
        "dataset": "coppelia-prex",
        "epochs": 0,
    },
)

file_config_path = op.join(__file__[: -len("train_ppo.py")], "config.ini")
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)

device = "cpu"#"cuda"
env = PrexWorld(
    max_episode_length=args["max_steps"],
    max_linear_speed=args["max_linear_speed"],
    max_angular_speed=args["max_angular_speed"],
    topic_pub=args["topic_pub"],
    type_ros2_msg=args["type_ros2_msg"],
    dt=args["dt"],
    verbose=args["verbose"],
    time_factor=args["time_factor"],
    clipping_limit=args["clipping_limit"],
    radius_target=args["radius_target"],
    max_speed_bonus=args["max_speed_bonus"],
    repeating_action=args["repeating_action"],
    # square
    perimeter=(2, 2),
)

# create replay buffer
batch_size = args["batch_size"]
n_sensors = args["n_sensors"]
state_dim = env.state_space  # Shape of state input (4, 84, 84)
action_dim = env.action_space
# replay_buffer = ReplayBuffer(
#     args["replay_buffer_size"],
#     batch_size,
#     state_dim,
#     action_dim,
#     normalize_rewards=False,
#     device=device,
# )

# define an agent
agent  = PPO(env,
             timesteps_per_batch=450,
             max_timesteps_per_episode=args["max_steps"],
             n_update_per_iteration=5,
             lr=args["ppo_lr"],
             gamma=args["gamma"],
             clip=args["clip"],
             lam=args["lam"],
             num_minibatches=6,
             ent_coef=0.0,
             target_kl=0.02,
             max_grad_norm=0.5,
             device=device
             )

agent.learn(total_timesteps=100_000)
# tot_episodes = 0
# timesteps = 0
# probability_training = 1.0
# save_on_episodes = args["save_on_episode"]
# running_avg_reward = 0
# running_avg_steps = 0

# folder_name = os.path.join("models", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
# path = args["path"]

# eps_return = 0
# once = True
# collect_random_timesteps = args["collect_random_steps"]

# obs, _, _, _ = env.reset()

# dict_derivatives_v ={"speed":0,"acc":0,"jerk":0,"snap":0,"crackle":0,"pop":0}
# dict_derivatives_w ={"speed":0,"acc":0,"jerk":0,"snap":0,"crackle":0,"pop":0}

# def update_dict(dict_,derivatives):
#     i=0
#     for k,v in dict_.items():
#         dict_[k] = derivatives[i]
#         i+=1
#     return dict_
