#!/usr/bin/env python3
import argparse
import torch
from datetime import datetime
import os
import time
import numpy as np


from utils.utils import (
    ReplayBuffer,
    parse_arguments_from_ini,
)


from algorithms.sac import SAC

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

file_config_path = op.join(__file__[: -len("evaluate.py")], "config.ini")
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)

device = "cuda"

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
replay_buffer = ReplayBuffer(
    args["replay_buffer_size"],
    batch_size,
    state_dim,
    action_dim,
    normalize_rewards=False,
    device=device,
)
# # replay_buffer = load_replay_buffer()

# define an agent
agent = SAC(
    "prex_ultrasonic_robot",
    env.state_space,  # envs[0].observation_space.shape[:3],
    env.action_space,
    replay_buffer=replay_buffer,
    device=device,
    actor_lr=args["actor_lr"],
    critic_lr=args["critic_lr"],
    tau=args["tau"],
    alpha=args["alpha"],
    gamma=args["gamma"],
    action_bounds=(
        (-args["max_linear_speed"], -args["max_angular_speed"]),
        (args["max_linear_speed"], args["max_angular_speed"]),
    ),
)


# set the agent in evaluate mode
agent.set_to_eval_mode()
eps_return = 0
obs, _, _, done = env.reset()


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global running_avg_reward
    global eps_return
    global cumul_return

    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action, entropy = agent.select_action(
        torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    )
    action = action[0]
    # clipping when close to 0
    if abs(action[0]) < 0.1:
        action[0] = 0.0
    if abs(action[1]) < 0.05:
        action[1] = 0.0

    v_linear = action[0]
    v_angular = action[1]
    wandb.log({"entropy": entropy})

    # round action
    action = np.round(action, 4)

    next_obs, reward, info, done = env.step(action)

    running_avg_reward += (reward - running_avg_reward) / (timesteps + 1)
    print(
        f"model = {fl} eps = {tot_episodes} step_count = {timesteps}, reward={reward:.3f}, runn_avg_reward={running_avg_reward:.3f}, distance={env.dist:.3f}"
    )

    wandb.log(
        {
            "v_linear_action": v_linear,
            "v_angular_action": v_angular,
            "v_linear_robot": env.linear_speed,
            "v_angular_robot": env.angular_speed,
            "dist_to_center": env.dist,
            "reward": reward,
            "runn_avg_reward": running_avg_reward,
            "alpha": agent.alpha,
            "theta": env.theta,
        }
    )

    eps_return += reward
    obs = next_obs

    if done is True:
        wandb.log({"ep_return": eps_return, "step_count": env.step_counter})
        obs, _, _, done = env.reset()
        tot_episodes += 1
        cumul_return += eps_return
        eps_return = 0

    timesteps += 1


folder_name = args["folder_name"]
path = args["path"]
nb_episodes = args["n_episodes"]
best_model = None
cumul_return = 0
max_ep_return = -np.inf
step_model = args["step_model"]
with open("results_models.txt", "a") as file:
    file.write(f'{args["folder_name"]}\n')

for fl in range(
    args["start_model"],
    args["end_model"],
):
    try:
        # load model
        agent.load_weights(path, folder_name, fl)
        print(
            "\n\n",
            "*" * 100,
            f"\n model:{fl}\n\n",
            "*" * 100,
        )
        tot_episodes = 0
        timesteps = 0
        running_avg_reward = 0
        dt = args["dt"]
        t = time.time()
        cumul_return = 0
        avg_cumul_return = 0
        while tot_episodes < nb_episodes:
            update(dt)

            # if time.time() - t > dt:
            #     update(dt)
            #     t = time.time()
        avg_cumul_return = cumul_return / nb_episodes
        # Open the file in append mode
        with open("results_models.txt", "a") as file:
            file.write(
                f"model:{fl}, episodes:{nb_episodes}, return:{avg_cumul_return}\n"
            )

        # if  avg_cumul_return > max_ep_return:
        #     best_model = fl
        #     max_ep_return = avg_cumul_return
        #     # Open the file in append mode
        #     with open("results_models.txt", "a") as file:
        #         file.write(
        #             f"model:{best_model}, episodes:{nb_episodes}, return:{max_ep_return}\n"
        #         )

        #     print("Content appended successfully.")

    except Exception as e:
        print("EXCEPTION OCCURRED:\n", e)
        print(f"the model {fl} does not exist.")
print("The best model is: ", best_model)
