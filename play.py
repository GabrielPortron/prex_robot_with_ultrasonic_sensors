#!/usr/bin/env python3
import torch
import os
import numpy as np


from utils.utils import (
    ReplayBuffer,
    parse_arguments_from_ini,
)


from algorithms.sac import SAC

import os.path as op


import rclpy
from envs.prex_ultrasonic_sensor_prex2 import PrexWorld

rclpy.init()


file_config_path = op.join(__file__[: -len("play.py")], "config.ini")
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)

device = "cuda"

env = PrexWorld(
    max_episode_length=10_000,
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
state_dim = env.state_space
action_dim = env.action_space
replay_buffer = ReplayBuffer(
    args["replay_buffer_size"],
    batch_size,
    state_dim,
    action_dim,
    normalize_rewards=False,
    device=device,
)

# define an agent
agent = SAC(
    "prex_ultrasonic_robot",
    env.state_space,
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
obs, _, _, _ = env.reset()
running_avg_reward = 0
running_avg_steps = 0


def update():
    global obs
    global tot_episodes
    global timesteps
    global running_avg_reward
    global running_avg_steps

    global eps_return

    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action, _ = agent.select_action(
        torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    )
    action = action[0]
    # round action
    action = np.round(action, 4)

    if action[0] == 0.0:
        if action[1] != 0.0:
            pass

    next_obs, reward, _, done = env.step(action)
    running_avg_reward = (running_avg_reward * (timesteps) + reward) / (timesteps + 1)

    print(
        f"eps = {tot_episodes} step_count = {timesteps}, reward = {reward:.3f}, runn_avg_reward = {running_avg_reward:.3f}"
    )

    eps_return += reward
    obs = next_obs

    if done is True:
        running_avg_steps = (running_avg_steps * (tot_episodes) + env.step_counter) / (
            tot_episodes + 1
        )
        print(f"runn_avg_steps = {running_avg_steps:.3f}")
        tot_episodes += 1
        timesteps = 0
        eps_return = 0.0
        input("press enter to reset...")
        obs, _, _, done = env.reset()

    timesteps += 1


folder_name = args["folder_name"]
path = args["path"]
fl = args["best_model"]

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
while True:
    update()
