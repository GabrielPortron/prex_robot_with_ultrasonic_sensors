import random

import gymnasium as gym
import numpy as np
import torch
import os
import pickle
import numpy as np
import math
import cv2

from zmq import device


def check_vanishing_gradient(model, epoch=0):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()  # L2 norm of gradients
            print(f"Epoch {epoch}, Layer {name}, Gradient Norm: {grad_norm}")
            if grad_norm < 1e-6:
                print(f"Warning: Vanishing gradient detected in layer {name}!")


def fill_replay_buffer(replay_buffer, env):
    action = np.zeros(2)
    done = False
    state = np.zeros(env.state_space[0])
    for d2 in range(30, 295):
        d1 = 325 - d2
        for d3 in range(30, 160):
            d4 = 190 - d3
            state[:4] = abs(np.array([d1, d2, d3, d4]) / 1e2 - env.goal1)
            state[4:8] = abs(np.array([d1, d2, d3, d4]) / 1e2 - env.goal2)
            state[8] = -1.0 if sum(state[:4]) > sum(state[4:8]) else 1.0
            state[9:13] = np.array([d1, d2, d3, d4])

            reward, done = env._compute_reward(state, action)
            # print(state, reward, done)
            replay_buffer.add(state, state.copy(), action, reward, done)

    return replay_buffer


def discretize_actions1(action, bounds):
    vl, va = action
    half_vl, half_va = bounds / 2
    discretized_action = np.zeros(2)
    if abs(vl) < half_vl / 2:
        discretized_action[0] = 0.0
    else:
        if vl > 0:
            discretized_action[0] = half_vl
        else:
            discretized_action[0] = -half_vl

    if abs(va) < half_vl / 2:
        discretized_action[1] = 0.0
    else:
        if va > 0:
            discretized_action[1] = half_va
        else:
            discretized_action[1] = -half_va

    return discretized_action


def discretize_actions2(action, bounds, n_acctions=4):
    vl, va = action
    increment_vl = bounds[0] * 2 / (n_acctions + 2)  # +1 for zero
    steps_vl = [-bounds[0] * increment_vl * i for i in range(n_acctions)]
    increment_va = bounds[1] * 2 / (n_acctions + 2)  # +1 for zero
    steps_va = [-bounds[1] * increment_va * i for i in range(n_acctions)]
    discretized_action = np.zeros(2)

    if (
        action[0] < steps_vl[(n_acctions / 2 + 1)]
        and action[0] > steps_vl[(n_acctions / 2 - 1)]
    ):
        action[0] = 0.0
    else:
        for step in steps_vl:
            if action[0] <= step:
                discretized_action[0] = step

    if (
        action[1] < steps_va[(n_acctions / 2 + 1)]
        and action[1] > steps_va[(n_acctions / 2 - 1)]
    ):
        action[1] = 0.0
    else:
        for step in steps_va:
            if action[1] <= step:
                discretized_action[1] = step

    return discretized_action


def load_replay_buffer(filename="replay_buffer"):
    print("loading replay buffer...")
    file = open(filename, "rb")
    replay_buffer = pickle.load(file)
    file.close()
    print("Done")
    print(len(replay_buffer))
    return replay_buffer


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def euler_to_quaternion(orientation, degrees=False, inverted=False):

    roll = orientation[0]
    pitch = orientation[1]
    yaw = orientation[2]
    
    if degrees:
        roll *= math.pi / 180
        pitch *= math.pi / 180
        yaw *= math.pi / 180
    
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    if inverted:
        return qw, qx, qy, qz
    else:
        return qx, qy, qz, qw

def quaternion_to_euler(orientation, degrees=False, inverted=False):

    if inverted:
        qw = orientation[0]
        qx = orientation[1]
        qy = orientation[2]
        qz = orientation[3]
    else:
        qx = orientation[0]
        qy = orientation[1]
        qz = orientation[2]
        qw = orientation[3]

    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)
    
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)
    
    if degrees:
        roll *= 180 / math.pi
        pitch *= 180 / math.pi
        yaw *= 180 / math.pi

    return roll, pitch, yaw

def create_video(video_path, images_path, fps):

    video = cv2.VideoWriter(
        filename=video_path,
        fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
        fps=fps,
        frameSize=(1024, 1024)
    )

    images = [img for img in os.listdir(images_path)]
    images.sort()
    
    for image in images:
        video.write(cv2.imread(os.path.join(images_path, image)))
        video.write(cv2.imread(os.path.join(images_path, image)))

    video.release()
    print("Video succesfully generated !")

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/self.py


class ReplayBuffer:

    @torch.no_grad()
    def __init__(
        self,
        capacity=10_000,
        batch_size=32,
        state_shape=(4, 120, 160),
        action_shape=(1, 2),
        device="cpu",
        normalize_rewards=False,
    ):
        self.device = device
        # self.content = []
        self.state_shape = state_shape

        self.capacity = capacity
        self.idx = 0
        self.filled = False
        self.batch_size = batch_size
        self.indices = np.zeros(batch_size)
        self.normalize_rewards = normalize_rewards
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.states_img = (
            torch.empty((capacity, *state_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.actions = (
            torch.empty((capacity, *action_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.rewards = (
            torch.empty(capacity, dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img = (
            torch.empty((capacity, *state_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.dones = torch.empty(capacity, dtype=torch.bool).detach().to(self.device)

    def change_bacth_size(self, batch_size):
        self.batch_size = batch_size
        del self.indices
        self.indices = np.zeros(batch_size)

    def save(self, filename="replay_buffer"):
        print("saving replay buffer...")
        file = open(filename, "wb")
        pickle.dump(self, file, 4)
        file.close()
        print("Done")

    @torch.no_grad()
    def add(self, obs, next_obs, actions, rewards, dones):
        temp_tensor = torch.tensor(obs, dtype=torch.float32).detach().to(self.device)
        size_sample = temp_tensor.shape[0]
        interval = [self.idx, 0]
        if self.idx + size_sample < self.capacity:
            interval[1] = self.idx + size_sample
        else:
            interval[1] = None  # size_sample + interval[0]

        self.states_img[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(actions, dtype=torch.float32).detach().to(self.device)
        )
        self.actions[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(rewards, dtype=torch.float32).detach().to(self.device)
        )
        self.rewards[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(next_obs, dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        temp_tensor = torch.tensor(dones, dtype=torch.float32).detach().to(self.device)
        self.dones[interval[0] : interval[1]] = temp_tensor
        del temp_tensor

        if self.idx + size_sample == self.capacity:
            self.filled = True

        self.idx = (self.idx + size_sample) % self.capacity

    def can_sample(self):
        res = False
        # if len(self) >= self.capacity:
        if len(self) >= self.batch_size * 2:
            res = True
        # print(f"{len(self)} collected")
        return res

    @torch.no_grad()
    def sample(self, sample_capacity=None, device="cpu"):
        if self.can_sample():
            if sample_capacity:
                idx = np.random.randint(0, len(self), sample_capacity)

            else:
                idx = np.random.randint(0, len(self), self.batch_size)

            self.indices[:] = idx

            rewards = self.rewards[self.indices].to(device)

            if self.normalize_rewards is True:
                rewards = (rewards - self.rewards.min()) / (
                    self.rewards.max() - self.rewards.min()
                )
            return (
                self.states_img[self.indices].to(device),
                self.actions[self.indices].to(device),
                rewards,
                self.next_states_img[self.indices].to(device),
                self.dones[self.indices].to(device),
            )
        else:
            assert "Can't sample: not enough elements!"

    def __len__(self):
        size = 0
        if self.filled:
            size = self.dones.shape[0]
        else:
            size = self.idx
        return size

    @torch.no_grad()
    def increase_capacity(self, new_capacity):
        assert new_capacity > self.capacity
        increment = new_capacity - self.capacity
        self.states_img = torch.cat(
            (
                self.states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        )
        self.actions = torch.cat(
            (
                self.actions,
                torch.empty((increment, *self.action_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.rewards = torch.cat(
            (
                self.rewards,
                torch.empty(increment, dtype=torch.float32).detach().to(self.device),
            )
        ).to(self.device)

        self.next_states_img = torch.cat(
            (
                self.next_states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.dones = torch.cat(
            (
                self.dones,
                torch.empty(increment, dtype=torch.bool).detach().to(self.device),
            )
        ).to(self.device)
        self.filled = False
        self.idx = self.capacity
        self.capacity = new_capacity


@torch.no_grad()
def saturate_replay_buffer(replay_buffer):
    print("Saturating replay buffer...")

    cursor = replay_buffer.idx
    while cursor < replay_buffer.capacity:

        if cursor * 2 < replay_buffer.capacity:
            x = cursor
        else:
            x = replay_buffer.capacity - cursor

        replay_buffer.states_img[cursor : cursor + x] = replay_buffer.states_img[:x]
        replay_buffer.actions[cursor : cursor + x] = replay_buffer.actions[:x]
        replay_buffer.rewards[cursor : cursor + x] = replay_buffer.rewards[:x]
        replay_buffer.next_states_img[cursor : cursor + x] = (
            replay_buffer.next_states_img[:x]
        )
        replay_buffer.dones[cursor : cursor + x] = replay_buffer.dones[:x]

        cursor += x
    replay_buffer.idx = cursor
    print("Done")


import configparser
import time


def read_file_if_modified(args, file_path, last_mod_time):
    # Get the current modification time of the file
    current_mod_time = os.path.getmtime(file_path)

    # If the file has been modified since the last read
    if current_mod_time != last_mod_time:
        with open(file_path, "r") as file:
            args = parse_arguments_from_ini(file_path)

        print("modifications in config.ini ")
        # Update the last modification time
        return current_mod_time, args, True

    return last_mod_time, args, False


def parse_arguments_from_ini(file_path):
    config = configparser.ConfigParser()
    config.read_file(open(file_path))

    arguments = {}

    for section, cfg in config.items():
        for key, value in config[section].items():
            print("Parsing the key: ", key)
            if value.lower() in ["none", "null"]:
                value = None
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            # elif "/" in value:
            #     pass
            elif "." in value:
                value = float(value)
            elif "'" in value or '"' in value:
                value = value[1:-1]
            else:
                try:
                    value = int(value)
                except:
                    # is a string
                    pass

            arguments[key] = value

    return arguments


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward
