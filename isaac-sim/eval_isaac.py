import torch
from datetime import datetime
import os
import time
import numpy as np

from utils.utils import(
    ReplayBuffer,
    parse_arguments_from_ini,
    read_file_if_modified
)

from algorithms.sac import SAC

import os.path as op

from envs.prex_isaac_env import PrexIsaacEnv

# --- 1 - Initializations -------------------------------------------------
RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = os.path.join("models", RUN_NAME)
LOGS_DIR = os.path.join("logs", RUN_NAME)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

file_config_path = "isaac-sim/config.ini"
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)

TOTAL_TIMESTEPS = 1_000_000

device = "cuda"

# --- 2 - Creating environment --------------------------------------------
print("[Evaluating] I - Creating environment...")

env = PrexIsaacEnv(
    max_episode_length=args["max_steps"],
    max_linear_speed=args["max_linear_speed"],
    max_angular_speed=args["max_angular_speed"],
    radius_target=args["radius_target"],
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0,
    verbose=args["verbose"],
    clipping_limit=args["clipping_limit"],
    max_speed_bonus=args["max_speed_bonus"],
    repeating_action=args["repeating_action"],
    device=device,
    arena_geometry=[(2.0, 2.0), 0.2, 0.5],
)

print("[Evaluating] ... Environment created")

# --- 3 - Creating replay buffer ------------------------------------------
print("[Evaluating] II - Creating replay buffer...")

batch_size = args["batch_size"]
state_dim = env.observation_space.shape
action_dim = env.action_space.shape
replay_buffer = ReplayBuffer(
    capacity=args["replay_buffer_size"],
    batch_size=batch_size,
    state_shape=state_dim,
    action_shape=action_dim,
    device=device,
    normalize_rewards=False
)

print("[Evaluating] ... Replay buffer created")

# --- 4 - Creating agent --------------------------------------------------
print("[Evaluating] III - Creating agent...")

agent = SAC(
    env_name="prex_ultrasonic_robot",
    state_dim=state_dim,
    action_dim=action_dim,
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

print("[Evaluating] ... Agent created")

# --- 5 - Evaluation --------------------------------------------------------
print("[Evaluating] Starting evaluation ...")

agent.set_to_eval_mode()
eps_return = 0
obs, _ = env.reset()

folder_name = "20260429_110343" #args["folder_name"]
path = os.getcwd() + "/isaac-sim/" #args["path"]
nb_episodes = args["n_episodes"]
best_model = None
cumul_return = 0
max_ep_return = -np.inf
step_model = args["step_model"]

with open("results_models.txt", "a") as file:
    file.write(f"{folder_name}\n")

for fl in range(args["start_model"], args["end_model"], 300):

    try:
        agent.load_weights(path, folder_name, fl)
        print("\n\n",
               "*" * 100,
               f"\n model: {fl} \n\n",
               "*" * 100
            )
        tot_episodes = 0
        timesteps = 0
        running_avg_reward = 0
        t = time.time()
        cumul_return = 0
        avg_cumul_return = 0

        while tot_episodes < nb_episodes:

            action, entropy = agent.select_action(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            )
            action = action[0]
            action = np.round(action, 4)

            linear_vel = action[0]
            angular_vel = action[1]

            if abs(linear_vel) < 0.1:
                linear_vel = 0.0
            if abs(angular_vel) < 0.05:
                angular_vel = 0.0

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            running_avg_reward += (reward - running_avg_reward) / (timesteps + 1)

            print(
                f"model = {fl}, eps = {tot_episodes}, step_count = {timesteps}, "
                f"reward = {reward:.3f}, runn_avg_reward = {running_avg_reward:.3f}, "
                f"distance = {env.dist:.3f}"
            )

            eps_return += reward
            obs = next_obs

            if done:
                obs, _ = env.reset()
                tot_episodes += 1
                cumul_return += eps_return
                eps_return = 0
            
            timesteps += 1
        
        avg_cumul_return = cumul_return / nb_episodes

        with open("results_models.txt", "a") as file:
            file.write(
                f"model: {fl}, episodes: {nb_episodes}, return: {avg_cumul_return}\n"
            )
        
        if  avg_cumul_return > max_ep_return:
            best_model = fl
            max_ep_return = avg_cumul_return

            with open("results_models.txt", "a") as file:
                file.write(
                    f"model:{best_model}, episodes:{nb_episodes}, return:{max_ep_return}\n"
                )

            print("Content appended successfully.")
    
    except Exception as e:
        print("EXCEPTION OCCURED:\n", e)
        print(f"The model {fl} does not exist.")

print("The best model is: ", best_model)