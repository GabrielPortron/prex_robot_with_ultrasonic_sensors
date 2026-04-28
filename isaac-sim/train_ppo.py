#!/usr/bin/env python3
"""
train_ppo.py — PPO training for Create3 reach-center task.

Uses Stable Baselines 3 + PrexIsaacEnv (Gymnasium-compatible).

Usage:
    python train_ppo.py                        # fresh training
    python train_ppo.py --resume models/best   # resume from checkpoint
"""

import os
import argparse
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from curriculum_callback import CurriculumCallback

import wandb
from wandb.integration.sb3 import WandbCallback


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume",   type=str, default=None,
                    help="Path to a saved model zip to resume training from")
parser.add_argument("--no-wandb", action="store_true",
                    help="Disable Weights & Biases logging")
args = parser.parse_args()


# ── Config ────────────────────────────────────────────────────────────────────
RUN_NAME    = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR  = os.path.join("models", RUN_NAME)
LOGS_DIR    = os.path.join("logs",   RUN_NAME)
BEST_DIR    = os.path.join(MODELS_DIR, "best")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)
os.makedirs(BEST_DIR,   exist_ok=True)

PPO_CONFIG = dict(
    learning_rate        = 1e-4,
    n_steps              = 4096,     # steps per rollout per env
    batch_size           = 128,
    n_epochs             = 5,       # gradient update passes per rollout
    gamma                = 0.995,     # discount factor
    gae_lambda           = 0.95,     # GAE smoothing
    clip_range           = 0.1,      # PPO clip epsilon
    ent_coef             = 0.005,     # entropy bonus (encourages exploration)
    vf_coef              = 0.5,      # value loss weight
    max_grad_norm        = 0.5,
    policy_kwargs        = dict(net_arch=[256, 256, 128]),  # actor + critic MLP size
    verbose              = 1,
    tensorboard_log      = LOGS_DIR,
    device               = "cuda",
)

TOTAL_TIMESTEPS    = 500_000
EVAL_FREQ          = 10_000    # evaluate every N steps
EVAL_EPISODES      = 5         # episodes per evaluation
CHECKPOINT_FREQ    = 50_000    # save a checkpoint every N steps


# ── Environment ───────────────────────────────────────────────────────────────
# Import after SimulationApp would be launched inside the env
from envs.prex_isaac_env import PrexIsaacEnv

print("[train_ppo] Creating environment...")
raw_env = PrexIsaacEnv(
    max_episode_length = 1000,
    max_linear_speed   = 0.7,
    max_angular_speed  = 0.4,
    radius_target      = 0.3,
    physics_dt         = 1.0 / 60.0,
    rendering_dt       = 1.0,
    repeating_action   = 1,
    verbose            = False,
    arena_geometry     = [(2.0, 2.0), 0.2, 0.5],
)

monitored_env = Monitor(raw_env, filename=os.path.join(LOGS_DIR, "monitor.csv"))

vec_env = DummyVecEnv([lambda: monitored_env])

env = VecNormalize(
    vec_env,
    norm_obs    = True,    # normalise observations
    norm_reward = True,    # normalise rewards
    clip_obs    = 10.0,    # clip normalised obs to [-10, 10]
    gamma       = 0.995,   # must match PPO gamma for reward normalisation
)

# ── Weights & Biases ──────────────────────────────────────────────────────────
callbacks = []

if not args.no_wandb:
    wandb.init(
        project  = "prex-isaac-ppo",
        name     = RUN_NAME,
        config   = {**PPO_CONFIG, "total_timesteps": TOTAL_TIMESTEPS},
        sync_tensorboard = True,
        save_code        = True,
    )
    callbacks.append(
        WandbCallback(
            gradient_save_freq = 1000,
            model_save_path    = os.path.join(MODELS_DIR, "wandb"),
            verbose            = 1,
        )
    )


# ── SB3 Callbacks ────────────────────────────────────────────────────────────

callbacks.append(
    CheckpointCallback(
        save_freq  = CHECKPOINT_FREQ,
        save_path  = MODELS_DIR,
        name_prefix= "ppo_prex",
        verbose    = 1,
    )
)

callbacks.append(CurriculumCallback())

callback = CallbackList(callbacks)


# ── Agent ─────────────────────────────────────────────────────────────────────
if args.resume:
    print(f"[train_ppo] Resuming from {args.resume}")
    model = PPO.load(
        args.resume,
        env          = env,
        tensorboard_log = LOGS_DIR,
        device       = PPO_CONFIG["device"],
    )
else:
    print("[train_ppo] Creating new PPO agent...")
    model = PPO("MlpPolicy", env, **PPO_CONFIG)

print(f"[train_ppo] Policy architecture:\n{model.policy}")


# ── Training ──────────────────────────────────────────────────────────────────
print(f"[train_ppo] Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
try:
    model.learn(
        total_timesteps  = TOTAL_TIMESTEPS,
        callback         = callback,
        reset_num_timesteps = not bool(args.resume),
        progress_bar     = True,
    )
except KeyboardInterrupt:
    print("\n[train_ppo] Training interrupted — saving current model.")

# ── Save final model ──────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "ppo_prex_final")
env_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")
model.save(model_path)
env.save(env_path)

print(f"[train_ppo] Final model saved to {model_path}.zip")
print(f"[train_ppo] Final normalisation stats saved to {env_path}.zip")

if not args.no_wandb:
    wandb.finish()

env.close()