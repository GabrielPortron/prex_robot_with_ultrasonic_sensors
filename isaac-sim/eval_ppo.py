#!/usr/bin/env python3
"""
eval_ppo.py — Evaluate a trained PPO agent on the Create3 reach-center task.

Usage:
    python eval_ppo.py --model models/20240422_103000/ppo_prex_final
    python eval_ppo.py --model models/20240422_103000/ppo_prex_final --episodes 20
    python eval_ppo.py --model models/20240422_103000/ppo_prex_final --render
    python eval_ppo.py --model models/20240422_103000/ppo_prex_final --snapshot
"""

import os
import argparse
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
parser.add_argument("--model",    type=str, required=True,
                    help="Path to the saved model zip (without .zip extension)")
parser.add_argument("--episodes", type=int, default=10,
                    help="Number of evaluation episodes (default: 10)")
parser.add_argument("--render",   action="store_true",
                    help="Enable Isaac Sim viewport rendering")
parser.add_argument("--snapshot", action="store_true",
                    help="Save a top-down PNG snapshot at the end of each episode")
parser.add_argument("--deterministic", action="store_true", default=True,
                    help="Use deterministic actions (default: True)")
parser.add_argument("--max-steps", type=int, default=1000,
                    help="Max steps per episode (default: 1000)")
args = parser.parse_args()


# ── Environment ───────────────────────────────────────────────────────────────
from envs.prex_isaac_env import PrexIsaacEnv

print(f"[eval_ppo] Loading environment...")
raw_env = PrexIsaacEnv(
    max_episode_length = args.max_steps,
    max_linear_speed   = 0.7,
    max_angular_speed  = 0.4,
    radius_target      = 0.3,
    physics_dt         = 1.0 / 60.0,
    rendering_dt       = 1.0 / 60.0 if args.render else 1.0,
    repeating_action   = 1,
    verbose            = True,
    arena_geometry     = [(2.0, 2.0), 0.2, 0.5],
)

vec_env = DummyVecEnv([lambda: raw_env])

model_path = os.path.join(args.model, "ppo_prex_final")
env_path = os.path.join(args.model, "vec_normalize.pkl")

env = VecNormalize.load(
    env_path,
    vec_env
)

env.training   = False
env.norm_reward = False

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[eval_ppo] Loading model from {model_path}...")
model = PPO.load(model_path, env=env, device="cpu")
print(f"[eval_ppo] Model loaded.")


# ── Optional snapshot setup ───────────────────────────────────────────────────
if args.snapshot:
    import omni.replicator.core as rep
    import carb

    snapshot_dir = os.path.join(
        "eval_snapshots", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    camera = rep.create.camera(
        position=(0, 0, 4),
        rotation=(-90, 0, 0),
        focal_length=12.0,
        clipping_range=(0.01, 100.0),
    )
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))
    carb.settings.get_settings().set(
        "/exts/omni.replicator.core/maxAssetLoadingTime", 10.0
    )
    carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)

    snapshot_writer = rep.WriterRegistry.get("BasicWriter")
    snapshot_writer.initialize(output_dir=snapshot_dir, rgb=True)
    snapshot_writer.attach([render_product])

    print(f"[eval_ppo] Snapshots will be saved to {snapshot_dir}/")


# ── Evaluation loop ───────────────────────────────────────────────────────────
episode_rewards  = []
episode_lengths  = []
success_count    = 0
flip_count       = 0
timeout_count    = 0

print(f"\n[eval_ppo] Running {args.episodes} episodes...\n")
print(f"{'Episode':>8} {'Steps':>8} {'Return':>10} {'Dist':>8} {'Outcome'}")
print("-" * 55)

for ep in range(args.episodes):

    ep_reward = 0
    ep_steps = 0
    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, info = env.step(action)

        ep_reward += reward[0]   # reward is also wrapped in a list by VecEnv
        ep_steps  += 1

        if done[0]:
            term_reason = raw_env.info.get("terminate", "")

            if "goal" in term_reason or ep_reward > 90:
                outcome = "SUCCESS"
                success_count += 1
            elif "flipped" in term_reason:
                outcome = "flipped"
                flip_count += 1
            else:
                outcome = "timeout"
                timeout_count += 1

            obs = env.reset()   # ← assign back to obs
            break

    episode_rewards.append(ep_reward)
    episode_lengths.append(ep_steps)

    print(f"{ep+1:>8} {ep_steps:>8} {ep_reward:>10.2f} "
          f"{raw_env.dist:>8.3f} {outcome}")

    # Take a snapshot at the end of this episode
    if args.snapshot:
        for _ in range(10):
            raw_env.world.step(render=True)
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        for _ in range(3):
            raw_env.world.step(render=True)


# ── Results summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  Episodes evaluated : {args.episodes}")
print(f"  Success rate       : {success_count}/{args.episodes} "
      f"({100 * success_count / args.episodes:.1f}%)")
print(f"  Flipped            : {flip_count}")
print(f"  Timeouts           : {timeout_count}")
print(f"  Mean return        : {np.mean(episode_rewards):.2f} "
      f"± {np.std(episode_rewards):.2f}")
print(f"  Min / Max return   : {np.min(episode_rewards):.2f} / "
      f"{np.max(episode_rewards):.2f}")
print(f"  Mean episode length: {np.mean(episode_lengths):.1f} steps")
print("=" * 55)

env.close()