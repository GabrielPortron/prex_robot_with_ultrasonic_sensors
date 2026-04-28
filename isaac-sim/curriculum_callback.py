from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CurriculumCallback(BaseCallback):
    """
    Advances the curriculum level in PrexIsaacEnv when the agent's
    rolling success rate exceeds a threshold.

    Curriculum levels:
        0 — spawn within 0.3 m of center   (easy)
        1 — spawn within 0.6 m of center   (medium)
        2 — spawn anywhere in the arena    (full)

    Parameters
    ----------
    success_threshold : float
        Rolling success rate required to advance (default: 0.7 = 70%)
    window_size : int
        Number of recent episodes to average over (default: 50)
    verbose : int
        0 = silent, 1 = print on level change
    """

    def __init__(
        self,
        success_threshold: float = 0.7,
        window_size: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.window_size       = window_size

        self._episode_successes = []   # rolling window of 1/0 per episode
        self._current_level     = 0

    # ── Called once at training start ────────────────────────────────────────
    def _on_training_start(self) -> None:
        env = self._get_env()
        env.curriculum_level = 0
        self._current_level  = 0
        if self.verbose:
            print("[CurriculumCallback] Starting at level 0 (easy).")

    # ── Called after every environment step ──────────────────────────────────
    def _on_step(self) -> bool:
        """
        SB3 calls this after each call to env.step().
        `self.locals["dones"]` is a boolean array (one entry per env).
        `self.locals["infos"]` is a list of info dicts.
        """
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                # An episode just ended — check if it was a success
                terminate_reason = info.get("terminate", "")
                success = 1 if "goal" in terminate_reason else 0
                self._episode_successes.append(success)

                # Keep only the last `window_size` episodes
                if len(self._episode_successes) > self.window_size:
                    self._episode_successes.pop(0)

                # Check if we should advance the curriculum
                if (
                    len(self._episode_successes) >= self.window_size
                    and self._current_level < 2
                ):
                    rolling_rate = np.mean(self._episode_successes)
                    if rolling_rate >= self.success_threshold:
                        self._advance_level(rolling_rate)

        return True   # returning False would stop training

    # ── Advance curriculum ────────────────────────────────────────────────────
    def _advance_level(self, rolling_rate: float) -> None:
        self._current_level += 1
        env = self._get_env()
        env.curriculum_level = self._current_level

        # Reset the window so we re-evaluate at the new level
        self._episode_successes.clear()

        level_names = {0: "easy", 1: "medium", 2: "full"}
        if self.verbose:
            print(
                f"\n[CurriculumCallback] "
                f"Advancing to level {self._current_level} "
                f"({level_names[self._current_level]}) — "
                f"rolling success rate was {rolling_rate:.1%} "
                f"over last {self.window_size} episodes.\n"
            )

    # ── Helper: unwrap the env to get PrexIsaacEnv ────────────────────────────
    def _get_env(self):
        env = self.training_env

        # Unwrap VecNormalize
        if hasattr(env, "venv"):
            env = env.venv

        # Unwrap DummyVecEnv
        if hasattr(env, "envs"):
            env = env.envs[0]

        # Unwrap Monitor
        if hasattr(env, "env"):
            env = env.env

        return env