import gym
import numpy as np


class EnvironmentWrapper:
    def __init__(self, env_name: str, render_mode: str = None):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.name = env_name

    def get_state_dim(self) -> int:
        return self.env.observation_space.shape[0]

    def get_action_dim(self) -> int:
        return self.env.action_space.n

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs

    def step(self, action: int):
        return self.env.step(action)

    def close(self):
        self.env.close()