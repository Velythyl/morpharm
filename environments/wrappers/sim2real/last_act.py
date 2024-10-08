import gymnasium.spaces
import numpy as np
import torch
from gymnasium import Wrapper



class LastActEnv(Wrapper):
    def __init__(self, env, device):
        super().__init__(env)

        assert len(self.observation_space.shape[1:]) == 1

        NUM_OBS = self.observation_space.shape[1] + self.action_space.shape[1]

        self.obs_space_shape = (self.observation_space.shape[0], NUM_OBS)
        self.observation_space = gymnasium.spaces.Box(low=np.ones(self.obs_space_shape) * -np.inf,
                                                high=np.ones(self.obs_space_shape) * np.inf)
        self.device = device

    def make_obs(self, obs, act=None):
        if act is None:
            act = torch.zeros((self.observation_space.shape[0], self.action_space.shape[1]), device=self.device)
        else:
            act = act.to(self.device)
        return torch.hstack((obs, act))

    def reset(self, **kwargs):
        obs = super(LastActEnv, self).reset(**kwargs)
        return self.make_obs(obs)

    def step(self, action):
        ret = super(LastActEnv, self).step(action)

        return self.make_obs(ret[0], action), *ret[1:]


