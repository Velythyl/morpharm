import gymnasium
import numpy as np
from etils import epath

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from environments.customenv.common_utils import random_sphere_numpy

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ReadObsSize(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, urdf_name, **kwargs):
        xml_file = urdf_name  # str(epath.resource_path('environments') / 'customenv/mujococustom/assets/trossen_wx250s/wx250s.xml')
        utils.EzPickle.__init__(self,
                                xml_file=xml_file,
                                **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

class CustomReacher(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, urdf_name, **kwargs):
        xml_file = urdf_name #str(epath.resource_path('environments') / 'customenv/mujococustom/assets/trossen_wx250s/wx250s.xml')

        temp_env = ReadObsSize(urdf_name=urdf_name)
        num_obs = temp_env.data.qpos.shape[0] + temp_env.data.qvel.shape[0] + 3

        utils.EzPickle.__init__(self,
            xml_file=xml_file,
                                **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.goal = np.array([0.55, 0, 0.3])

    def get_ee_xpos(self):
        return self.data.xpos[-1]

    def step(self, a):
        vec = self.get_ee_xpos() - self.goal
        reward_dist = -np.linalg.norm(vec)
        reward = reward_dist

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, target_pos_raw=self.goal, target_pos=self.goal, tip_pos=self.get_ee_xpos()),
        )

    def _random_target(self):
        return np.array([0.3, 0, 0.3])

        """Returns a target location in a random circle slightly above xy plane."""
        point = random_sphere_numpy( 0.3, 0.6, shape=(1,))[0]
        point[-1] = np.clip(np.abs(point[-1]), 0.1, 0.6)
        return point

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) * 0
            + self.init_qpos
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        ) * 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        self_pos = self.data.qpos.flat
        self_vel = self.data.qvel.flat
        return np.concatenate(
            [
                self_pos,
                self_vel,
                self.goal
            ]
        )

gymnasium.register("CustomReacher", "environments.customenv.mujococustom.custom_reacher:CustomReacher")