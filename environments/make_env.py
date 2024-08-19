import dataclasses
import functools
import gc
import json
import random
from typing import Any, Tuple

import numpy as np
import torch
from gymnasium import Wrapper
from gymnasium.vector import AsyncVectorEnv
import gymnasium

from environments.config_utils import envkey_multiplex, num_multiplex, slice_multiplex, monad_multiplex, \
    splat_multiplex, marshall_multienv_cfg, build_dr_dataclass
from environments.func_utils import monad_coerce
from environments.wrappers.infologwrap import InfoLogWrap
from environments.wrappers.mujoco.domain_randomization import MujocoDomainRandomization
from environments.wrappers.multiplex import MultiPlexEnv
from environments.wrappers.np2torch import Np2TorchWrapper
from environments.wrappers.recordepisodestatisticstorch import RecordEpisodeStatisticsTorch
from environments.wrappers.renderwrap import RenderWrap
from environments.wrappers.sim2real.last_act import LastActEnv
from environments.loggingutils.eval import evaluate
from environments.loggingutils.every_n import EveryN

CUSTOM_ENVS = ["go1", "widow"]

@monad_coerce
def make_mujoco(mujoco_cfg, seed):
    import gymnasium
    ENVNAME = envkey_multiplex(mujoco_cfg)

    import os
    curdir = "/".join(__file__.split("/")[:-1])
    seekdir = f"{curdir}/customenv/mujococustom/assets"
    URDF_PATH = None
    for p, ds, fs in os.walk(seekdir):
        if f"{ENVNAME}.xml" in fs:
            URDF_PATH = f"{p}/{ENVNAME}"
            break
    assert URDF_PATH is not None

    OG_URDF_PATH = f"{URDF_PATH}.xml"
    TARGET_URDF_PATH = f"{URDF_PATH}_target.xml"
    with open(OG_URDF_PATH, "r") as f:
        file = f.read()

    #SEP = "</worldbody>"
    new_file = file #file.split(SEP)

    #new_file = f"""{new_file[0]}
#<body name="morpharm_target" pos="0 0 0.01" gravcomp="1">
#<!-- The joint `ref` attributes are removed for brax. -->
#    <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.66 .66" stiffness="0" type="slide"/>
#    <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.66 .66" stiffness="0" type="slide"/>
#    <joint armature="0" axis="0 0 1" damping="0" limited="true" name="target_z" pos="0 0 0" range="-.66 .66" stiffness="0" type="slide"/>
#    <geom conaffinity="0" contype="0" name="target" pos="0 0 0" size=".009" type="sphere" rgba="0 1 0 1"/>
#</body>
#{SEP}
#{new_file[1]}
#"""

    NUM_OBS = new_file.count("<joint")

    with open(TARGET_URDF_PATH, "w") as f:
        f.write(new_file)

    ENV2RANGE = {
        "wx250s": (0.3, 0.6),
    }

    from environments.customenv.mujococustom.custom_reacher import CustomReacher # noqa

    class SeededEnv(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._seed = seed

        def reset(self, **kwargs):
            ret = super(SeededEnv, self).reset(seed=self._seed)
            import numpy as np
            new_seed = np.random.randint(0, 20000)
            np.random.seed(new_seed)
            self._seed = new_seed
            return ret

    class NoRenderWhenNone(Wrapper):
        def render(self):
            if self.render_mode != "rgb_array":
                return None #np.zeros((self.width, self.height,3))
            else:
                return self.env.render()

    dr_config = build_dr_dataclass(mujoco_cfg)

    def thunk(seed, render_mode):
        np.random.seed(seed)
        random.seed(seed)

        env = gymnasium.make("CustomReacher", max_episode_steps=mujoco_cfg.max_episode_length, autoreset=True, render_mode=render_mode, urdf_name=TARGET_URDF_PATH)
        env = SeededEnv(env)

        if dr_config.DO_DR:
            env = MujocoDomainRandomization(env,
                                            percent_below=dr_config.percent_below,
                                            percent_above=dr_config.percent_above,
                                            do_on_reset=dr_config.do_on_reset,
                                            do_at_creation=dr_config.do_at_creation,
                                            do_on_N_step=dr_config.do_on_N_step
                                            )
        env = NoRenderWhenNone(env)
        return env

    print("Pre async")
    env = AsyncVectorEnv([functools.partial(thunk, seed=seed + i, render_mode="rgb_array" if i ==0 else "depth_array") for i in range(mujoco_cfg.num_env)], shared_memory=True, copy=False, context="fork")
    print("Post async")

    class AsynVectorFixes(Wrapper):
        def close(self):
            return self.env.close(terminate=True)

        def render(self, *args, **kwargs):
            ret = self.env.call_async("render")
            ret = self.env.call_wait()
            return ret[0]   # only return first env's video

    env = AsynVectorFixes(env)

    class NoResetInfoWrapper(Wrapper):
        def reset(self, **kwargs):
            return super(NoResetInfoWrapper, self).reset(**kwargs)[0]

    env = NoResetInfoWrapper(env)
    env = Np2TorchWrapper(env, mujoco_cfg.device)

    class MujocoRenderWrapper(Wrapper):
        def render(self, **kwargs):
            return self.env.render()
    env = MujocoRenderWrapper(env)

    print(f"Mujoco env built: {envkey_multiplex(mujoco_cfg)}")
    return env


@dataclasses.dataclass
class ONEIROS_METADATA:
    cfg: Any

    prefix: str

    single_action_space: Tuple
    single_observation_space: Tuple

    multi_action_space: Tuple
    multi_observation_space: Tuple

    @property
    def env_key(self):
        return self.cfg.env_key


def get_json_identifier(sliced_multiplex_env_cfg):
    dico = dict(vars(sliced_multiplex_env_cfg))["_content"]
    dico = {key: str(val) for key, val in dico.items()}
    return json.dumps(dico).replace(" ", "")

    prefix = envkey_multiplex(sliced_multiplex_env_cfg)
    dr_config = dataclasses.asdict(build_dr_dataclass(sliced_multiplex_env_cfg))

    dr_config["framestack"] = sliced_multiplex_env_cfg.framestack
    dr_config["mat_framestack_instead"]

    dr_config_json = json.dumps(dataclasses.asdict(dr_config))
    prefix = f"{prefix} {dr_config_json}"

def make_multiplex(multiplex_env_cfg, seed, MAX_OBS_SPACE=None, MAX_ACT_SPACE=None):
    #KEEP_ALIVE = KeepAlive()

    base_envs = []
    for sliced_multiplex in splat_multiplex(multiplex_env_cfg):
        base_envs += make_mujoco(sliced_multiplex, seed)

    base_envs = list(filter(lambda x: x is not None, base_envs))
    assert len(base_envs) == num_multiplex(multiplex_env_cfg)

    def single_action_space(env):
        return env.action_space.shape[1:]

    def single_observation_space(env):
        return env.observation_space.shape[1:]

    def num_envs(env):
        assert env.observation_space.shape[0] == env.action_space.shape[0]
        return env.observation_space.shape[0]


    LAST_ACTION = slice_multiplex(multiplex_env_cfg, 0).last_action

    if MAX_OBS_SPACE is None and MAX_ACT_SPACE is None:
        MAX_OBS_SPACE = base_envs[-1].observation_space
        MAX_ACT_SPACE = base_envs[-1].action_space

    for i, env in enumerate(base_envs):
        if LAST_ACTION:
            env = LastActEnv(env, device=multiplex_env_cfg.device[0])

        def nan_to_num(x):
            return torch.nan_to_num(torch.nan_to_num(x, nan=-np.inf), neginf=-1000, posinf=1000)
        class NanToNumObs(gymnasium.Wrapper):
            def reset(self, **kwargs):
                return nan_to_num(self.env.reset(**kwargs))
            def step(self, action):
                action = nan_to_num(action)
                rets = super().step(action)
                return nan_to_num(rets[0]), nan_to_num(rets[1]), *rets[2:]
        base_envs[i] = NanToNumObs(env)

        if base_envs[i].observation_space.shape[-1] > MAX_OBS_SPACE.shape[-1]:
            MAX_OBS_SPACE = base_envs[i].observation_space
        if base_envs[i].action_space.shape[-1] > MAX_ACT_SPACE.shape[-1]:
            MAX_ACT_SPACE = base_envs[i].action_space

    class Padder(gymnasium.Wrapper):
        def __init__(self, env):
            super().__init__(env)

            self.observation_space = MAX_OBS_SPACE
            self.action_space = MAX_ACT_SPACE

        def pad_obs(self, obs):
            return torch.nn.functional.pad(obs, pad=(0, self.observation_space.shape[-1] - obs.shape[-1]))

        def reset(self, **kwargs):
            return self.pad_obs(self.env.reset(**kwargs))

        def step(self, action):
            action = action[:,:self.env.action_space.shape[-1]]
            ret = super().step(action)
            return self.pad_obs(ret[0]), *ret[1:]

    for i, env in enumerate(base_envs):
        base_envs[i] = Padder(env)

    PROTO_ACT = single_action_space(base_envs[0])
    PROTO_OBS = single_observation_space(base_envs[0])
    PROTO_NUM_ENV = num_envs(base_envs[0])

    def metadata_maker(cfg, prefix, num_env):
        return ONEIROS_METADATA(cfg, prefix, PROTO_ACT, PROTO_OBS, (num_env, *PROTO_ACT), (num_env, *PROTO_OBS))

    PRIV_KEYS = []
    for i, env in enumerate(base_envs):
        env = RenderWrap(env)
        env = RecordEpisodeStatisticsTorch(env, device=multiplex_env_cfg.device[0],
                                           num_envs=slice_multiplex(multiplex_env_cfg, i).num_env)
        # TODO one prefix for each DR type...
        prefix = envkey_multiplex(slice_multiplex(multiplex_env_cfg, i))
        dr_config = build_dr_dataclass(slice_multiplex(multiplex_env_cfg, i))
        prefix = f"{prefix} {get_json_identifier(slice_multiplex(multiplex_env_cfg, i))}"
        env = InfoLogWrap(env, prefix=prefix)

        assert single_action_space(env) == PROTO_ACT
        assert single_observation_space(env) == PROTO_OBS
        assert num_envs(env) == PROTO_NUM_ENV

        METADATA_PREFIX = f"{envkey_multiplex(slice_multiplex(multiplex_env_cfg, i))} L{slice_multiplex(multiplex_env_cfg, i).dr_percent_below} H{slice_multiplex(multiplex_env_cfg, i).dr_percent_above}"
        METADATA_PREFIX = METADATA_PREFIX.replace(".", ",")
        env.ONEIROS_METADATA = metadata_maker(slice_multiplex(multiplex_env_cfg, i), METADATA_PREFIX, PROTO_NUM_ENV)

        base_envs[i] = env

    env = MultiPlexEnv(base_envs, multiplex_env_cfg.device[0], )
    assert env.observation_space.shape[0] == PROTO_NUM_ENV * len(base_envs)
    env.ONEIROS_METADATA = metadata_maker(multiplex_env_cfg, "MULTIPLEX" if len(base_envs) > 1 else base_envs[0].ONEIROS_METADATA.prefix, PROTO_NUM_ENV * len(base_envs))

    return env, MAX_OBS_SPACE, MAX_ACT_SPACE




def make_train_and_eval(multienv_cfg, seed: int):
    multienv_cfg = marshall_multienv_cfg(multienv_cfg)

    DEBUG_VIDEO = False # True # False #True #False

    if not DEBUG_VIDEO:
        print("Building training envs...")
        train_env, MAX_OBS_SPACE, MAX_ACT_SPACE = make_multiplex(multienv_cfg.train, seed)
        gc.collect()
        print("...done!")

    print("Building eval envs...")
    eval_and_video_envs = []
    DEBUG_ACTION_SEQUENCE = None
    for i, sliced_multiplex in enumerate(splat_multiplex(multienv_cfg.eval)):
        sliced_multiplex = monad_multiplex(sliced_multiplex)
        eval_and_video_envs += [make_multiplex(sliced_multiplex, seed + i + 1, MAX_OBS_SPACE, MAX_ACT_SPACE)[0]]

        if not DEBUG_VIDEO:
            assert eval_and_video_envs[
                       -1].ONEIROS_METADATA.single_action_space == train_env.ONEIROS_METADATA.single_action_space
            assert eval_and_video_envs[
                       -1].ONEIROS_METADATA.single_observation_space == train_env.ONEIROS_METADATA.single_observation_space
        gc.collect()
        if DEBUG_VIDEO:
            NUM_DEBUG_STEPS = 500
            np.random.seed(1)
            if DEBUG_ACTION_SEQUENCE is None:
                DEBUG_ACTION_SEQUENCE = torch.concatenate(
                            [torch.from_numpy(np.random.uniform(low=-10, high=10, size=eval_and_video_envs[-1].action_space.shape[1:])[None]).to("cuda")[None] for i in
                             range(NUM_DEBUG_STEPS)]).detach()

            class Agent:
                def __init__(self):
                    self.i = 0
                    self.actions = DEBUG_ACTION_SEQUENCE
                    self.actions.requires_grad = False

                def get_action(self, *args):
                    self.i = self.i + 1
                    return self.actions[self.i - 1]

            evaluate(nsteps=0, eval_envs=eval_and_video_envs[-1], NUM_STEPS=NUM_DEBUG_STEPS,
                     DO_VIDEO=True, agent=Agent())
    print("...done!")



    EVAL_FREQ = multienv_cfg.eval_freq
    if EVAL_FREQ and EVAL_FREQ != "None":
        eval_envs = eval_and_video_envs
    else:
        eval_envs = []


    hook_steps = []
    hooks = []
    for _env in eval_envs:
        hook_steps.append(EVAL_FREQ)

        func = functools.partial(evaluate, eval_envs=_env, NUM_STEPS=multienv_cfg.num_eval_steps,
                          DO_VIDEO=multienv_cfg.do_eval_video)

        hooks.append(func)
    all_hooks = EveryN(hook_steps, hooks)

    return train_env, all_hooks
