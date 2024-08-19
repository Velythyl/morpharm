"""
Reads the configuration file and carries out the instructions according to the arguments passed using the settings in the configuration file.
"""
import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from ppo import train

logging.basicConfig(level=logging.INFO)


def do_exp(cfg):
    import wandb

    seed = cfg.multienv.env_seed
    if seed == -1:
        seed = random.randint(0, 20000)
        cfg.multienv.env_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from environments.make_env import make_train_and_eval
    train_envs, eval_hooks = make_train_and_eval(cfg.multienv, seed)

    logging.info("==========Begin trainning the Agent==========")

    train(train_envs, eval_hooks)

    logging.info("==========Trainning Completed==========")

    exit()


@hydra.main(version_base=None, config_path="config", config_name="conf")
def main(cfg):
    def fill_in_cfg(cfg):
        omegaconf2dict = lambda c: OmegaConf.to_container(c)
        dict2omegaconf = lambda c: OmegaConf.create(c)

        train_env = cfg.train_env
        eval_env = cfg.eval_env

        cfg = omegaconf2dict(cfg)

        cfg["multienv"]["train"] = omegaconf2dict(train_env)
        cfg["multienv"]["eval"] = omegaconf2dict(eval_env)

        del cfg["train_env"]
        del cfg["eval_env"]
        cfg = dict2omegaconf(cfg)
        return cfg

    cfg = fill_in_cfg(cfg)

    do_exp(cfg)

if __name__ == '__main__':
    main()
