from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def test(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 6)
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_Kp_factor = False
    env_cfg.domain_rand.randomize_Kd_factor = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.domain_rand.camera.randomize_camera = False
    train_cfg.runner.resume = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _ = env.get_observations()
    # load policy
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    with torch.inference_mode():
        for i in range(10 * int(env.max_episode_length)):
            # actions = 0. * torch.ones(env.num_envs, env.num_actions, device=env.device)
            actions = torch.zeros((env.num_envs, env_cfg.env.num_actions), device=env.device)
            obs, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    test(args=get_args())
