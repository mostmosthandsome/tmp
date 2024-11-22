# import Libs
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# Set Args
import sys

def main():

    sys.argv = sys.argv[0:1]+["--task=Dancer_leg"]
    # sys.argv = sys.argv[0:1]+["--task=wk4","--headless"]
    sys.argc = len(sys.argv)
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.domain_rand.randomize_init_dof = False
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.env.num_envs = 1200

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _ = env.reset()

    obs_np = np.zeros((500, env.num_envs ,env.num_obs))
    image_np = np.zeros((500, env.num_envs ,env.num_actions))

    start_time = time.time()
    for i in range(500):
        actions = 0.*torch.ones(env.num_envs, env.num_actions, device=env.device)
    #     actions = torch_rand_float(-1, 1, [env.num_envs, env.num_actions], device=env.device)
    #     (obs, privileged_obs, obs_history, image, est_dict, obs_future), rewards, dones, infos = env.step(actions)
        (obs, privileged_obs, obs_history, est_dict, obs_future), rewards, dones, infos = env.step(actions)
        obs_np[i] = obs.cpu().numpy()
    #     print(i, "obs_h, obs:\n", obs_history[0,[0,1,2,3]])
    #     print("privileged_obs:\n", privileged_obs[0,:])
    #     print("body_vel:\n", body_vel[0,:])
    #     print("obs:", obs[:, -4:])
        print("tau:",  torch.where(torch.isnan(env.torques)))
        # if i%100==0:
        #     print(np.round(time.time()-start_time,2), np.round((i+1)*env.dt,2))
    print("Done")

if __name__ == "__main__":
    main()
