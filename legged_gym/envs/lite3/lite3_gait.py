from isaacgym.torch_utils import *
from isaacgym import gymapi

import torch
from torch import Tensor
from typing import Dict

from legged_gym.utils import class_to_dict
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.gait_manager import GaitManager
from legged_gym.envs import LeggedRobot
from legged_gym.envs.lite3.lite3_gait_config import Lite3GaitCfg

class Lite3Gait(LeggedRobot):
    force_reward_weight: Tensor
    speed_reward_weight: Tensor

    def __init__(self, cfg: Lite3GaitCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.gait_manager = GaitManager(cfg=self.cfg.gait,
                                        num_legs=len(self.feet_indices),
                                        num_robots=self.num_envs, dt=self.dt)

        self.dof_dict = {n: self.dof_names.index(n) for n in self.dof_names}
        if hasattr(self.cfg.normalization, "obs_dof_order"):
            obs_dof_order = self.cfg.normalization.obs_dof_order
            self.obs_dof_indices = [self.dof_dict[name] for name in obs_dof_order]
            self.obs_dof_indices = torch.Tensor(data=self.obs_dof_indices).to(self.device).to(torch.long)
        else:
            self.obs_dof_indices = None

        if hasattr(self.cfg.normalization, "action_dof_order"):
            action_dof_order = self.cfg.normalization.action_dof_order
            self.action_dof_indices = [self.dof_dict[name] for name in action_dof_order]
            self.action_dof_indices = torch.Tensor(data=self.action_dof_indices).to(self.device).to(torch.long)
        else:
            self.action_dof_indices = None

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.reward_scales = class_to_dict(self.cfg.rewards.item)

    def step(self, actions):
        self.foot_contact_forces_prev = self.contact_forces[:, self.feet_indices, :]

        obs_tuple, reward, dones, extra = super().step(actions)

        estimation_dict = {}
        for key in self.cfg.env.estimation_terms:
            if "vel" in key.lower():
                estimation_dict[key] = self.body_vel_buf
            elif "hm" in key.lower() or "heightmap" in key.lower():
                estimation_dict[key] = self.heights

        observation_tuple = (obs_tuple[0], obs_tuple[1], obs_tuple[2], estimation_dict, obs_tuple[-1])

        return observation_tuple, reward, dones, extra

    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gait_manager.run()
        frc_weight_np = self.gait_manager.get_frc_penalty_coeff()
        self.force_reward_weight = torch.from_numpy(frc_weight_np).to(torch.float).to(self.device)
        self.speed_reward_weight = 1. - self.force_reward_weight

        super().post_physics_step()

    # check_termination(self) is inherited from base class

    def reset_idx(self, env_ids):
        """ Reset some environments.
        Mostly inherited from the base class. Check LeggedRobot.reset_idx() for more info.
        Args:
            env_ids (Tensor[torch.long]): Index Tensor of environment ids to reset
        """
        if len(env_ids) == 0:
            return
        self.gait_manager.reset(env_ids)

        self.root_pos_trajs[env_ids] = self.root_states[env_ids, :3].unsqueeze(-1).expand(-1, -1,
                                                                                          int(self.cfg.rewards.mean_vel_window / self.dt)).contiguous()

        super().reset_idx(env_ids)

    # self.compute_reward(self) is directly using implementation in base class

    def _observation_vector_assemble(self) -> Tensor:
        if self.obs_dof_indices is not None:
            dof_pos_reorder = (self.dof_pos - self.default_dof_pos)[:, self.obs_dof_indices]
            dof_vel_reorder = self.dof_vel[:, self.obs_dof_indices]
        else:
            dof_pos_reorder = self.dof_pos - self.default_dof_pos
            dof_vel_reorder = self.dof_vel

        return torch.cat((
            torch.from_numpy(self.gait_manager.get_phase_states()).to(self.device).to(torch.float),
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            dof_pos_reorder * self.obs_scales.dof_pos,
            dof_vel_reorder * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

    def compute_observations_f(self, env_ids=None):
        """
        Set observation vector into obs_buf_f. Both "_f" for future.
        Called before reset, in case reset() violates the standard output for decoder.
        :return: None (self.obs_buf_f is for output)
        """
        self.obs_buf_f = self._observation_vector_assemble()

    def compute_observations(self, env_ids=None):
        """
        Computes observations and other buffers.
        """
        self.obs_buf = self._observation_vector_assemble()

        self.body_vel_buf = self.base_lin_vel * self.obs_scales.lin_vel
        self.privileged_obs_buf = torch.cat((self.body_vel_buf, self.obs_buf), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
                min=-1, max=1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.obs_history_buf = torch.cat((self.obs_history_buf[:, self.num_obs:], self.obs_buf), dim=-1)

    # ------------- Callbacks --------------
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero().flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            # in heading mode, ang_vel_yaw(self.commands[:, 2]) is recomputed from heading error
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(self.cfg.commands.heading_ang_vel_kp * wrap_to_pi(self.commands[:, 3] - heading),
                                             min=self.cfg.commands.ranges.ang_vel_yaw[0],
                                             max=self.cfg.commands.ranges.ang_vel_yaw[1])

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        self.measured_foot_heights = self._get_foot_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


        self.root_pos_trajs = torch.cat((self.root_states[:, :3].unsqueeze(-1),
                                         self.root_pos_trajs[..., :-1]), dim=-1)
        power = torch.clip(torch.sum(torch.abs(self.torques * self.dof_vel), dim=1), min=0.01)
        lin_vel_mean = self.root_pos_trajs[:, :2, 0] - self.root_pos_trajs[:, :2, -1]
        window_lengths = torch.clip(self.episode_length_buf * self.dt, self.dt, self.cfg.rewards.mean_vel_window)
        lin_vel_mean = lin_vel_mean / window_lengths.unsqueeze(-1)
        transportVel = self.total_mass * 9.81 * torch.clip(torch.norm(lin_vel_mean[:, :2], dim=1), min=0.01)
        self.cot_vel = power / transportVel

        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero().flatten()
        self._randomize_dof_props(env_ids)

    def _compute_torques(self, actions):
        if self.action_dof_indices is not None:
            action_reorder = torch.zeros(actions.shape, device=self.device, dtype=torch.float)
            for i in range(len(self.cfg.normalization.action_dof_order)):
                name = self.cfg.normalization.action_dof_order[i]
                action_reorder[:, self.dof_names.index(name)] = actions[:, i]
        else:
            action_reorder = actions
        return super()._compute_torques(actions=action_reorder)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_start_index = 11
        noise_list = [torch.zeros(noise_start_index),
                      torch.ones(3) * noise_scales.ang_vel * self.obs_scales.ang_vel,
                      torch.ones(3) * noise_scales.gravity,
                      torch.ones(self.num_dof)*noise_scales.dof_pos * self.obs_scales.dof_pos,
                      torch.ones(self.num_dof)*noise_scales.dof_vel * self.obs_scales.dof_vel,
                      torch.zeros(self.num_actions)]
        leveled_noise_list = [v.to(self.obs_buf.device).to(self.obs_buf.dtype) * noise_level for v in noise_list]
        noise_vec = torch.cat(leveled_noise_list, dim=-1)
        noise_vec = noise_vec.to(self.obs_buf.dtype).to(self.obs_buf.device)
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        Currently doesn't have any extra buffers compared with base class.
        Currently doesn't have any extra buffers compared with base class.
        """
        super()._init_buffers()
        self.force_reward_weight = torch.zeros(self.num_envs, len(self.feet_indices),
                                               dtype=torch.float, device=self.device, requires_grad=False)
        self.speed_reward_weight = torch.ones(self.num_envs, len(self.feet_indices),
                                              dtype=torch.float, device=self.device, requires_grad=False)
        self.root_pos_trajs = self.root_states[:, :3].unsqueeze(-1).expand(-1, -1,
                                                                           int(self.cfg.rewards.mean_vel_window / self.dt)).contiguous()

        self.foot_contact_forces_prev = self.contact_forces[:, self.feet_indices, :]

    def _prepare_reward_function(self):
        super()._prepare_reward_function()

        self.step_reward = {
            name: torch.zeros(self.num_envs,
                              dtype=torch.float,
                              device=self.device,
                              requires_grad=False)
            for name in self.reward_scales.keys()}

        self.bell = lambda x, a, b, t, o: a / (1 + torch.pow(torch.square((x - t)/b), 2*o))
        self.exp = lambda x, a, b, t: a * torch.exp(-torch.square((x - t)/b))
        self.linear = lambda x, a: a * x

        print("reward_names:", self.reward_names)


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            if "order" in self.reward_scales[name].keys():
                rew = self.bell(self.reward_functions[i](), self.reward_scales[name]['coeff'],
                                self.reward_scales[name]['decay'], self.reward_scales[name]['targ'],
                                self.reward_scales[name]['order'])
            elif "decay" in self.reward_scales[name].keys():
                rew = self.exp(self.reward_functions[i](), self.reward_scales[name]['coeff'],
                               self.reward_scales[name]['decay'], self.reward_scales[name]['targ'])
            else:
                rew = self.linear(self.reward_functions[i](), self.reward_scales[name]['coeff'])
            self.rew_buf += rew
            self.step_reward[name] = rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_names:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew



    def _reward_lin_Vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_mean = self.root_pos_trajs[:, :3, 0] - self.root_pos_trajs[:, :3, -1]
        window_lengths = torch.clip(self.episode_length_buf * self.dt, self.dt, self.cfg.rewards.mean_vel_window)
        lin_vel_mean = lin_vel_mean / window_lengths.unsqueeze(-1)
        lin_vel_body = quat_rotate(self.base_quat, lin_vel_mean)

        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        return lin_vel_error

    def _reward_ang_Vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        return ang_vel_error

    def _reward_zVel(self):
        # Tracking of linear velocity commands (xy axes)
        z_vel_error = torch.abs(self.root_states[:, 9])
        return z_vel_error

    def _reward_bTwist(self):
        # Penalize non stationary base orientation vel
        return torch.norm(self.base_ang_vel[:, :2], dim=-1)

    def _reward_bRot(self):
        # Penalize non stationary base orientation vel
        return torch.norm(self.projected_gravity[:, :2], dim=-1)

    def _reward_bHgt(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=-1)
        # base_height = self.root_states[:, 2]
        return base_height - self.cfg.rewards.base_height_target

    def _reward_cotr(self):
        return torch.clip(self.cot_vel - 0.4, min=0.)

    def _reward_eVel(self):
        return torch.sum(self.speed_reward_weight * torch.norm(self.rb_states[:, self.feet_indices, 7:10], dim=-1),
                         dim=-1)

    def _reward_eFrc(self):
        return torch.sum(self.force_reward_weight * torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1),
                         dim=-1)

    def _reward_ipct(self):
        # penalize high contact forces
        tmp1 = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - torch.norm(
            self.foot_contact_forces_prev, dim=-1)
        return torch.norm(tmp1, dim=-1) / self.total_mass / 9.81

    def _reward_cnct_forces(self):
        # penalize high contact forces
        cnct_sum = torch.sum(self.contact_forces[:, self.feet_indices, 2], dim=1)
        return torch.clip(cnct_sum / self.total_mass / 9.81, min=0.)

    def _reward_smth(self):
        return torch.norm(self.torques / self.torque_limits, dim=1)

    def _reward_jVel(self):
        return torch.norm(self.dof_vel, dim=-1) / torch.clip(torch.norm(self.commands[:, :2], dim=-1), min=0.2)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_action(self):
        # Penalize large actions
        return torch.norm(self.actions, dim=1) / torch.clip(torch.norm(self.commands[:, :2], dim=-1), min=0.2)

    def _reward_power(self):
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)
