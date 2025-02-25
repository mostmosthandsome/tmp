import math

import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gait_manager import GaitManager
from legged_gym.envs.WukongIV.wk4_leg_config import Wk4RoughCfg


class WukongLegbase(BaseTask):
    gait_manager: GaitManager = None
    def __init__(self, cfg: Wk4RoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.debug_wp = True
        self.init_done = False
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self._parse_cfg(self.cfg)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        if self.gait_manager is None:
            self.gait_manager = GaitManager(self.cfg.gait, self.num_envs, 2, self.dt)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.look_at_env = None
        if not self.headless:
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_Z, "toggle_viewer_target")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_X, "increase_viewer_target")

    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)
        for evt in self.viewer_events:
            if evt.action == "toggle_viewer_target" and evt.value > 0:
                self._toggle_viewer_target()
            elif evt.action == "increase_viewer_target" and evt.value > 0:
                self._increase_viewer_target()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        self.foot_contact_forces_prev = self.contact_forces[:, self.feet_indices, :]
        self.torques_prev = self.torques

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.obs_history = torch.clip(self.obs_history, -clip_obs, clip_obs)
        self.body_vel_buf = torch.clip(self.body_vel_buf, -clip_obs, clip_obs)
        self.obs_buf_f = torch.clip(self.obs_buf_f, -clip_obs, clip_obs)

        estimation_dict = {}
        for key in self.cfg.env.estimation_terms:
            if "vel" in key.lower():
                estimation_dict[key] = self.body_vel_buf
            elif "hm" in key.lower() or "heightmap" in key.lower():
                estimation_dict[key] = self.foot_heightmap_buf
            elif "hgt" in key.lower() or "height" in key.lower():
                estimation_dict[key] = self.base_height_buf

        observation_tuple = (self.obs_buf, self.privileged_obs_buf, self.obs_history, estimation_dict, self.obs_buf_f)

        return observation_tuple, self.rew_buf, self.reset_buf, self.extras

    def torque_reply(self, torq):
        self.render()
        self.root_states[:, 0:13] = torch.tensor([0, 0, 3.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.torques = torq.to(self.device)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # quat_rotate_inverse: transform base frame to world frame
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.update_gait_signal()
        self.compute_reward()
        self.compute_observations_f()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()
        # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.before_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.root_pos_trajs = torch.cat((self.root_states[:, :3].unsqueeze(-1),
                                         self.root_pos_trajs[..., :-1]), dim=-1)

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        if self.viewer and self.enable_viewer_sync and self.cfg.commands.enable_waypoint and self.debug_wp:
            self._draw_waypoints_vis()

        if not self.headless and self.look_at_env is not None:
            self.set_camera(self.root_states[self.look_at_env, :3] + 1.732, self.root_states[self.look_at_env, :3])

    def update_gait_signal(self):
        self.gait_manager.run(self.commands[:, :3].detach().cpu().numpy())
        self.force_reward_weight = torch.from_numpy(self.gait_manager.get_frc_penalty_coeff()).float().to(self.device)
        self.speed_reward_weight = 1.0 - self.force_reward_weight

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        incline_flag_buf = self.projected_gravity[:,2] > -0.5
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= incline_flag_buf
        self.reset_buf |= self.time_out_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        (obs, privileged_obs, obs_history, estimation, _), _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, obs_history, estimation

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (Tensor[torch.long]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
            self.update_command_range_by_terrain(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self._randomize_dof_props(env_ids)
        # self._resample_waypoint_commands(env_ids)
        self._resample_commands(env_ids)

        self.gait_manager.reset(env_ids)

        # reset buffers
        self.obs_history_buf[env_ids] = 0
        self.obs_history[env_ids] = 0
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.lag_buffer[env_ids] = 0

        self.root_pos_trajs[env_ids] = self.root_states[env_ids, :3].unsqueeze(-1).expand(-1, -1,
                                                                                          int(self.cfg.rewards.mean_vel_window / self.dt)).contiguous()
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids])
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = 0
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

    def _observation_vector_assemble(self) -> Tensor:
        if self.obs_dof_indices is not None:
            dof_pos_reorder = (self.dof_pos - self.default_dof_pos)[:, self.obs_dof_indices]
            dof_vel_reorder = self.dof_vel[:, self.obs_dof_indices]
        else:
            dof_pos_reorder = self.dof_pos - self.default_dof_pos
            dof_vel_reorder = self.dof_vel
        # command should express in base frame
        self.cmd_w[:, 2] = 0
        cmd_t = quat_rotate_inverse(self.base_quat, self.cmd_w)
        # self.commands[:, :2] = cmd_t[:, :2]

        return torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            dof_pos_reorder / self.dof_scale[:, self.obs_dof_indices],
            dof_vel_reorder * self.obs_scales.dof_vel,
            self.actions,
            torch.from_numpy(self.gait_manager.get_phase_states()).float().to(self.device)
        ), dim=-1)

    def compute_observations_f(self):
        self.obs_buf_f = self._observation_vector_assemble()

    def compute_observations(self):
        self.obs_buf = self._observation_vector_assemble()

        self.body_vel_buf = self.base_lin_vel * self.obs_scales.lin_vel
        self.base_height_buf = torch.clip(self.base_height - self.cfg.rewards.base_height_target, -0.3,
                                          0.3) * self.obs_scales.base_height
        self.foot_heightmap_buf = (self.root_states[:, 2].unsqueeze(1) - self.measured_foot_heights.flatten(
            start_dim=1) - self.cfg.rewards.base_height_target) * self.obs_scales.foot_height

        self.privileged_obs_buf = torch.cat(
            (self.body_vel_buf, self.obs_buf, self.base_height_buf, self.foot_heightmap_buf), dim=-1)
        # self.privileged_obs_buf = torch.cat((self.body_vel_buf, self.obs_buf), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights, -1,
                1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], self.obs_buf.unsqueeze(1)), dim=1)
        self.obs_history = torch.flatten(self.obs_history_buf[:, self.obs_stride - 1::self.obs_stride, :], start_dim=-2)

    def get_observations(self):

        estimation_dict = {}
        for key in self.cfg.env.estimation_terms:
            if "vel" in key.lower():
                estimation_dict[key] = self.body_vel_buf
            elif "hm" in key.lower() or "heightmap" in key.lower():
                estimation_dict[key] = self.foot_heightmap_buf
            elif "hgt" in key.lower() or "height" in key.lower():
                estimation_dict[key] = self.base_height_buf
        # return self.obs_buf, self.obs_history, estimation_dict, self.obs_buf_f
        return self.obs_buf, self.obs_history, estimation_dict

    def get_torque(self):
        return self.torques

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs, self.device)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_pos_limits_range = torch.zeros(self.num_dof, dtype=torch.float, device=self.device,
                                                    requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits_range[i] = r
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _randomize_dof_props(self, env_ids):
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand([len(env_ids), self.num_dof], dtype=torch.float,
                                                          device=self.device,
                                                          requires_grad=False) * (
                                                       max_strength - min_strength) + min_strength
        if self.cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand([len(env_ids), self.num_dof], dtype=torch.float,
                                                     device=self.device,
                                                     requires_grad=False) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if self.cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand([len(env_ids), self.num_dof], dtype=torch.float,
                                                     device=self.device,
                                                     requires_grad=False) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_rigid_body_props(self, props, env_id):

        if self.cfg.domain_rand.randomize_base_mass:
            mass_rng = self.cfg.domain_rand.added_mass_range
            com_rng = self.cfg.domain_rand.com_displacement_range
            props[0].mass += np.random.uniform(mass_rng[0], mass_rng[1])
            props[0].com = gymapi.Vec3(np.random.uniform(com_rng[0], com_rng[1]),
                                       np.random.uniform(com_rng[0], com_rng[1]),
                                       np.random.uniform(com_rng[0], com_rng[1]))
        mass = 0.
        for i, p in enumerate(props):
            mass += p.mass
        self.total_mass[env_id] = mass
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self.is_halfway_resample[env_ids] = True
        self._resample_commands(env_ids)
        self._update_vel_commands()
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height = self._get_base_height()
        self.measured_foot_heights = self._get_foot_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._randomize_dof_props(env_ids)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        self.commands[env_ids, 0] = torch_rand_tensor(self.cmd_range[env_ids, 0],
                                                      self.cmd_range[env_ids, 1],
                                                      device=self.device)
        self.commands[env_ids, 1] = torch_rand_tensor(self.cmd_range[env_ids, 2],
                                                      self.cmd_range[env_ids, 3],
                                                      device=self.device)

        zero_cmd_dec = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) < 0.2
        zero_cmd_dec = zero_cmd_dec[env_ids]
        nonzero_cmd = (torch.norm(self.commands[env_ids, :2], dim=1)) > 0
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            hard_terrain_level = self.terrain_levels[env_ids] > 4
        else:
            hard_terrain_level = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool, requires_grad=False)
        shall_keep_moving = self.is_halfway_resample[env_ids] & nonzero_cmd & hard_terrain_level
        shall_stop_index = zero_cmd_dec & (~shall_keep_moving)
        self.is_halfway_resample[:] = False
        if len(shall_stop_index):
            self.commands[env_ids[shall_stop_index], :] = 0

        if self.cfg.commands.heading_command:
            # heading command is independent
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)  # target yaw
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])  # yaw
            if len(shall_stop_index):
                self.commands[env_ids[shall_stop_index], 3] = heading[env_ids[shall_stop_index]]
            self.cmd_w[env_ids, 0] = (self.commands[env_ids, 0] * torch.cos(self.commands[env_ids, 3])
                                      - self.commands[env_ids, 1] * torch.sin(self.commands[env_ids, 3]))
            self.cmd_w[env_ids, 1] = (self.commands[env_ids, 0] * torch.sin(self.commands[env_ids, 3])
                                      + self.commands[env_ids, 1] * torch.cos(self.commands[env_ids, 3]))
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            if len(shall_stop_index):
                self.commands[env_ids[shall_stop_index], 2] = 0

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        self.exp_dist_buf[env_ids] = torch.norm(self.cmd_w[env_ids, :2], dim=1) * self.cfg.commands.resampling_time

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                            requires_grad=False)

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat((self.lag_buffer[:, 1:], actions.clone().unsqueeze(1)), dim=1)
            # self.randomize_lag_indices = torch.clip(self.randomize_lag_indices + torch.randint(-1, 2, (self.num_envs,)), 0, self.cfg.domain_rand.lag_timesteps)
            this_action = self.lag_buffer[torch.arange(self.num_envs), self.randomize_lag_indices]
        else:
            this_action = actions

        if self.action_dof_indices is not None:
            self.joint_pos_target[:, self.action_dof_indices] = this_action * self.dof_scale[:, self.action_dof_indices]
        else:
            self.joint_pos_target[:, :self.num_actions] = this_action * self.cfg.control.action_scale

        self.joint_pos_target = self.joint_pos_target + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) -
                       self.d_gains * self.Kd_factors * self.dof_vel)
        elif control_type == "V":
            torques = (self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_vel) -
                       self.d_gains * self.Kd_factors * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            torques = self.joint_pos_target
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _update_vel_commands(self):
        cfg_cmd = self.cfg.commands
        if cfg_cmd.heading_command:
            # in heading mode, ang_vel_yaw(self.commands[:, 2]) is recomputed from heading error
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(cfg_cmd.tracking_strength * wrap_to_pi(self.commands[:, 3] - heading),
                                             cfg_cmd.ranges.ang_vel_yaw[0],
                                             cfg_cmd.ranges.ang_vel_yaw[1])
        elif cfg_cmd.enable_waypoint:
            lin_vel = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            dist = self.waypoint - self.root_states[:, :2]
            dist_flag = (torch.norm(dist, dim=1) > 0.08).unsqueeze(1)
            lin_vel[:, :2] = self.cfg.commands.dist2lin_vel_scale * dist * dist_flag
            base_vel_cmd = quat_rotate_inverse(self.base_quat, lin_vel)
            theta = torch.atan2(base_vel_cmd[:, 1], base_vel_cmd[:, 0])
            max_vel = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
            max_vel[:,0] = self.command_ranges["lin_vel_x"][1] * torch.cos(theta)
            max_vel[:,1] = self.command_ranges["lin_vel_y"][1] * torch.sin(theta)
            scale = (base_vel_cmd.norm(dim=1) / max_vel.norm(dim=1)).clip(min=1.0).unsqueeze(1)
            base_vel_cmd /= scale
            # print("dist:", dist)
            # print("base_vel_cmd:", base_vel_cmd)
            self.commands[:,:2] = base_vel_cmd[:,:2]

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        zero_cmd_dec = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) < 0.2
        zero_cmd_dec = zero_cmd_dec[env_ids]

        if self.cfg.commands.enable_waypoint:
            self.waypoint[env_ids] = self.root_states[env_ids, :2]
            resample_idx = (~zero_cmd_dec).nonzero(as_tuple=False).flatten()

            if len(resample_idx) > 0:
                self.waypoint[env_ids[resample_idx],:2] = self.env_origins[env_ids[resample_idx], :2]
                displacement = torch_rand_float(2, 8, (len(resample_idx), 1),device=self.device).squeeze(1)
                angle = torch_rand_float(-math.pi, math.pi, (len(resample_idx), 1),device=self.device).squeeze(1)
                self.waypoint[env_ids[resample_idx], 0] += displacement * torch.cos(angle)
                self.waypoint[env_ids[resample_idx], 1] += displacement * torch.sin(angle)
                self.exp_dist_buf[env_ids[resample_idx]] = displacement

            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                             self.command_ranges["heading"][1], (len(env_ids), 1),
                                                             device=self.device).squeeze(1)  # target yaw
                # forward = quat_apply(self.base_quat, self.forward_vec)
                # heading = torch.atan2(forward[:, 1], forward[:, 0])
                # self.commands[env_ids[stay_idx], 3] = heading[env_ids[stay_idx]]
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                             self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                             device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 0] = (torch.rand(len(env_ids), dtype=torch.float, device=self.device,requires_grad=False)
                                         * (self.cmd_range[env_ids, 1] - self.cmd_range[env_ids, 0]) + self.cmd_range[env_ids, 0])
            self.commands[env_ids, 1] = (torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False)
                                         * (self.cmd_range[env_ids, 3] - self.cmd_range[env_ids, 2]) + self.cmd_range[env_ids, 2])


            self.commands[env_ids[zero_cmd_dec], :] = 0

            if self.cfg.commands.heading_command:
                # heading command is independent
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                             self.command_ranges["heading"][1], (len(env_ids), 1),
                                                             device=self.device).squeeze(1)  # target yaw
                forward = quat_apply(self.base_quat, self.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])  # yaw
                self.commands[env_ids[zero_cmd_dec], 3] = heading[env_ids][zero_cmd_dec]

                self.cmd_w[env_ids, :2] = self.commands[env_ids, :2]
                self.cmd_w[env_ids, 0] = self.commands[env_ids, 0] * torch.cos(
                    self.commands[env_ids, 3]) - self.commands[env_ids, 1] * torch.sin(self.commands[env_ids, 3])
                self.cmd_w[env_ids, 1] = self.commands[env_ids, 0] * torch.sin(
                    self.commands[env_ids, 3]) + self.commands[env_ids, 1] * torch.cos(self.commands[env_ids, 3])
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                             self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                             device=self.device).squeeze(1)
                self.commands[env_ids[zero_cmd_dec], 2] = 0

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.08).unsqueeze(1)
            # self.cmd_w[env_ids, :2] *= (torch.norm(self.cmd_w[env_ids, :2], dim=1) > 0.08).unsqueeze(1)
            self.exp_dist_buf[env_ids] = torch.norm(self.cmd_w[env_ids, :2], dim=1) * self.cfg.commands.resampling_time

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.cfg.domain_rand.randomize_init_state:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.6, 1.5, (len(env_ids), self.num_dof),
                                                                            device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1.5, 1.5, (len(env_ids), 2),
                                                              device=self.device)  # xy position within 3m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.robot_orgin[env_ids] = self.root_states[env_ids, :3]

        # base velocities

        if self.cfg.domain_rand.randomize_init_state:
            self.root_states[env_ids, 7] = 1.0 * torch.rand(len(env_ids), device=self.device) - 0.5
            self.root_states[env_ids, 8] = 0.6 * torch.rand(len(env_ids), device=self.device) - 0.3
            self.root_states[env_ids, 9] = 0.4 * torch.rand(len(env_ids), device=self.device) - 0.2
            self.root_states[env_ids, 10:13] = torch_rand_float(-0.2, 0.2, (len(env_ids), 3),
                                                                device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0., 0., (len(env_ids), 6),
                                                               device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = (self.root_states[:, 7:9] + torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                                                device=self.device))  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if self.cfg.terrain.robot_move:
            distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
            # robots that walked far enough progress to harder terains
            # move_up = distance > self.terrain.env_length / 2
            move_up = distance > self.exp_dist_buf[env_ids]
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                               dim=1) * self.max_episode_length_s * 0.5) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
            # Robots that solve the last level are sent to a random one
            self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                       torch.randint_like(self.terrain_levels[env_ids],
                                                                          self.max_terrain_level),
                                                       torch.clip(self.terrain_levels[env_ids],
                                                                  0))  # (the minumum level is zero)
            self.exp_dist_buf[env_ids] = 0.
        else:
            self.terrain_levels[env_ids] = torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_range_by_terrain(self, env_ids):
        """ Implements the game-inspired command ranges.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.init_done:
            # don't change on initial reset
            return
        self.cmd_range[env_ids, ] = self.cmd_max - (self.cmd_max - self.cmd_min[self.terrain_types[env_ids]])/(self.cfg.terrain.num_cols-1) * self.terrain_levels[env_ids].unsqueeze(1)

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands
            If the tracking reward is above 80% of the maximum, expand the range of commands by 0.5
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["bVel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["bVel"]["coeff"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

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
        gait_signal_num = self.gait_manager.num_legs * 2
        noise_list = [torch.ones(3) * noise_scales.ang_vel * self.obs_scales.ang_vel,
                      torch.ones(3) * noise_scales.gravity,
                      torch.zeros(3),  # command
                      # torch.ones(self.num_dof) * noise_scales.dof_pos * self.obs_scales.dof_pos,
                      # torch.ones(self.num_dof) * noise_scales.dof_vel * self.obs_scales.dof_vel,
                      torch.ones(self.num_actions) * noise_scales.dof_pos * self.obs_scales.dof_pos,
                      torch.ones(self.num_actions) * noise_scales.dof_vel * self.obs_scales.dof_vel,
                      torch.zeros(self.num_actions),
                      torch.zeros(gait_signal_num)
                      ]
        leveled_noise_list = [v.to(self.obs_buf.device).to(self.obs_buf.dtype) * noise_level for v in noise_list]
        noise_vec = torch.cat(leveled_noise_list, dim=-1)
        noise_vec = noise_vec.to(self.obs_buf.dtype).to(self.obs_buf.device)
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rb_states_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 通过 view，self.dof_pos 与 self.dof_vel 共享 gym 中的实时 joint state 内部数据 (self.dof_state)
        # 之后的计算，读取中无需另 getState
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.root_pos_trajs = self.root_states[:, :3].unsqueeze(-1).expand(-1, -1,
                                                                           int(self.cfg.rewards.mean_vel_window / self.dt)).contiguous()

        self.randomize_lag_indices = torch.randint(0, self.cfg.domain_rand.lag_timesteps + 1, (self.num_envs,))
        self.lag_buffer = torch.zeros(self.num_envs, self.cfg.domain_rand.lag_timesteps + 1, self.num_actions,
                                      dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        # shape: num_envs, num_bodies, xyz axis

        self.foot_contact_forces_prev = self.contact_forces[:, self.feet_indices, :]
        # shape: num_envs, num_bodies, 13 = 7(pos xyz and quat) + 6(lin_vel and ang_vel)
        self.rb_states = gymtorch.wrap_tensor(rb_states_tensor).view(self.num_envs, -1, 13)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)

        self.torques_prev = self.torques
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.before_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                               requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.waypoint = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)  # x pos, y pos
        self.robot_orgin = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.is_halfway_resample = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.cmd_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.exp_dist_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)

        # lin vel command ranges (Vx adn Vy) vary in different terrain difficulties
        self.cmd_range = torch.tensor([self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
                                            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1]],
                                           dtype=torch.float, device=self.device, requires_grad=False)
        self.cmd_range = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.cmd_range[:, ] = torch.tensor([self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
                                            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1]],
                                           dtype=torch.float, device=self.device, requires_grad=False)
        self.cmd_max = torch.tensor([self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
                                     self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1]],
                                    dtype=torch.float, device=self.device, requires_grad=False)
        self.cmd_min = torch.zeros(self.cfg.terrain.num_cols, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.cmd_min[:int(self.cfg.terrain.num_cols*self.cfg.terrain.terrain_proportions[0]), ] = (
            torch.tensor([-0.8, 1.0, -0.3, 0.3],dtype=torch.float, device=self.device, requires_grad=False))
        self.cmd_min[int(self.cfg.terrain.num_cols*self.cfg.terrain.terrain_proportions[0]):, ] = (
            torch.tensor([-0.4, 0.6, -0.2, 0.2],dtype=torch.float, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity = self.projected_gravity / torch.sum(self.projected_gravity)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.base_height_points = self._init_base_height_points()
        self.foot_height_points = self._init_foot_height_points()
        self.measured_heights = 0
        self.measured_foot_heights = 0
        self.force_reward_weight = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float,
                                               device=self.device, requires_grad=False)
        self.speed_reward_weight = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float,
                                               device=self.device, requires_grad=False)

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

        # joint positions offsets and PD gains
        self.dof_scale = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

            if self.action_dof_indices is not None:
                for dof_name in self.cfg.control.action_scale.keys():
                    if dof_name in name:
                        self.dof_scale[i] = self.cfg.control.action_scale[dof_name]

            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.dof_scale = self.dof_scale.unsqueeze(0)

        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.body_vel_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.obs_history_buf = torch.zeros(self.num_envs, self.history_buffer_len, self.num_obs, device=self.device,
                                           dtype=torch.float)
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, device=self.device, dtype=torch.float)
        self.base_height_buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        self.foot_heightmap_buf = torch.zeros(self.num_envs, self.num_foot_height_points, device=self.device,
                                              dtype=torch.float)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non-zero reward scales in the cfg
        """
        for key in list(self.reward_scales.keys()):
            coeff = self.reward_scales[key]['coeff']
            if coeff == 0:
                self.reward_scales.pop(key)
            # else:
            #     self.reward_scales[key]['coeff'] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, v in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode
        self.step_reward = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

        self.bell = lambda x, a, b, t, o: a / (1 + torch.pow(torch.square((x - t) / b), 2 * o))
        self.exp = lambda x, a, b, t: a * torch.exp(-torch.square((x - t) / b))
        self.linear = lambda x, a: a * x

        print("reward_names:", self.reward_names)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_length[0]
        tm_params.transform.p.y = -self.terrain.cfg.border_width[0]
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        print("self.num_dof:", self.num_dof, "\tself.num_dofs:", self.num_dofs)
        print("body_names:", body_names)
        print('dof_names:', self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.total_mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # feet_names[0], feet_names[1] = feet_names[1], feet_names[0]  # swap left and right
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])
        print("feet_names:", feet_names)
        print("feet_indices:", self.feet_indices)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        print("forbidden_link_names:", termination_contact_names)
        print("forbidden_link_indices:", self.termination_contact_indices)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        # self.dt is control dt
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.item)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.rand_interval = np.ceil(self.cfg.domain_rand.rand_interval_s / self.dt)

        self.obs_stride = self.cfg.env.history_stride
        self.history_buffer_len = self.cfg.env.num_history_frames * self.obs_stride

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_waypoints_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws way points
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(1, 0, 0))
        wp = self.waypoint.cpu().numpy()
        hgt = self.root_states[:,2].cpu().numpy()
        for i in range(self.num_envs):
            if i == self.look_at_env:
                pos = wp[i]
                sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], hgt[i]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_base_height_points(self):
        points = torch.zeros(self.num_envs, 1, 3, device=self.device, requires_grad=False)
        return points

    def _init_foot_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_foot_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_foot_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_foot_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_foot_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # Convert the robot-frame scan points to world frame p_wf = rot_yaw_rf * p_rf + p_robot
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points),
                                    self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # There is a border around the terrain, so need further transformation
        points[..., 0] += self.terrain.cfg.border_length[0]
        points[..., 1] += self.terrain.cfg.border_width[0]
        # Split continuous scan point coordinates into decimation grids (w/o accessing the terrain object dynamically)
        points = (points / self.terrain.cfg.horizontal_scale).long()
        # view() returns a resized copy of the calling tensor. *shape=-1 means maintaining the original shape
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        # self.height_samples is a static height gridmap of the whole terrain.
        # It is generated when Terrain() instance is created.
        # The XY resolution of self.height_samples is self.terrain.cfg.horizontal_scale
        # Z resolution is self.terrain.cfg.vertical_scale
        # Subsequent table-checking involves 1 grid forward so the clip range should be shape-2
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        # Table-checking to get the minimal height among the three clocks
        # Instead of directly taking the height under current grid,
        # this minimal operation is to prevent get height of stairs when robot is not stepping onto it.
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # Convert the output values back to meters
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_base_height(self):
        if self.cfg.terrain.mesh_type == 'plane':
            heights = self.root_states[:, 2]
            return heights.unsqueeze(1)

        points = (self.root_states[:, :3]).clone().unsqueeze(1)
        points[..., 0] += self.terrain.cfg.border_length[0]
        points[..., 1] += self.terrain.cfg.border_width[0]
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights = self.height_samples[px, py]
        return self.root_states[:, 2].unsqueeze(1) - heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_foot_heights(self):
        if self.cfg.terrain.mesh_type == 'plane':
            heights = self.rb_states[:, self.feet_indices, 2].repeat_interleave(self.num_foot_height_points, dim=1)
            return heights

        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_foot_height_points),
                                self.foot_height_points).unsqueeze(1) + self.rb_states[:, self.feet_indices,
                                                                        :3].unsqueeze(2)
        points[..., 0] += self.terrain.cfg.border_length[0]
        points[..., 1] += self.terrain.cfg.border_width[0]
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, :, 0].view(-1)
        py = points[:, :, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.max(heights1, heights2)
        heights = torch.max(heights, heights3)
        return heights.view(self.num_envs, len(self.feet_indices), self.num_foot_height_points).to(
            torch.float) * self.terrain.cfg.vertical_scale
        # return torch.mean(
        #     heights.view(self.num_envs, len(self.feet_indices), self.num_foot_height_points).to(torch.float),
        #     dim=2) * self.terrain.cfg.vertical_scale

    # ------------ reward functions----------------
    def _reward_lin_Vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_mean = self.root_pos_trajs[:, :2, 0] - self.root_pos_trajs[:, :2, -1]
        # window_lengths = torch.clip(self.episode_length_buf * self.dt, self.dt, self.cfg.rewards.mean_vel_window)
        # lin_vel_mean = lin_vel_mean / window_lengths.unsqueeze(-1)
        # lin_vel_error = torch.norm(self.cmd_w[:, :2] - lin_vel_mean, dim=1)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return lin_vel_error

    def _reward_way_point(self):
        # Go to waypoint
        dist = (self.root_states[:,:2] - self.waypoint).norm(dim=1)
        dist *= dist>0.08
        return 1.-0.25*dist

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
        power = torch.clip(torch.sum(torch.abs(self.torques[:, self.action_dof_indices] * self.dof_vel[:, self.action_dof_indices]), dim=1), min=0.01)
        lin_vel_mean = self.root_pos_trajs[:, :2, 0] - self.root_pos_trajs[:, :2, -1]
        window_lengths = torch.clip(self.episode_length_buf * self.dt, self.dt, self.cfg.rewards.mean_vel_window)
        lin_vel_mean = lin_vel_mean / window_lengths.unsqueeze(-1)
        transportVel = self.total_mass * 9.81 * torch.clip(torch.norm(lin_vel_mean[:, :2], dim=1), min=0.5)
        self.cot_vel = power / transportVel
        return torch.clip(self.cot_vel - 1.6, min=0.)

    def _reward_eVel(self):
        return torch.sum(self.speed_reward_weight * torch.norm(self.rb_states[:, self.feet_indices, 7:10], dim=-1),
                         dim=-1)

    def _reward_eFrc(self):
        robot_gravity = (self.cfg.sim.gravity[2] * self.total_mass).unsqueeze(1)
        reaction_frc = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        return torch.sum(self.force_reward_weight * reaction_frc / robot_gravity, dim=-1)

    def _reward_ipct(self):
        # penalize high contact forces
        robot_gravity = self.cfg.sim.gravity[2] * self.total_mass
        grf_diff = torch.norm(self.contact_forces[:, self.feet_indices, :] - self.foot_contact_forces_prev, dim=-1)
        return torch.norm(grf_diff, dim=-1) / robot_gravity

    def _reward_cnct_forces(self):
        # penalize high contact forces
        robot_gravity = self.cfg.sim.gravity[2] * self.total_mass
        cnct_sum = torch.sum(self.contact_forces[:, self.feet_indices, 2], dim=1)
        return torch.clip(cnct_sum / robot_gravity, min=0.)

    def _reward_smth(self):
        return torch.norm(self.torques[:, self.action_dof_indices] / self.torque_limits[self.action_dof_indices], dim=1)

    def _reward_jPos(self):
        # Penalize dof positions too. close to the limit
        out_of_limits = -1 * (
                    self.dof_pos[:, self.action_dof_indices] - self.dof_pos_limits[self.action_dof_indices, 0]).clip(
            max=0.)  # lower limit
        out_of_limits += (
                    self.dof_pos[:, self.action_dof_indices] - self.dof_pos_limits[self.action_dof_indices, 1]).clip(
            min=0.)
        out_of_limits = out_of_limits / self.dof_pos_limits_range[self.action_dof_indices]
        return torch.sum(out_of_limits, dim=1)

    def _reward_jVel(self):
        return torch.norm(self.dof_vel[:, self.action_dof_indices], dim=-1) / torch.clip(torch.norm(self.commands[:, :2], dim=-1), min=0.2)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_action(self):
        # Penalize large actions
        return torch.norm(self.actions, dim=1) / torch.clip(torch.norm(self.commands[:, :2], dim=-1), min=0.2)

    def _reward_power(self):
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _toggle_viewer_target(self):
        if self.look_at_env is None:
            self.look_at_env = 0
        else:
            self.look_at_env = None

    def _increase_viewer_target(self):
        if isinstance(self.look_at_env, int):
            self.look_at_env += 1
            if self.look_at_env >= self.num_envs:
                self.look_at_env = 0
            print("Looking at: ", self.look_at_env)
