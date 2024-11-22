from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.WukongIV.wk4_vis_config import Wukong4VisualCfg

import torch
from torch import Tensor
from typing import Tuple, Dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gait_manager import GaitManager

import matplotlib.pyplot as plt


class Wukong4Visual(LeggedRobot):
    def __init__(self, cfg: Wukong4VisualCfg, sim_params, physics_engine, sim_device, headless):
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
        self.sim_params = sim_params
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.num_heights = int(len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y))
        self.gait_manager = GaitManager(self.cfg.gait, self.num_envs, 2, self.dt)

    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)
        self._visualize_perception(self.look_at_env, use_line=False)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.reset_buf[:] = 0
        # step physics and render each frame
        self.render()
        self.foot_contact_forces_prev = self.contact_forces[:, self.feet_indices, :]
        self.torques_prev = self.torques

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.reset_buf |= torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.obs_history_buf = torch.clip(self.obs_history_buf, -clip_obs, clip_obs)
        self.body_vel_buf = torch.clip(self.body_vel_buf, -clip_obs, clip_obs)
        self.obs_buf_f = torch.clip(self.obs_buf_f, -clip_obs, clip_obs)
        self.heights = torch.clip(self.heights, -clip_obs, clip_obs)

        estimation_dict = {}
        for key in self.cfg.env.estimation_terms:
            if "vel" in key.lower():
                estimation_dict[key] = self.body_vel_buf
            elif "hm" in key.lower() or "heightmap" in key.lower():
                estimation_dict[key] = self.heights
            elif "fhm" in key.lower() or "foot_height" in key.lower():
                estimation_dict[key] = self.foot_heightmap_buf
        # print("estimation_dict")
        # for k, d in estimation_dict.items():
        #     print(k, ":", d.shape)
        #     if d.shape[1] < 5:
        #         print(d)
        observation_tuple = (self.obs_buf,
                             self.privileged_obs_buf,
                             self.obs_history,
                             self.depth_image[:, :-1],
                             estimation_dict,
                             self.obs_buf_f)

        return observation_tuple, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # contact force update is done every simulation step forward

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # quat_rotate_inverse: transform base frame to world frame
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.update_gait_singnal()
        self.compute_reward()
        self.compute_observations_f()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations(env_ids)
        # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.before_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.root_pos_trajs = torch.cat((self.root_states[:, :3].unsqueeze(-1),
                                         self.root_pos_trajs[..., :-1]), dim=-1)

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        if not self.headless and self.look_at_env is not None:
            self.set_camera(self.root_states[self.look_at_env, :3] + 1.732, self.root_states[self.look_at_env, :3])

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
        super().reset_idx(env_ids=env_ids)
        self.gait_manager.reset(env_ids)
        self.root_pos_trajs[env_ids] = self.root_states[env_ids, :3].unsqueeze(-1).expand(-1, -1,
                                                                                          int(self.cfg.rewards.mean_vel_window / self.dt)).contiguous()

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
        self.commands[:, :2] = cmd_t[:, :2]

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

    def compute_observations(self, env_ids):
        self.obs_buf = self._observation_vector_assemble()

        self.body_vel_buf = self.base_lin_vel * self.obs_scales.lin_vel
        # print("body_vel：", self.body_vel_buf.shape, ", ", self.body_vel_buf)
        self.privileged_obs_buf = torch.cat((self.body_vel_buf, self.obs_buf), dim=-1)
        if self.cfg.terrain.measure_foot_heights:
            self.foot_heightmap_buf = (self.root_states[:, 2].unsqueeze(1) - self.measured_foot_heights.flatten(
                start_dim=1) - self.cfg.rewards.base_height_target) * self.obs_scales.foot_height
            # print("foot_heightmap：", self.foot_heightmap_buf.shape, ", ", self.foot_heightmap_buf)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.foot_heightmap_buf), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            self.heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights, -1,
                1.) * self.obs_scales.height_measurements
            # self.heights = torch.cat([heights, torch.clip(
            #     self.rb_states[:, self.feet_indices, 2] - self.measured_foot_heights - 0.08, -0.8, 0.8) * 5], dim=-1)
            # print("heights：", self.heights.shape)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], self.obs_buf.unsqueeze(1)), dim=1)
        self.obs_history = torch.flatten(self.obs_history_buf[:, self.obs_stride - 1::self.obs_stride, :], start_dim=-2)
        self._update_visual_observation(env_ids=env_ids)

    def get_observations(self):
        estimation_dict = {}
        for key in self.cfg.env.estimation_terms:
            if "vel" in key.lower():
                estimation_dict[key] = self.body_vel_buf
            elif "hm" in key.lower() or "heightmap" in key.lower():
                estimation_dict[key] = self.heights
            elif "fhm" in key.lower() or "foot_height" in key.lower():
                estimation_dict[key] = self.foot_heightmap_buf
        return self.obs_buf, self.obs_history, self.depth_image[:, :-1], estimation_dict

    def _update_visual_observation(self, env_ids):
        """
        :param env_ids: a list of env indexes, who has just been reset
        :return:
        """
        if self.terrain is None or self.terrain.ray_tracer is None:
            # Skip this method if the ray tracer is not required
            return

        render_camera = self.episode_length_buf % 5 == 0
        render_ids = render_camera.nonzero(as_tuple=False).flatten()
        render_reset_ids = torch.where(render_ids.unsqueeze(1) == env_ids)[0]

        ### Use IsaacGym's built-in rendering
        if self.cfg.terrain.use_isaacgym_rendering:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)               # update graphics in simulation
            # with torch.no_grad():
            self.gym.render_all_camera_sensors(self.sim)   # render all camera sensors
            # allow access to image tensors in GPU (stop and start to make simulation temporarily stop writing, memory leak in simulation)
            self.gym.start_access_image_tensors(self.sim)

            # process all envs(only execute when collective reset at the beginning of Runner)
            if self.reset_start:
                # depth are rendered in simulation, and could be accessed by depth_image_raw
                depth_img = torch.clip(-torch.stack(self.depth_image_raw)[..., self.camera.depth_crop["height"][0]:-self.camera.depth_crop["height"][1],
                                                                               self.camera.depth_crop["width"][0]:-self.camera.depth_crop["width"][1]], 0, 2).unsqueeze(1)
                depth_img = (depth_img - self.cfg.normalization.depth_mean) * self.obs_scales.depth_image
                self.depth_image[:] = depth_img.repeat(1, self.vision_in_channels + self.vision_lag, 1, 1)

                cam_pos_p, cam_rot_r = self.terrain.ray_tracer.update_cam_pos_manual(self.root_states[:, :3], self.root_states[:, 3:7],
                                                                                     self.camera_transform_p, self.camera_transform_r, self.fov)
                self.cam_pos_buf[:] = torch.cat((cam_pos_p, cam_rot_r), dim=-1)
            else:
                if(render_ids.numel()):
                    depth_img = torch.clip(-torch.stack([self.depth_image_raw[i] for i in render_ids])[..., self.camera.depth_crop["height"][0]:-self.camera.depth_crop["height"][1],
                                                                                                            self.camera.depth_crop["width"][0]:-self.camera.depth_crop["width"][1]], 0, 2).unsqueeze(1)
                    depth_img = (depth_img - self.cfg.normalization.depth_mean) * self.obs_scales.depth_image          # 将深度缩放到[-0.5, 0.5]间
                    self.depth_image[render_ids] = torch.cat((self.depth_image[render_ids, 1:, :, :], depth_img), dim=1)
                    self.depth_image[env_ids] = depth_img[render_reset_ids].repeat(1, self.vision_in_channels + self.vision_lag, 1, 1)

                    cam_pos_p, cam_rot_r = self.terrain.ray_tracer.update_cam_pos_manual(self.root_states[render_ids, :3], self.root_states[render_ids, 3:7],
                                                                                         self.camera_transform_p[render_ids], self.camera_transform_r[render_ids], env_ids=render_ids)
                    self.cam_pos_buf[render_ids] = torch.cat((cam_pos_p, cam_rot_r), dim=-1)

            # 禁止访问GPU中的图像向量
            self.gym.end_access_image_tensors(self.sim)

        ### Use Warp's ray tracer for rendering
        else:
            # process all envs(only execute when collective reset at the beginning of Runner)
            if self.reset_start:
                self.depth_image_raw = []
                # Initialize the camera when the environment is just built
                cam_pos_p, cam_rot_r = self.terrain.ray_tracer.update_cam_pos_manual(self.root_states[:, :3], self.root_states[:, 3:7],
                                                                                     self.camera_transform_p, self.camera_transform_r, self.fov)
                self.cam_pos_buf[:] = torch.cat((cam_pos_p, cam_rot_r), dim=-1)

                self.terrain.ray_tracer.render()
                self.depth_image_raw = self.terrain.ray_tracer.get_depth_map().clone()
                # Crop depth image
                cur_env_ids = torch.arange(self.num_envs)
                depth_img = self.depth_image_raw[cur_env_ids, self.camera.depth_crop["height"][0]:-self.camera.depth_crop["height"][1],
                                                              self.camera.depth_crop["width"][0]:-self.camera.depth_crop["width"][1]].unsqueeze(1)
                # Scaled depth image, original expression: depth_img = (depth_img / 2) - 0.5
                depth_img = (depth_img - self.cfg.normalization.depth_mean) * self.obs_scales.depth_image
                self.depth_image[:] = depth_img.repeat(1, self.vision_in_channels + self.vision_lag, 1, 1)
                self.obs_history_buf = self.obs_buf.unsqueeze(1).repeat(1, self.history_buffer_len, 1)
            else:
                self.depth_image_raw = []
                if(render_ids.numel()):
                    # 对所有要渲染的环境
                    # 用reset后的身体位姿和相机相对身体位姿更新warp的相机位姿(root_states[i, 3:7]是[0, 0, 0, 1])
                    # cam_pos_p, cam_rot_r = self.terrain.diffray.update_cam_pos_reset(self.root_states[i, :3], camera_transform_p, camera_transform_r, self.fov[i])
                    cam_pos_p, cam_rot_r = self.terrain.ray_tracer.update_cam_pos_manual(self.root_states[render_ids, :3], self.root_states[render_ids, 3:7],
                                                                                         self.camera_transform_p[render_ids], self.camera_transform_r[render_ids], env_ids=render_ids)
                    self.cam_pos_buf[render_ids] = torch.cat((cam_pos_p, cam_rot_r), dim=-1)

                    self.terrain.ray_tracer.render(env_ids=render_ids)
                    self.depth_image_raw = self.terrain.ray_tracer.get_depth_map().clone()
                    # Crop depth image
                    depth_img = self.depth_image_raw[render_ids, self.camera.depth_crop["height"][0]:-self.camera.depth_crop["height"][1],
                                                                 self.camera.depth_crop["width"][0]:-self.camera.depth_crop["width"][1]].unsqueeze(1)
                    # Scaled depth image, original expression: depth_img = (depth_img / 2) - 0.5
                    depth_img = (depth_img - self.cfg.normalization.depth_mean) * self.obs_scales.depth_image
                    self.depth_image[render_ids] = torch.cat((self.depth_image[render_ids, 1:, :, :], depth_img), dim=1)
                    self.depth_image[env_ids] = depth_img[render_reset_ids].repeat(1, self.vision_in_channels + self.vision_lag, 1, 1)

    def _process_dof_props(self, props, env_id):
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
        # props = super()._process_dof_props(props=props, env_id=env_id)
        return props

    def update_gait_singnal(self):
        self.gait_manager.run(self.commands[:, :3].detach().cpu().numpy())
        self.force_reward_weight = torch.from_numpy(self.gait_manager.get_frc_penalty_coeff()).float().to(self.device)
        self.speed_reward_weight = 1.0 - self.force_reward_weight

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        command_timespan = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % command_timespan == 0).nonzero(as_tuple=False).flatten()
        self.is_halfway_resample[env_ids] = True
        self._resample_commands(env_ids)

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        tracking_mode_name = self.cfg.commands.tracking_mode.lower()
        if tracking_mode_name == "y_position":
            self.commands[:, 2] = torch.clip(wrap_to_pi(torch.atan2(self.env_origins_y - self.root_states[:, 1],
                                                                    torch.ones(self.num_envs,
                                                                               dtype=torch.float,
                                                                               device=self.device) *
                                                                    self.cfg.commands.tracking_strength) - heading),
                                             -1., 1.)
            inplace_turn_ids = ((self.commands[:, 0] == 0) &
                                (self.episode_length_buf >= command_timespan)).nonzero(as_tuple=False).flatten()
            heading_ids = ((self.commands[:, 0] > 0) & (self.root_states[:, 0] > 85)).nonzero(as_tuple=False).flatten()
            self.commands[inplace_turn_ids, 2] = self.commands[inplace_turn_ids, 4]
            self.commands[heading_ids, 2] = torch.clip(wrap_to_pi(self.commands[heading_ids, 3] - heading[heading_ids]),
                                                       min=-1., max=1.)

        elif tracking_mode_name == "heading":
            self.commands[:, 2] = torch.clip(wrap_to_pi((self.commands[:, 3] - heading) *
                                                        self.cfg.commands.tracking_strength),
                                             self.cfg.commands.ranges.ang_vel_yaw[0],
                                             self.cfg.commands.ranges.ang_vel_yaw[1])

        self.base_height = self._get_base_height()
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.terrain.measure_foot_heights:
            self.measured_foot_heights = self._get_foot_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._randomize_dof_props(env_ids)

    def _resample_commands(self, env_ids):

        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.cmd_w[env_ids, :2] = self.commands[env_ids, :2]
        zero_cmd_dec = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) < 0.2
        nonzero_cmd = (torch.norm(self.cmd_w[env_ids, :2], dim=1)) > 0
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            hard_terrain_level = self.terrain_levels[env_ids] > 4
        else:
            hard_terrain_level = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool, requires_grad=False)
        shall_keep_moving = self.is_halfway_resample[env_ids] & nonzero_cmd & hard_terrain_level
        zero_cmd_dec = zero_cmd_dec & (~shall_keep_moving)
        self.is_halfway_resample[:] = False
        self.commands[env_ids, :][zero_cmd_dec, :] = 0

        tracking_mode_name = self.cfg.commands.tracking_mode.lower()
        if tracking_mode_name == "heading" or tracking_mode_name == "y_position":
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)  # target yaw
            if tracking_mode_name == "y_position":
                self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                             self.command_ranges["ang_vel_yaw"][1],
                                                             (len(env_ids), 1),
                                                             device=self.device).squeeze(1)

            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])  # yaw
            self.commands[env_ids, :][zero_cmd_dec, 3] = heading[env_ids][zero_cmd_dec]

            self.cmd_w[env_ids, 0] = (self.commands[env_ids, 0] * torch.cos(self.commands[env_ids, 3])
                                      - self.commands[env_ids, 1] * torch.sin(self.commands[env_ids, 3]))
            self.cmd_w[env_ids, 1] = (self.commands[env_ids, 0] * torch.sin(self.commands[env_ids, 3])
                                      + self.commands[env_ids, 1] * torch.cos(self.commands[env_ids, 3]))
        elif tracking_mode_name == "ang_vel":
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[env_ids, :][zero_cmd_dec, 2] = 0

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.04).unsqueeze(1)
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

        control_type = self.cfg.control.control_type
        if control_type == "P":
            self.joint_pos_target = self.joint_pos_target + self.default_dof_pos
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

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
            This function is filled with customized parameters.
            They might need to be changed if the terrain is different.
        Args:
            env_ids (List[int]): Environment indexes to be reset
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
            self.env_origins_y[env_ids] = self.root_states[env_ids, 1].clone()
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.env_origins_y[env_ids] = self.root_states[env_ids, 1].clone()
            # base velocities
        if self.cfg.domain_rand.randomize_init_state:
            self.root_states[env_ids, 7] = 1.0 * torch.rand(len(env_ids), device=self.device) - 0.5
            self.root_states[env_ids, 8] = 0.6 * torch.rand(len(env_ids), device=self.device) - 0.3
            self.root_states[env_ids, 9] = 0.4 * torch.rand(len(env_ids), device=self.device) - 0.2
            self.root_states[env_ids, 10:13] = torch_rand_float(-0.2, 0.2, (len(env_ids), 3),
                                                                device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0., 0., (len(env_ids), 6),
                                                               device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

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
        gait_signal_num = 4
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

    def _init_buffers(self):
        super()._init_buffers()

        self.lag_buffer = torch.zeros(self.num_envs, self.cfg.domain_rand.lag_timesteps + 1, self.num_actions,
                                      dtype=torch.float, device=self.device, requires_grad=False)
        self.root_pos_trajs = self.root_states[:, :3].unsqueeze(-1).expand(-1, -1,
                                                                           int(self.cfg.rewards.mean_vel_window / self.dt)).contiguous()

        self.foot_contact_forces_prev = self.contact_forces[:, self.feet_indices, :]

        self.torques_prev = self.torques
        self.is_halfway_resample = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.cmd_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.exp_dist_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)

        self.base_height_points = self._init_base_height_points()
        if self.cfg.terrain.measure_foot_heights:
            self.foot_height_points = self._init_foot_height_points()

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
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if self.action_dof_indices is not None:
                for dof_name in self.cfg.control.action_scale.keys():
                    if dof_name in name:
                        self.dof_scale[i] = self.cfg.control.action_scale[dof_name]
        self.dof_scale = self.dof_scale.unsqueeze(0)

        self.obs_history_buf = torch.zeros(self.num_envs, self.history_buffer_len, self.num_obs, device=self.device,
                                           dtype=torch.float)
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, device=self.device, dtype=torch.float)
        self.base_height_buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)

        if self.cfg.terrain.measure_foot_heights:
            self.foot_heightmap_buf = torch.zeros(self.num_envs, self.num_foot_height_points, device=self.device,
                                                  dtype=torch.float)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        super()._create_envs()
        # swap left and right feet indexes
        self.feet_indices = self.feet_indices.flip(dims=(0,))

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_length
        hf_params.transform.p.y = -self.terrain.cfg.border_width
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.T.flatten().astype(np.int16), hf_params)
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
        self.edge_masks = {k: [torch.tensor(v).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device) for v in vs] for k, vs in self.terrain.edge_masks.items()}


    def _parse_cfg(self, cfg):
        if not self.cfg.terrain.depth_camera_on:
            print("[wkVis.parseCfg] Shrinking camera to 1x1 to save VRAM.")
            self.cfg.env.camera.vision_in_channels = 1
            self.cfg.env.camera.depth_width = 1
            self.cfg.env.camera.depth_height = 1
        super()._parse_cfg(self.cfg)

        self.reward_scales = class_to_dict(self.cfg.rewards.item)
        self.obs_stride = self.cfg.env.history_stride
        self.history_buffer_len = self.cfg.env.num_history_frames * self.obs_stride
        self.camera = self.cfg.env.camera


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
        Args: env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
        Raises: NameError: [description]
        Returns: [type]: [description]
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
        points[:, :, 0] += self.terrain.cfg.border_length[0]
        points[:, :, 1] += self.terrain.cfg.border_width[0]
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
        heights = self.height_samples[px, py]
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_base_height(self):
        if self.cfg.terrain.mesh_type == 'plane':
            heights = self.root_states[:, 2]
            return heights.unsqueeze(1)

        points = (self.root_states[:, :3]).clone().unsqueeze(1)
        points[:, :, 0] += self.terrain.cfg.border_length[0]
        points[:, :, 1] += self.terrain.cfg.border_width[0]
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights = self.height_samples[px, py]
        # print("base_height", heights * self.terrain.cfg.vertical_scale)
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

        heights = self.height_samples[px, py]
        return heights.view(self.num_envs, len(self.feet_indices), self.num_foot_height_points).to(
            torch.float) * self.terrain.cfg.vertical_scale

    def _init_depth_map_points(self, env_id):
        sx = 2.0 * torch.arange(self.cfg.env.camera.camera_width, device=self.device, requires_grad=False) / \
             float(self.cfg.env.camera.camera_width) - 1.0
        sy = 2.0 * torch.arange(self.cfg.env.camera.camera_height, device=self.device, requires_grad=False) / \
             float(self.cfg.env.camera.camera_height) - 1.0
        sx = sx[self.cfg.env.camera.depth_crop["width"][0]: -self.cfg.env.camera.depth_crop["width"][1]]  # crop depth image
        sy = sy[self.cfg.env.camera.depth_crop["height"][0]: -self.cfg.env.camera.depth_crop["height"][1]]  # crop depth image
        # sy = sy.flip(0)  # origin in left up in depth image
        sx, sy = torch.meshgrid(sx, sy)

        tan = self.fov[env_id]

        if self.terrain is None or self.terrain.ray_tracer is None:
            aspect = 1.0
        else:
            aspect = self.terrain.ray_tracer.camera.aspect

        self.depth_map_points = torch.stack([sx * tan, sy * tan * aspect, -torch.ones_like(sx)], dim=-1)

    def _visualize_perception(self, env_id, depth_frame=-1,
                              use_line=False, depth_gap=7, **kwarg):
        """
        Arg:
        env_id : id of env needs to visualize, only support a num
        depth_frame: frame of depth image need to visualize, -1 for the last frame
        use_line: decide whether you use line to visualize depth image
        depth_gap: describe for how many depth_image points visualizer draw a point,
            a good choice is a prime number of the num of pixels (e.g. 7, 11)

        HeightMap(R):   self.heights (num_envs, num_height_points)
                            terrain minus target height in base frame.
        DepthImage(B):  self.depth_image (num_envs, num_frames, height, width)
        FootHeight(G):  self.foot_heightmap_buf (num_envs, len(self.feet_indices) * self.num_foot_height_points)

        self.height_points:         xy coordinates of self.heights in base frame
        self.foot_height_points:    xy coordinates of self.foot_heightmap_buf in base frame
                            (need to add the position of leg, i.e. self.rb_states[:, self.feet_indices, :3])

        """
        if env_id is None:
            return

        self.gym.clear_lines(self.viewer)

        sphere_transform = gymapi.Transform()

        # height map (base xy, world z frame)
        if self.heights.shape[1] == self.num_height_points:
            base_pos_transform = gymapi.Transform(p=gymapi.Vec3(*self.root_states[env_id, :3]), r=None)

            height_map_points = self.height_points[env_id, :]
            height_map_points[:, 2] = - (self.heights[env_id, :] / self.obs_scales.height_measurements \
                                         + self.cfg.rewards.base_height_target)
            height_map_points = quat_apply_yaw(self.base_quat[env_id].repeat(1, self.num_height_points),
                                               height_map_points)
            for p in height_map_points:
                sphere_transform.p = gymapi.Vec3(*p)
                sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, sphere_transform,
                                                              color=(1, 0, 0))
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], base_pos_transform)
        else:
            print(f"the shape of self.heights is {self.heights.shape}, will not visualize")

        # depth image, only visualize the last frame
        if list(self.depth_image.shape[2:]) != [self.depth_height, self.depth_width]:
            print(f"the shape of self.depth_image is {self.depth_image.shape}, will not visualize")
        elif not self.cfg.terrain.depth_camera_on:
            pass
        elif hasattr(self, 'depth_map_points') and self.terrain.ray_tracer is not None:
            camera_transform = self.gym.get_camera_transform(self.sim, self.envs[env_id],
                                                             self.camera_handles[env_id])

            depth_image = (self.depth_image[env_id, depth_frame,
                           :] / self.obs_scales.depth_image) + self.cfg.normalization.depth_mean
            depth_image = depth_image.transpose(0, 1)
            depth_map_points = torch.mul(self.depth_map_points,
                                         depth_image.unsqueeze(-1).expand(-1, -1, 3)).flatten(0, 1)

            for i, p in enumerate(depth_map_points):
                if i % depth_gap == 0:
                    sphere_transform.p = gymapi.Vec3(*p)
                    sphere_transform.r = gymapi.Quat(*self.terrain.ray_tracer.camera.rot)
                    sphere_geom = gymutil.WireframeSphereGeometry(0.01, 5, 5, sphere_transform, color=(0, 1, 0))
                    if use_line:
                        gymutil.draw_line(camera_transform.transform_point(gymapi.Vec3(*p)),
                                          camera_transform.transform_point(gymapi.Vec3(0, 0, 0)),
                                          gymapi.Vec3(0, 1, 0),
                                          self.gym, self.viewer, self.envs[env_id])  # draw line
                    else:
                        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id],
                                           camera_transform)  # draw sphere
        else:
            self._init_depth_map_points(env_id)

        # foot_height (base xy, world z frame)
        if self.foot_heightmap_buf.shape[1] == len(self.feet_indices) * self.num_foot_height_points:
            for i in range(len(self.feet_indices)):
                foot_pos = gymapi.Vec3(
                    *self.rb_states[env_id, self.feet_indices, 0:2][i])  # only apply horizontial transform

                foot_transform = gymapi.Transform(p=foot_pos, r=None)

                foot_height_points = self.foot_height_points[env_id].clone()
                foot_height_points[:, 2] = self.foot_heightmap_buf \
                    [env_id, i * self.num_foot_height_points: (i + 1) * self.num_foot_height_points]
                foot_height_points[:, 2] = -(foot_height_points[:, 2] / self.obs_scales.foot_height \
                                             + self.cfg.rewards.base_height_target - self.root_states[env_id, 2])
                foot_height_points = quat_apply_yaw(self.base_quat[env_id].repeat(1, self.num_foot_height_points),
                                                    foot_height_points)
                for p in foot_height_points:
                    sphere_transform.p = gymapi.Vec3(*p)
                    sphere_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, sphere_transform, color=(0, 0, 1))
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], foot_transform)
        else:
            print(f"the shape of self.foot_heightmap_buf is {self.foot_heightmap_buf.shape}, will not visualize")

    def _reward_lin_Vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_mean = self.root_pos_trajs[:, :2, 0] - self.root_pos_trajs[:, :2, -1]
        window_lengths = torch.clip(self.episode_length_buf * self.dt, self.dt, self.cfg.rewards.mean_vel_window)
        lin_vel_mean = lin_vel_mean / window_lengths.unsqueeze(-1)
        lin_vel_error = torch.norm(self.cmd_w[:, :2] - lin_vel_mean, dim=1)
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
        base_height = self.root_states[:, 2] - torch.mean(self.measured_heights, dim=-1)
        # base_height = self.root_states[:, 2]
        return base_height - self.cfg.rewards.base_height_target

    def _reward_cotr(self):
        power = torch.clip(torch.sum(torch.abs(self.torques * self.dof_vel), dim=1), min=0.01)
        lin_vel_mean = self.root_pos_trajs[:, :2, 0] - self.root_pos_trajs[:, :2, -1]
        window_lengths = torch.clip(self.episode_length_buf * self.dt, self.dt, self.cfg.rewards.mean_vel_window)
        lin_vel_mean = lin_vel_mean / window_lengths.unsqueeze(-1)
        transportVel = self.total_mass * 9.81 * torch.clip(torch.norm(lin_vel_mean[:, :2], dim=1), min=0.5)
        self.cot_vel = power / transportVel
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
        return torch.norm(self.dof_vel, dim=-1) / torch.clip(torch.norm(self.commands[:, :2], dim=-1), min=0.2)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf


    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        # print(torch.norm(self.contact_forces[0, self.feet_indices, :2], dim=-1) > 0.5 * torch.abs(self.contact_forces[0, self.feet_indices, 2]),
        #       torch.norm(self.contact_forces[0, self.feet_indices, :2], dim=-1), torch.abs(self.contact_forces[0, self.feet_indices, 2]))
        return (torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                          0.5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)).float()


    def _reward_ftEdge(self):
        # penalize feet close to the terrain edges
        feet_pos_xy = self.rb_states[:, self.feet_indices, :2]
        feet_pos_xy[..., 0] += self.terrain.cfg.border_length[0]
        feet_pos_xy[..., 1] += self.terrain.cfg.border_width[0]
        feet_pos_xy = (feet_pos_xy / self.terrain.cfg.horizontal_scale).round().long()  # [num_envs, 4, 2]
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.edge_masks['x'][0].shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.edge_masks['x'][0].shape[1] - 1)

        self.feet_at_edges = []
        rew_feet_edge = torch.zeros(self.num_envs, device=self.device)
        for i in range(len(self.cfg.terrain.edge_width_threshs)):
            feet_at_edge_x = self.edge_masks['x'][i][feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
            feet_at_edge_y = self.edge_masks['y'][i][feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
            self.feet_at_edges.append(self.contact_filt & (feet_at_edge_x | feet_at_edge_y))
            rew_feet_edge += (self.terrain_levels > 3) * torch.sum(self.feet_at_edges[i].float(), dim=-1) * \
                             self.cfg.terrain.edge_rew_coeffs[i]
        self.rew_feet_edge = rew_feet_edge

        return rew_feet_edge
