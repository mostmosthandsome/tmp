# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import warp as wp
import math

import numpy as np
from numpy.random import choice
from scipy import interpolate
import torch
from scipy.ndimage import binary_dilation
from matplotlib import pyplot as plt

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

wp.config.verify_fp = True
wp.init()


@wp.struct
class RenderMesh:
    """Mesh to be ray traced.
    Assumes a triangle mesh as input.
    Per-vertex normals are computed with compute_vertex_normals()
    """

    id: wp.uint64
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    pos: wp.array(dtype=wp.vec3)
    rot: wp.array(dtype=wp.quat)


@wp.struct
class Camera:
    """Basic camera for ray tracing"""

    horizontal: float
    vertical: float
    aspect: float
    tan: wp.array(dtype=float)
    pos: wp.array(dtype=wp.vec3)        # camera pos in world frame
    rot: wp.array(dtype=wp.quat)        # camera rot in world frame


class Example:
    """A basic differentiable ray tracer

    Non-differentiable variables:
    camera.horizontal: camera horizontal aperture size
    camera.vertical: camera vertical aperture size
    camera.aspect: camera aspect ratio
    camera.pos: camera displacement
    camera.rot: camera rotation (quaternion)
    pix_width: final image width in pixels
    pix_height: final image height in pixels
    render_mesh.indices: mesh vertex indices

    Differentiable variables:
    render_mesh.pos: parent transform displacement
    render_mesh.quat: parent transform rotation (quaternion)
    render_mesh.vertices: mesh vertex positions

    Note that: You are not supposed to change the name of this class!!!
    It should be kept as 'Example'.
    Otherwise, the render() method will raise an exception about not finding the kernel method.
    """

    def __init__(self, cfg: LeggedRobotCfg.terrain, num_envs, points_, indices_, device):
        self.device = device
        if cfg.num_cols > 4: self.num_envs_max = 2048
        else: self.num_envs_max = 32    # onlt test locally
        self.num_envs = num_envs

        cam_pos = wp.array([wp.vec3(37, 37, 1) for _ in range(self.num_envs)], dtype=wp.vec3, device=self.device)
        cam_rot = wp.array([wp.quat(0.707, 0.0, 0.0, 0.707) for _ in range(self.num_envs)], dtype=wp.quat, device=self.device)
        cam_tan = wp.array([np.tan(np.radians(87 / 2)) for _ in range(self.num_envs)], dtype=float, device=self.device)
        self.cam_pos = torch.tensor([self.num_envs, 3], device=self.device, requires_grad=False)
        self.cam_rot = torch.tensor([self.num_envs, 4], device=self.device, requires_grad=False)

        horizontal_aperture = 106  # Realsense resolution
        vertical_aperture = 60  # Realsense resolution
        aspect = vertical_aperture / horizontal_aperture

        self.width = 106
        self.height = int(aspect * self.width)
        self.num_pixels = self.width * self.height

        self.border_length_ = cfg.border_length
        self.border_width_ = cfg.border_width

        points = points_
        indices = indices_

        with wp.ScopedDevice(device=self.device):
            # construct RenderMesh
            self.render_mesh = RenderMesh()
            self.mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, requires_grad=False), indices=wp.array(indices, dtype=int))
            self.render_mesh.id = self.mesh.id
            self.render_mesh.vertices = self.mesh.points
            self.render_mesh.indices = self.mesh.indices
            self.render_mesh.pos = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
            self.render_mesh.rot = wp.array(np.array([0.0, 0.0, 0.0, 1.0]), dtype=wp.quat, requires_grad=False)

            # construct camera
            self.camera = Camera()
            self.camera.horizontal = horizontal_aperture
            self.camera.vertical = vertical_aperture
            self.camera.aspect = aspect
            self.camera.tan = cam_tan
            self.camera.pos = cam_pos
            self.camera.rot = cam_rot

            self.depth = wp.zeros((self.num_envs_max, self.num_pixels), dtype=float, requires_grad=False)

    def update_cam_pos(self, cam_pos, cam_rot, cam_tan):
        """
        update cam pos and rot in world frame
        """
        self.camera.pos = wp.vec3(cam_pos.x + self.border_length_, cam_pos.y + self.border_length_, cam_pos.z)
        self.camera.rot = wp.quat(cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w)
        self.camera.tan = cam_tan

    def quat_multiply(self, q1, q2):
        w = q1[:, 3] * q2[:, 3] - q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2]
        x = q1[:, 3] * q2[:, 0] + q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1]
        y = q1[:, 3] * q2[:, 1] - q1[:, 0] * q2[:, 2] + q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0]
        z = q1[:, 3] * q2[:, 2] + q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3]
        quat = torch.stack([x, y, z, w], dim=1)
        return quat

    def quat_rotate(self, q, v):
        # q: [num_envs, 4], v: [num_envs, 3]
        qvec = q[:, :3]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return v + 2 * (q[:, 3].unsqueeze(1) * uv + uuv)

    def update_cam_pos_manual(self, env_pos, env_rot, cam_pos, cam_rot, cam_tan=None, env_ids=None):
        """
        update cam pos and rot in world frame by manually calculating
        """
        env_pos_ = env_pos.clone()
        env_pos_[:, 0] += self.border_length_[0]
        env_pos_[:, 1] += self.border_width_[0]
        pos = self.quat_rotate(env_rot.clone(), cam_pos.clone()) + env_pos_
        rot = self.quat_multiply(env_rot.clone(), cam_rot.clone())    # use quaternion multiplication to rotate the camera
        if env_ids is not None:
            self.cam_pos[env_ids] = pos
            self.cam_rot[env_ids] = rot
            self.camera.pos = wp.from_torch(self.cam_pos, dtype=wp.vec3, requires_grad=False)
            self.camera.rot = wp.from_torch(self.cam_rot, dtype=wp.quat, requires_grad=False)
        else:
            self.cam_pos = pos
            self.cam_rot = rot
            self.camera.pos = wp.from_torch(self.cam_pos, dtype=wp.vec3, requires_grad=False)
            self.camera.rot = wp.from_torch(self.cam_rot, dtype=wp.quat, requires_grad=False)
            self.camera.tan = wp.array(cam_tan, dtype=float, device=self.device)

        pos_ = pos.clone()
        pos_[:, 0] -= self.border_length_[0]
        pos_[:, 1] -= self.border_width_[0]
        return pos_, rot

    def update(self):
        pass

    def render(self, env_ids=None):
        with wp.ScopedDevice(self.device):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
            # call launch to run kernel function, the result will be stored in self.depth
            wp.launch(kernel=Example.draw_kernel,
                      dim=self.num_pixels * len(env_ids),    # num of threads
                      inputs=[self.render_mesh,
                              self.camera,
                              self.width,
                              self.height,
                              self.depth,
                              wp.from_torch(env_ids.int(), dtype=wp.int32, requires_grad=False),
                      ]
            )

    @wp.kernel
    def draw_kernel(mesh: RenderMesh,
                    camera: Camera,
                    rays_width: int,
                    rays_height: int,
                    depth: wp.array(shape=(2048, 106*60), dtype=float),
                    env_ids: wp.array(dtype=wp.int32),
                    ):
        # check id for each thread
        tid = wp.tid()
        env_id = tid // (rays_width * rays_height)
        tid_local = tid % (rays_width * rays_height)
        env_id = env_ids[env_id]

        # compute coordinate of current pixel
        x = tid_local % rays_width
        y = rays_height - tid_local // rays_width

        # compute standard coordinate
        sx = 2.0 * float(x) / float(rays_width) - 1.0
        sy = 2.0 * float(y) / float(rays_height) - 1.0

        # compute view ray in world space
        ro_world = camera.pos[env_id]
        rd_world = wp.normalize(wp.quat_rotate(camera.rot[env_id], wp.vec3(sx * camera.tan[env_id], sy * camera.tan[env_id] * camera.aspect, -1.0)))

        # angles between view ray and camera axis
        ry = math.atan(sy * camera.tan[env_id] * camera.aspect)
        rx = math.atan(sx * camera.tan[env_id])

        # compute view ray in mesh space
        inv = wp.transform_inverse(wp.transform(mesh.pos[0], mesh.rot[0]))
        ro = wp.transform_point(inv, ro_world)   # ray origin
        rd = wp.transform_vector(inv, rd_world)  # ray directions

        t = float(0.0)
        ur = float(0.0)
        vr = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)

        # compute nearest intersection of ray with mesh
        if wp.mesh_query_ray(mesh.id, ro, rd, 2.1, t, ur, vr, sign, n, f):
            # distance to the intersection point(t is the distance to camera center, dis_ is the distance to the intersection point)
            dis_ = t * math.cos(ry) * math.cos(rx)
        else:
            dis_ = 2.

        if dis_ >= 2. or dis_ <= 0.000001:
            depth[env_id][tid_local] = 2.
        elif dis_ <= 0.2:
            depth[env_id][tid_local] = 0.2
        else:
            depth[env_id][tid_local] = dis_

    def get_depth_map(self):
        depth_map_ = wp.torch.to_torch(self.depth.reshape((self.num_envs_max, self.height, self.width)))
        return depth_map_


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0, move_y != 0


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots, device) -> None:
        self.device = device
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        if len(cfg.terrain_proportions) < 9:
            cfg.terrain_proportions = cfg.terrain_proportions + [0] * (9-len(cfg.terrain_proportions))
        cfg.terrain_proportions = np.array(cfg.terrain_proportions)
        if np.sum(cfg.terrain_proportions) > 1:
            cfg.terrain_proportions /= np.sum(cfg.terrain_proportions)
        elif np.sum(cfg.terrain_proportions) < 1:
            cfg.terrain_proportions[-1] = np.sum(cfg.terrain_proportions[:-1])
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border_length = [int(l / self.cfg.horizontal_scale) for l in self.cfg.border_length]
        self.border_width = [int(w / self.cfg.horizontal_scale) for w in self.cfg.border_width]
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + sum(self.border_width)
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + sum(self.border_length)
        self.ftedge_rew_types = cfg.ftedge_rew_types
        if self.ftedge_rew_types is not None:
            self.in_ftedge_rew_types = np.zeros(cfg.num_cols, dtype=bool)

        # rough border
        if self.cfg.border_type == 'rough':
            min_height = int(self.cfg.heightfield_range[0] / self.cfg.vertical_scale)
            max_height = int(self.cfg.heightfield_range[1] / self.cfg.vertical_scale)
            step = int(self.cfg.heightfield_resolution / self.cfg.vertical_scale)
            heights_range = np.arange(min_height, max_height + step, step)
            self.height_field_raw = np.random.choice(heights_range, (self.tot_rows, self.tot_cols)).astype(np.int16)
        elif self.cfg.border_type == 'flat':
            self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # create the terrain
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        elif cfg.arranged:
            self.arranged_terrain(difficulty=cfg.terrain_difficulty)
        else:
            self.randomized_terrain(difficulty=cfg.terrain_difficulty)

        self.heightsamples = self.height_field_raw
        # if self.type == "trimesh" or self.type == "heightfield":
        self.vertices, self.triangles, self.terrain_edge_mask_x, self.terrain_edge_mask_y = \
                                                        convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                       self.cfg.horizontal_scale,
                                                                                       self.cfg.vertical_scale,
                                                                                       self.cfg.slope_treshold)
        points = self.vertices                        # vertices of mesh
        indices = self.triangles.flatten(order='C')   # indices of triangles
        self.ray_tracer = Example(cfg, self.num_robots, points, indices, self.device) if cfg.depth_camera_on else None


        if self.ftedge_rew_types is not None:
            # get the terrain edge mask
            self.edge_masks = {'x': [], 'y': []}
            complete_mask_x = np.zeros_like(self.terrain_edge_mask_x, dtype=bool)
            complete_mask_y = np.zeros_like(self.terrain_edge_mask_y, dtype=bool)

            ftedge_rew_mask = self.get_ftedge_rew_mask()

            if self.cfg.visualize_edge_masks: plt.figure(); i = 1
            for edge_width_thresh in self.cfg.edge_width_threshs:
                half_edge_width = int(edge_width_thresh / self.cfg.horizontal_scale)  # half width of the edge area (e.g. 5cm)
                # mask of far edge area
                structure_x = np.ones((half_edge_width * 2 + 1, 1))
                structure_y = np.ones((1, half_edge_width * 2 + 1))
                dilated_mask_x = binary_dilation(self.terrain_edge_mask_x, structure=structure_x)  # djlation along x-axis
                dilated_mask_y = binary_dilation(self.terrain_edge_mask_y, structure=structure_y)  # dilation along y-axis

                # subtract previous masks to get exclusive regions
                exclusive_mask_x = np.logical_and(dilated_mask_x, np.logical_not(complete_mask_x))
                exclusive_mask_y = np.logical_and(dilated_mask_y, np.logical_not(complete_mask_y))
                # update complete masks
                complete_mask_x = np.logical_or(complete_mask_x, dilated_mask_x)
                complete_mask_y = np.logical_or(complete_mask_y, dilated_mask_y)

                exclusive_mask_x = np.logical_and(exclusive_mask_x, ftedge_rew_mask)
                exclusive_mask_y = np.logical_and(exclusive_mask_y, ftedge_rew_mask)

                self.edge_masks['x'].append(exclusive_mask_x)
                self.edge_masks['y'].append(exclusive_mask_y)

                if self.cfg.visualize_edge_masks:
                    plt.subplot(1, len(self.cfg.edge_width_threshs), i)
                    plt.imshow(np.logical_or(exclusive_mask_x, exclusive_mask_y))
                    i += 1

            if self.cfg.visualize_edge_masks: plt.show()

    def get_ftedge_rew_mask(self):
        """ get the mask of the types of terrain needing foot edge reward
        """
        mask = np.zeros_like(self.terrain_edge_mask_x, dtype=bool)
        for i in range(len(self.in_ftedge_rew_types)):
            if self.in_ftedge_rew_types[i]:
                mask_x1 = self.border_length[0] + i * self.width_per_env_pixels
                mask_x2 = self.border_length[0] + (i + 1) * self.width_per_env_pixels
                mask_y1 = self.border_width[0]
                mask_y2 = self.border_width[0] + self.cfg.num_rows * self.length_per_env_pixels
                mask[mask_y1:mask_y2, mask_x1:mask_x2] = True
        return mask

    def randomized_terrain(self, difficulty=0.9):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            terrain, terrain_type = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def arranged_terrain(self, difficulty=0.9):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = j * 1.0 / self.cfg.num_cols
            terrain, terrain_type = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain, terrain_type = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

                # judge if subterrain is the types needing foot edge reward
                if self.ftedge_rew_types is not None:
                    self.in_ftedge_rew_types[j] = terrain_type in self.ftedge_rew_types

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.vertical_scale,
                                               horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.length_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        platform_size = 3.
        depth = 0.7

        # 0. smooth slope
        if choice < self.proportions[0]:
            terrain_type = 0
            slope = difficulty * 0.45
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)
        # 1. rough slope (up and down)
        elif choice < self.proportions[1]:
            terrain_type = 1
            slope = difficulty * 0.45
            if choice < self.proportions[1] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfield_range[0] + self.cfg.rough_slope_range[0],
                                                 max_height=self.cfg.heightfield_range[1] + self.cfg.rough_slope_range[1],
                                                 step=self.cfg.heightfield_resolution,
                                                 downsampled_scale=0.2)
        # 2. stairs (up), 3. stairs (down)
        elif choice < self.proportions[3]:
            terrain_type = 3
            # hard
            # step_height = 0.05 + 0.12 * difficulty
            # step_length = [0.21, 0.41]
            # easy
            step_height = 0.05 + 0.1 * difficulty
            step_length = [0.31, 0.41]

            if choice > self.proportions[2]:
                terrain_type = 2
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=np.random.uniform(step_length[0], step_length[1]),
                                                 step_height=step_height, platform_size=platform_size)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfield_range[0],
                                                 max_height=self.cfg.heightfield_range[1],
                                                 step=self.cfg.heightfield_resolution,
                                                 downsampled_scale=0.1)
        # 4. discrete obstacles
        elif choice < self.proportions[4]:
            terrain_type = 4
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            discrete_obstacles_height = 0.05 + difficulty * 0.1
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=platform_size)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfield_range[0],
                                                 max_height=self.cfg.heightfield_range[1],
                                                 step=self.cfg.heightfield_resolution,
                                                 downsampled_scale=0.1)
        # 5. stepping stones
        elif choice < self.proportions[5]:
            terrain_type = 5
            stepping_stones_size = 1.5
            stepping_stones_distance = 0.5 * (1.05 - difficulty)

            terrain_utils.stepping_stones_terrain(terrain,
                                                  stone_size=stepping_stones_size,
                                                  stone_distance=stepping_stones_distance,
                                                  max_height=0.,
                                                  platform_size=platform_size)
        # 6. gap
        elif choice < self.proportions[6]:
            terrain_type = 6
            gap_size = 0.15 + 0.45 * difficulty
            # gap_terrain(terrain, gap_size=gap_size)
            gap_terrain(terrain, difficulty, depth=depth, max_height=0., platform_size=platform_size)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfield_range[0],
                                                 max_height=self.cfg.heightfield_range[1],
                                                 step=self.cfg.heightfield_resolution,
                                                 downsampled_scale=0.1)
        # 7. pit
        elif choice < self.proportions[7]:
            terrain_type = 7
            pit_depth = 0.2 + 0.15 * difficulty
            pyramid_pit_terrain(terrain, depth=depth)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfield_range[0],
                                                 max_height=self.cfg.heightfield_range[1],
                                                 step=self.cfg.heightfield_resolution,
                                                 downsampled_scale=0.1)
        # 8. blance beam
        elif choice < self.proportions[8]:
            terrain_type = 8
            bridge_terrain(terrain, difficulty, depth=depth, max_height=0., max_height_bridge=0.03)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfield_range[0],
                                                 max_height=self.cfg.heightfield_range[1],
                                                 step=self.cfg.heightfield_resolution,
                                                 downsampled_scale=0.1)
        # 9. rough flat
        else:
            terrain_type = 9
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfield_range[0],
                                                 max_height=self.cfg.heightfield_range[1],
                                                 step=self.cfg.heightfield_resolution,
                                                 downsampled_scale=0.1)
        # terrain.transpose()
        return terrain, terrain_type

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border_length[0] + i * self.length_per_env_pixels
        end_x = self.border_length[0] + (i + 1) * self.length_per_env_pixels
        start_y = self.border_width[0] + j * self.width_per_env_pixels
        end_y = self.border_width[0] + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 0.5) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 0.5) / terrain.horizontal_scale)
        # print("region: ",x1, x2, y1, y2)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def pyramid_gap_terrain(terrain, difficulty, depth=0.4, max_height=0., platform_size=3.):
    gap_width_ = 0.15 + min(0.35 * difficulty, 0.35)     # 0.15, 0.45, 0.45
    gap_width = int(gap_width_ / terrain.horizontal_scale)
    depth_range_radius = int((0.05 + min(difficulty * 0.1, 0.1)) / terrain.vertical_scale)
    depth_mid = -int(depth / terrain.vertical_scale)
    depth_range = np.arange(depth_mid - depth_range_radius, depth_mid + depth_range_radius, step=1)
    gap_x_ = [1.1, 2.6]
    gap_x = [int(gap_x_[i] / terrain.horizontal_scale) for i in range(len(gap_x_))]
    max_height = int(max_height / terrain.vertical_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    terrain.height_field_raw[:, :] = np.random.choice(height_range)
    for i in range(len(gap_x)):
        center_x1_ = center_x + gap_x[i] + gap_width // 2
        center_y1_ = center_y + gap_x[i] + gap_width // 2
        center_x2_ = center_x - gap_x[i] - gap_width // 2
        center_y2_ = center_y - gap_x[i] - gap_width // 2
        x11 = center_x1_ - gap_width // 2; x12 = x11 + gap_width
        x21 = center_x2_ + gap_width // 2; x22 = x21 - gap_width
        y11 = center_y1_ - gap_width // 2; y12 = y11 + gap_width
        y21 = center_y2_ + gap_width // 2; y22 = y21 - gap_width
        depth_ = np.random.choice(depth_range)
        terrain.height_field_raw[x11: x12, y22: y12] = depth_
        terrain.height_field_raw[x22: x21, y22: y12] = depth_
        terrain.height_field_raw[x22: x12, y11: y12] = depth_
        terrain.height_field_raw[x22: x12, y22: y21] = depth_

    return terrain

def gap_terrain(terrain, difficulty, depth=0.4, max_height=0., platform_size=3.):
    gap_width_ = 0.15 + min(0.35 * difficulty, 0.35)
    gap_width = int(gap_width_ / terrain.horizontal_scale)
    depth_range_radius = int((0.05 + min(difficulty * 0.1, 0.1)) / terrain.vertical_scale)
    depth_mid = -int(depth / terrain.vertical_scale)
    depth_range = np.arange(depth_mid - depth_range_radius, depth_mid + depth_range_radius, step=1)
    stage_dist_ = 1.5     # distance between stages's center
    stage_x_num = 2
    stage_size = int((stage_dist_ - gap_width_) / terrain.horizontal_scale)
    platform_size = int((2. - gap_width_) / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)
    rand_y_coeff = max(0, min(difficulty * 0.4, 0.4))
    rand_y_flag = True

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    terrain.height_field_raw[:, :] = np.random.choice(depth_range)
    for i in range(stage_x_num):
        for d in [-1, 1]:
            center_x_i = center_x + d * (int((1.+stage_dist_/2) / terrain.horizontal_scale) + i*int(stage_dist_ / terrain.horizontal_scale))
            x1 = center_x_i - stage_size//2
            x2 = center_x_i + stage_size//2
            if rand_y_flag: start_y1 = center_y + np.random.randint(-((gap_width)*rand_y_coeff), max(((gap_width)*rand_y_coeff),1))
            else: start_y1 = center_y
            start_y2 = start_y1 + int(1. / terrain.horizontal_scale)
            start_y3 = start_y1 - int(1. / terrain.horizontal_scale)
            y1 = start_y2 + gap_width//4; y1 = y1 if y1 <= terrain.width else terrain.width
            y2 = start_y2 - gap_width//4
            y3 = start_y3 + gap_width//4
            y4 = start_y3 - gap_width//4; y4 = y4 if y4 >= 0 else 0
            terrain.height_field_raw[x1:x2, y1:] = np.random.choice(height_range)
            terrain.height_field_raw[x1:x2, y3:y2] = np.random.choice(height_range)
            terrain.height_field_raw[x1:x2, :y4] = np.random.choice(height_range)

    # center platform
    x1 = center_x - platform_size//2
    x2 = center_x + platform_size//2
    y1 = center_y - int(1.4*platform_size)//2
    y2 = center_y + int(1.4*platform_size)//2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # center side platforms
    y1_side = max(y1 - stage_size - gap_width//2, 0)
    y2_side = max(y1 - gap_width//2, 0)
    y3_side = min(y2 + gap_width//2, terrain.width)
    y4_side = min(y2 + stage_size + gap_width//2, terrain.width)
    terrain.height_field_raw[x1:x2, y1_side:y2_side] = np.random.choice(height_range)
    terrain.height_field_raw[x1:x2, y3_side:y4_side] = np.random.choice(height_range)

def pit_terrain(terrain, depth, platform_width=[1.5, 4], platform_length=[1, 2.5, 0.5, 1]):
    depth = int(depth / terrain.vertical_scale)
    platform_length1 = int(np.random.uniform(platform_length[0], platform_length[1]) / terrain.horizontal_scale)
    platform_length2= int(np.random.uniform(platform_length[2], platform_length[3]) / terrain.horizontal_scale)
    platform_width1 = int(np.random.uniform(platform_width[0], platform_width[1]) / terrain.horizontal_scale / 2)
    platform_width2 = int(np.random.uniform(platform_width[0], platform_width[1]) / terrain.horizontal_scale / 2)

    x1 = terrain.length // 2 + int(1.5 / terrain.horizontal_scale)
    x2 = x1 + platform_length1
    x3 = terrain.length // 2 - int(1.5 / terrain.horizontal_scale)
    x4 = x3 - platform_length2
    y1 = terrain.width // 2 - platform_width1
    y2 = terrain.width // 2 + platform_width1
    y3 = terrain.width // 2 - platform_width2
    y4 = terrain.width // 2 + platform_width2
    terrain.height_field_raw[:] = 0
    terrain.height_field_raw[x1:x2, y1:y2] = depth
    terrain.height_field_raw[x4:x3, y3:y4] = depth


def pyramid_bridge_terrain(terrain, difficulty, depth=0.4, max_height=0.):
    bridge_width_ = 0.5 - min(0.2 * difficulty, 0.2)
    bridge_width = int(bridge_width_ / terrain.horizontal_scale)
    depth_range_radius = int((0.05 + min(difficulty * 0.1, 0.1)) / terrain.vertical_scale)
    depth_mid = -int(depth / terrain.vertical_scale)
    depth_range = np.arange(depth_mid - depth_range_radius, depth_mid + depth_range_radius, step=1)
    max_height = int(max_height / terrain.vertical_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)
    platform_size = 2.4 - min(0.5 * difficulty, 0.5)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    terrain.height_field_raw[:, :] = 0

    x11 = center_x + platform_size // 2; x12 = terrain.length - platform_size // 2
    y11 = center_y + platform_size // 2; y12 = terrain.width - platform_size // 2
    x21 = center_x - platform_size // 2; x22 = platform_size // 2
    y21 = center_y - platform_size // 2; y22 = platform_size // 2
    depth_ = np.random.choice(depth_range)
    terrain.height_field_raw[x11: x12, y22: y12] = depth_
    terrain.height_field_raw[x22: x21, y22: y12] = depth_
    terrain.height_field_raw[x22: x12, y11: y12] = depth_
    terrain.height_field_raw[x22: x12, y22: y21] = depth_

    y_b1 = center_y - bridge_width // 2; y_b2 = y_b1 + bridge_width
    terrain.height_field_raw[x11: x12, y_b1: y_b2] = np.random.choice(height_range)
    terrain.height_field_raw[x22: x21, y_b1: y_b2] = np.random.choice(height_range)
    x_b1 = center_x - bridge_width // 2; x_b2 = x_b1 + bridge_width
    terrain.height_field_raw[x_b1: x_b2, y11: y12] = np.random.choice(height_range)
    terrain.height_field_raw[x_b1: x_b2, y22: y21] = np.random.choice(height_range)

    return terrain

def bridge_terrain(terrain, difficulty, depth=0.4, max_height=0., max_height_bridge=0.02):
    bridge_width_ = 0.5 - min(0.25 * difficulty, 0.2)
    bridge_width = int(bridge_width_ / terrain.horizontal_scale)
    depth_range_radius = int((0.05 + min(difficulty * 0.1, 0.1)) / terrain.vertical_scale)
    depth_mid = -int(depth / terrain.vertical_scale)
    depth_range = np.arange(depth_mid - depth_range_radius, depth_mid + depth_range_radius, step=1)
    platform_length_ = 2.0 - min(0.8 * difficulty, 0.8)
    platform_length = int(platform_length_ / terrain.horizontal_scale)
    platform_width = [3.2, 3.6]
    platform_width1 = int(platform_width[0] / terrain.horizontal_scale)
    platform_width2 = int(np.random.uniform(platform_width[0], platform_width[1]) / terrain.horizontal_scale)
    platform_width3 = int(platform_width[0] / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)
    height_bridge_ = 0.0 + min(max_height_bridge * difficulty, max_height_bridge)
    height_bridge = int(height_bridge_ / terrain.vertical_scale)
    height_range_bridge = np.arange(-height_bridge-1, height_bridge, step=1)
    bridge_edge = int(0.02 / terrain.horizontal_scale)
    num_block_bridge = 5

    terrain.height_field_raw[:, :] = np.random.choice(depth_range)
    center_x = terrain.length // 2
    center_y = terrain.width // 2

    # front and back platforms
    front_x1 = 0
    front_x2 = platform_length//2
    front_x3 = terrain.length - platform_length//2
    front_x4 = terrain.length
    front_y1 = center_y - platform_width1//2
    front_y2 = center_y + platform_width1//2
    front_y3 = center_y - platform_width3//2
    front_y4 = center_y + platform_width3//2
    terrain.height_field_raw[front_x1:front_x2, front_y1:front_y2] = np.random.choice(height_range)
    terrain.height_field_raw[front_x3:front_x4, front_y3:front_y4] = np.random.choice(height_range)

    # center platform
    center_x1 = center_x - platform_length//2
    center_x2 = center_x + platform_length//2
    center_y1 = center_y - platform_width2//2
    center_y2 = center_y + platform_width2//2
    terrain.height_field_raw[center_x1:center_x2, center_y1:center_y2] = np.random.choice(height_range)

    # bridge
    bridge_x1 = front_x2 - bridge_edge
    bridge_x2 = center_x1 + bridge_edge
    bridge_x3 = center_x2 - bridge_edge
    bridge_x4 = front_x3 + bridge_edge
    if bridge_width % 2 == 0:
        bridge_y1 = center_y - bridge_width//2
        bridge_y2 = center_y + bridge_width//2
    else:
        bridge_y1 = center_y - bridge_width//2
        bridge_y2 = center_y + bridge_width//2 + 1
    for i in range(num_block_bridge):
        bridge_block_length1 = round((bridge_x2 - bridge_x1) // num_block_bridge)
        bridge_block_length2 = round((bridge_x4 - bridge_x3) // num_block_bridge)
        bridge_block_x1 = bridge_x1 + i * bridge_block_length1
        bridge_block_x2 = bridge_x1 + (i+1) * bridge_block_length1
        bridge_block_x3 = bridge_x3 + i * bridge_block_length2
        bridge_block_x4 = bridge_x3 + (i+1) * bridge_block_length2
        if i == num_block_bridge-1:
            bridge_block_x2 = bridge_x2
            bridge_block_x4 = bridge_x4
        terrain.height_field_raw[bridge_block_x1:bridge_block_x2, bridge_y1:bridge_y2] = np.random.choice(height_range_bridge)
        terrain.height_field_raw[bridge_block_x3:bridge_block_x4, bridge_y1:bridge_y2] = np.random.choice(height_range_bridge)
