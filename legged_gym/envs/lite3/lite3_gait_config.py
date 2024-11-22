from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.lite3.lite3_config import Lite3RoughCfg, Lite3RoughCfgPPO
from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam
import os


class Lite3GaitCfg(Lite3RoughCfg):
    class env(Lite3RoughCfg.env):
        num_envs = 256
        num_history_frames = 10
        history_stride = 1
        num_observations = 45 + 8
        num_observations_history = num_observations * num_history_frames
        num_privileged_obs = num_observations + 198 + 3
        # if not None a priviledge_obs_buf will be returned by step() (critic obs for asymetric training).
        # None is returned otherwise
        episode_length_s = 20  # episode length in seconds
        estimation_terms = ["vel"]

    class terrain(Lite3RoughCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        # rough terrain only:
        num_cols = 5  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        depth_camera_on = False

    class commands(Lite3RoughCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        heading_ang_vel_kp = 0.3

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1., 1.]  # min max [rad/s]
            heading = [-3.14, 3.14]
            turn_heading = [-1.2, 1.2]

    class control(Lite3RoughCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'HipX': 20., 'HipY': 20, 'Knee': 20}  # [N*m/rad]
        damping = {'HipX': 0.5, 'HipY': 0.5, 'Knee': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class sim(Lite3RoughCfg.sim):
        dt = 0.005

    class asset(Lite3RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Lite3/urdf/Lite3_right_first.urdf'
        name = "MiniLiteV3"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class init_state(Lite3RoughCfg.init_state):
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FR_HipX': -0.1,  # [rad]
            'FL_HipX': 0.1,  # [rad]
            'HR_HipX': -0.1,  # [rad]
            'HL_HipX': 0.1,  # [rad]

            'FR_HipY': 0.8,  # [rad]
            'FL_HipY': 0.8,  # [rad]
            'HR_HipY': 1.,  # [rad]
            'HL_HipY': 1.,  # [rad]

            'FR_Knee': -1.5,  # [rad]
            'FL_Knee': -1.5,  # [rad]
            'HR_Knee': -1.5,  # [rad]
            'HL_Knee': -1.5,  # [rad]
        }

    class normalization(Lite3RoughCfg.normalization):
        class obs_scales(Lite3RoughCfg.normalization.obs_scales):
            lin_vel = 1
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 10 / 3

        obs_dof_order = [
            "FR_HipX", "FR_HipY", "FR_Knee", "FL_HipX", "FL_HipY", "FL_Knee",
            "HR_HipX", "HR_HipY", "HR_Knee", "HL_HipX", "HL_HipY", "HL_Knee"
        ]
        clip_observations = 100.

        action_dof_order = [
            "FR_HipX", "FR_HipY", "FR_Knee", "FL_HipX", "FL_HipY", "FL_Knee",
            "HR_HipX", "HR_HipY", "HR_Knee", "HL_HipX", "HL_HipY", "HL_Knee"
        ]
        clip_actions = 100.

    class noise(Lite3RoughCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(Lite3RoughCfg.noise.noise_scales):
            lin_vel = 0.1
            ang_vel = 0.2
            dof_pos = 0.01
            dof_vel = 0.05
            gravity = 0.05
            height_measurements = 0.1

    class rewards(Lite3RoughCfg.rewards):
        class item:
            lin_Vel = GaussianRewardParam(0.2, 0.447, 0)
            ang_Vel = GaussianRewardParam(0.1, 0.447, 0)

            bRot = GaussianRewardParam(0.05, 0.1, 0)
            bTwist = GaussianRewardParam(0.05, 0.447, 0)

            zVel = GaussianRewardParam(0.0, 10, 0)
            bHgt = GaussianRewardParam(0.1, 0.0707, 0)

            cotr = CauchyRewardParam(0.05, 0.2, 1, 0)
            eVel = CauchyRewardParam(0.125, 0.5, 1, 0)
            eFrc = CauchyRewardParam(0.125, 0.2, 1, 0)
            ipct = CauchyRewardParam(0.05, 0.2, 3, 0)

            cnct_forces = CauchyRewardParam(0.05, 0.3, 3, 1)
            smth = CauchyRewardParam(0.05, 1, 1, 0)
            jVel = CauchyRewardParam(0.05, 25, 2, 0)

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        base_height_target = 0.3
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        # percentage of urdf limits, values above this limit are penalized
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.7

        foot_height_target = 0.2
        max_contact_force = 100.  # forces above this value are penalized

        mean_vel_window = 1.0

    class gait:
        frequency = 2.0
        contactTolerance = 0.05
        swingRatio = "default"
        state_type = 'adaptive_sine'
        name = "trot"


class Lite3GaitCfgPPO(Lite3RoughCfgPPO):
    seed = 255

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [2048, 512, 128]
        critic_hidden_dims = [2048, 512, 128]
        encoder_hidden_dims = [1024, 256, 64]
        latent_dims = 19  # 16 for implicit encoding and 3 for explicit velocity estimation

        estimation_dims = {"vel": 3, "implicit": 16}

    class runner(Lite3RoughCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'
        run_name = 'trot'
        experiment_name = 'gait_lite3'
        max_iterations = 10000  # number of policy updates
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/lite3/lite3_gait_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/lite3/lite3_gait.py"]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * n steps / n minibatches
        learning_rate = 5.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 0.5
        # est_loss_coeff = {"hgt": 2}  # est h+z
        # est_loss_coeff = {"vel": 40, "hm": 0.5} # est v+hm+z
        est_loss_coeff = {"vel": 20} # est v/v+z
        # est_loss_coeff = {"vel": 40, "hgt": 2, "hm": 0.5}
        # est_loss_coeff = {} # est z
        obs_mse_coeff = 5
        vae_kl_coeff = 1
