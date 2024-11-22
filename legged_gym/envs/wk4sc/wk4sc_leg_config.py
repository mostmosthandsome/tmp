from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import time
from legged_gym.utils.reward import GeneralRewardParam, GaussianRewardParam, CauchyRewardParam


class Wk4scWPLocoCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_history_frames = 50
        history_stride = 1
        num_observations = 45 + 4
        num_observations_history = num_observations * num_history_frames
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

        # num_privileged_obs = num_observations + 3 + 1 + 18 + 81  # base vel, base height and foot height
        # num_privileged_obs = num_observations + 3 + 9  # base vel, base height and foot height
        num_privileged_obs = num_observations + 3 + 81 + 1  # base vel, heightmap
        estimation_terms = ["vel"]  # est v+z

    class terrain(LeggedRobotCfg.terrain):
        measure_heights = True
        # mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        # measured_points_x = [-0.1, 0., 0.1]  # 1mx1m rectangle (without center line)
        # measured_points_y = [-0.1, 0., 0.1]
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        measured_foot_points_x = [-0.1, 0., 0.1]  # 1mx1.6m rectangle (without center line)
        measured_foot_points_y = [-0.1, 0., 0.1]  # 1mx1.6m rectangle (without center line)

        curriculum = True
        robot_move = True
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.2, 0.4, 0.1, 0.1, 0.1, 0, 0, 0.1]
        terrain_proportions = [0.2, 0.3, 0.1, 0.1, 0.3]
        static_friction = 0.75
        dynamic_friction = 0.7
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        terrain_length = 8.
        terrain_width = 8.

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        # default: Vx, Vy, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # for humanoid, expand for arm operation
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        dist2lin_vel_scale = 2.0

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.4, 0.4]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            heading = [-1.57, 1.57]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.73]  # x,y,z [m]
        shoulder_y_range = [-0.3, 0.3]
        elbow_range = [-0.3, 0.5]
        rand_joint = ['Shoulder_Y_R', 'Shoulder_Y_L', 'Elbow_R', 'Elbow_L']
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'Waist': 0.,
            'Shoulder_Z_R': -0.1,
            'Shoulder_X_R': 0.,
            'Shoulder_Y_R': 0.,
            'Elbow_R': 0.525,
            'Wrist_X_R': 0,
            'Wrist_Z_R': 0,
            'Wrist_Y_R': 0,

            'Shoulder_Z_L': -0.1,
            'Shoulder_X_L': 0.,
            'Shoulder_Y_L': 0.,
            'Elbow_L': 0.525,
            'Wrist_X_L': 0,
            'Wrist_Z_L': 0,
            'Wrist_Y_L': 0,

            'Hip_Z_R': 0,
            'Hip_X_R': 0.006,
            'Hip_Y_R': -0.52,
            'Knee_R': 0.786,
            'Ankle_Y_R': -0.263,
            'Ankle_X_R': -0.006,
            'Hip_Z_L': 0,
            'Hip_X_L': 0.006,
            'Hip_Y_L': -0.52,
            'Knee_L': 0.786,
            'Ankle_Y_L': -0.263,
            'Ankle_X_L': -0.006
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'Waist': 400.0, 'Shoulder_Z': 200, 'Shoulder_X': 200, 'Shoulder_Y': 200,
                     'Elbow': 200, 'Wrist_X': 40, 'Wrist_Z': 40, 'Wrist_Y': 40,
                     'Hip_Z': 100, 'Hip_X': 80, 'Hip_Y': 240, 'Knee': 240, 'Ankle_Y': 80, 'Ankle_X': 60
                     }  # [N*m/rad]
        damping = {'Waist': 6., 'Shoulder_Z': 4, 'Shoulder_X': 4, 'Shoulder_Y': 4,
                   'Elbow': 3, 'Wrist_X': 0.5, 'Wrist_Z': 0.5, 'Wrist_Y': 0.5,
                   'Hip_Z': 1, 'Hip_X': 1, 'Hip_Y': 2.5, 'Knee': 4, 'Ankle_Y': 1, 'Ankle_X': 0.8
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = {'Waist': 0.1, 'Shoulder_Z': 0.05, 'Shoulder_X': 0.02, 'Shoulder_Y': 0.5,
                        'Elbow': 0.3, 'Wrist_X': 0.8, 'Wrist_Z': 0.2, 'Wrist_Y': 0.3,
                        'Hip_Z': 0.05, 'Hip_X': 0.1, 'Hip_Y': 0.35, 'Knee': 0.5, 'Ankle_Y': 0.2, 'Ankle_X': 0.1}

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/CETC-01/urdf/CETC-01.urdf'
        name = "Wukong4s-CETC"
        foot_name = 'FOOT'
        terminate_after_contacts_on = ['TORSO', 'UPPERARM', 'ILIUM', 'ISCHIUM', 'THIGH']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-2., 7.5]
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]
        rand_interval_s = 8
        push_robots = False
        push_interval_s = 6
        # Unit of interval is sec
        max_push_vel_xy = 1.
        randomize_init_state = True
        randomize_lag_timesteps = True
        lag_timesteps = 2

    class rewards(LeggedRobotCfg.rewards):

        class item:
            lin_Vel = CauchyRewardParam(0.2, 0.35, 1, 0)
            way_point = GeneralRewardParam(0.5, 0)
            ang_Vel = GaussianRewardParam(0.15, 0.4, 0)

            bRot = GaussianRewardParam(0.1, 0.1, 0)
            bTwist = GaussianRewardParam(0.1, 0.447, 0)

            zVel = GaussianRewardParam(0.0, 10, 0)
            bHgt = GaussianRewardParam(0.1, 0.0707, 0)

            cotr = CauchyRewardParam(0.05, 0.2, 1, 0)
            eVel = CauchyRewardParam(0.1, 0.5, 1, 0)
            eFrc = CauchyRewardParam(0.1, 0.2, 1, 0)
            ipct = CauchyRewardParam(0.04, 0.2, 3, 0)

            cnct_forces = CauchyRewardParam(0.04, 0.3, 3, 1)
            smth = CauchyRewardParam(0.04, 1, 1, 0)
            jPos = CauchyRewardParam(0.04, 0.2, 1, 0)
            jVel = CauchyRewardParam(0.04, 20, 2, 0)

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.6  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.71
        foot_height_target = 0.12
        max_contact_force = 600.  # forces above this value are penalized
        mean_vel_window = 1.0  # window length for computing average body velocity (in sec)

    class gait(LeggedRobotCfg.gait):
        name = "stance_walk"
        frequency = 1.2
        contactTolerance = 0.05
        offset = None
        state_type = 'ADAPTIVE_SINE'

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.166667
            dof_pos = 1.0
            dof_vel = 0.1
            height_measurements = 1.
            base_height = 30.
            foot_height = 30.

        clip_observations = 100.
        clip_actions = 10.

        whole_body_dof_order = [
            'Waist',
            'Shoulder_Z_R', 'Shoulder_X_R', 'Shoulder_Y_R', 'Elbow_R', 'Wrist_X_R', 'Wrist_Z_R', 'Wrist_Y_R',
            'Shoulder_Z_L', 'Shoulder_X_L', 'Shoulder_Y_L', 'Elbow_L', 'Wrist_X_L', 'Wrist_Z_L', 'Wrist_Y_L',
            'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_R', 'Ankle_Y_R', 'Ankle_X_R',
            'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_L', 'Ankle_Y_L', 'Ankle_X_L'
        ]

        obs_dof_order = [
            'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_R', 'Ankle_Y_R', 'Ankle_X_R',
            'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_L', 'Ankle_Y_L', 'Ankle_X_L'
        ]

        action_dof_order = [
            'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_R', 'Ankle_Y_R', 'Ankle_X_R',
            'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_L', 'Ankle_Y_L', 'Ankle_X_L'
        ]

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurements = 0.05

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [-2, -2, 6]  # [m]
        lookat = [0., 0, 3.]  # [m]

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

class Wk4scWPLocoCfgPPO(LeggedRobotCfgPPO):
    seed = time.time()

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [2048, 512, 128]
        critic_hidden_dims = [2048, 512, 128]
        encoder_hidden_dims = [1024, 256, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        latent_dims = 16 + 3  # 16: implicit encoding; 3: velocity estimation;
        estimation_dims = {"vel": 3, "implicit": 16}

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'

        max_iterations = 10000  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'waypoints_s2'
        experiment_name = 'rough_wk4sc'

        save_interval = 100
        resume = True
        load_run = '240817_033143_waypoints_v_s1'  # -1 = last run
        # load_run = -1
        checkpoint = -1  # -1 = last saved model

        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/wk4sc/wk4sc_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/wk4sc/wk4sc_leg.py"]

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
        est_loss_coeff = {"vel": 20}  # est v/v+z
        obs_mse_coeff = 2
        vae_kl_coeff = 0.6
