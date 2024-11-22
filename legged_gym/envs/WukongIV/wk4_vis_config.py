from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import time
from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam


class Wukong4VisualCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_history_frames = 25
        history_stride = 2
        num_observations = 45 + 4
        num_privileged_obs = num_observations + 198 + 3 + 25*2
        num_observations_history = num_observations * num_history_frames
        num_actions = 12
        depth_width = 86
        depth_height = 60
        vision_in_channels = 2
        episode_length_s = 20  # episode length in seconds
        estimation_terms = ["vel", "heightmap"]
        # estimation_terms = ["foot_height", "heightmap"]
        # estimation_terms = ["vel", "foot_hgt", "heightmap"]
        finetune = True

        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm

        class camera(LeggedRobotCfg.env.camera):
            camera_width = 106
            camera_height = 60
            depth_crop = [12, 8]  # cropped border of depth image [left, right]
            depth_width = camera_width - sum(depth_crop)
            depth_height = 60
            fov = 87.0  # unit:degree
            vision_in_channels = 2

    class terrain(LeggedRobotCfg.terrain):
        robot_move = True
        use_isaacgym_rendering = False

        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh

        border_type = 'rough'
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8, 0.9, 1.0, 1.1, 1.2]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measure_foot_heights = True
        # measured_foot_points_x = [0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 1mx1.6m rectangle (without center line)
        # measured_foot_points_y = [-0.1, 0., 0.1]  # 1mx1.6m rectangle (without center line)
        measured_foot_points_x = [-0.1, -0.05, 0., 0.05, 0.1]
        measured_foot_points_y = [-0.1, -0.05, 0., 0.05, 0.1]
        # measured_foot_points_x = [0.]  # 1mx1.6m rectangle (without center line)
        # measured_foot_points_y = [0.]  # 1mx1.6m rectangle (without center line)
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        restitution = 0.
        border_length = [2, 10]  # [back, front]
        border_width = [10, 10]   # [right, left]
        curriculum = True
        arranged = False
        terrain_difficulty = 0.8
        static_friction = 0.75
        dynamic_friction = 0.7
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs down, stairs up, discrete, gap, pit, flat]
        terrain_type_name   = ['smoothSlope','roughSlope','upStairs ','downStairs','discrete','stones','gap', 'pit', 'balance', 'flat']
        # terrain_proportions = [0.2,          0.4,         0.1,        0.1,         0.2,       0.0,     0.0,   0.0,   0.0,       0.0]
        terrain_proportions = [0.0,          0.0,         0.1,        0.1,         0.2,       0.0,     0.6,   0.0,   0.0,       0.0]
        # terrain_proportions = [0.0, 0.2, 0.3, 0.3, 0.2, 0., 0., 0.]
        # terrain_proportions = [0.0, 0.1,   0.2, 0.2,   0.2,   0.0, 0.2,   0.1]
        # terrain_proportions = [0.0, 0.0,   0.0, 0.0,   0.0,   0, 1, 0.0, 0]
        sparse_types = [8]
        ftedge_rew_types = [2, 3]

        visualize_edge_masks = False
        edge_width_threshs = [0.05, 0.1]   # threshold for terrain edge
        edge_rew_coeffs    = [1.0,  0.5]   # reward coefficients for terrain edge


        # terrain_proportions = [0]
        # Heightfeild only:
        heightfield_range = [-0.03, 0.03]
        # heightfield_range = [-0.0, 0.0]
        rough_slope_range = [-0.02, 0.02]
        heightfield_resolution = 0.005
        depth_camera_on = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        tracking_mode = "heading"  # choose from y_position, heading, ang_vel
        tracking_strength = 0.2  # for y_position mode, turning sharpens (could have oscillation) with larger strength
        num_commands = 5  # default: vx, vy, omega_yaw, heading, const_omega_yaw
        # in y_position mode heading and ang_vel_yaw are recomputed from coronal position error
        # for some cases in y_position mode, the command could be formulated as the other two modes
        # check Wukong4Visual._post_physics_step_callback() for further explanation
        # in heading mode ang_vel_yaw is recomputed from heading error
        # in ang_vel mode, ang_vel_yaw is directly assigned
        resampling_time = 10.  # time before command are changed[s]

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.5]  # min max [m/s]
            lin_vel_y = [-0.15, 0.15]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            # turn_heading = [-1.57, 1.57]
            # heading = [-1.57, 1.57]
            turn_heading = [-0.1, 0.1]
            heading = [-0.1, 0.1]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.77]  # x,y,z [m]
        rot = [0., 0.0, 0.0, 1]  # x,y,z,w [quat]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'Waist': 0.,
            'Shoulder_Z_R': -0.1,
            'Shoulder_X_R': 0.,
            'Shoulder_Y_R': 0.,
            'Elbow_R': 0.525,
            'Shoulder_Z_L': -0.1,
            'Shoulder_X_L': 0.,
            'Shoulder_Y_L': 0.,
            'Elbow_L': 0.525,

            'Hip_Z_R': 0,
            'Hip_X_R': -0.08,
            'Hip_Y_R': -0.4,
            'Knee_R': 0.95,
            'Ankle_Y_R': -0.44,
            'Ankle_X_R': 0.04,
            'Hip_Z_L': 0,
            'Hip_X_L': -0.08,
            'Hip_Y_L': -0.4,
            'Knee_L': 0.95,
            'Ankle_Y_L': -0.44,
            'Ankle_X_L': 0.04
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'Waist': 200.0, 'Shoulder_Z': 30, 'Shoulder_X': 40, 'Shoulder_Y': 60, 'Elbow': 30,
                     'Hip_Z': 100, 'Hip_X': 80, 'Hip_Y': 240, 'Knee': 240, 'Ankle_Y': 80, 'Ankle_X': 60
                     }  # [N*m/rad]
        # orgin PD of ankle: P: 'Ankle_Y': 50, 'Ankle_X': 50; D:'Ankle_Y': 0.5, 'Ankle_X': 0.3
        damping = {'Waist': 4., 'Shoulder_Z': 0.5, 'Shoulder_X': 0.5, 'Shoulder_Y': 0.8, 'Elbow': 0.5,
                   'Hip_Z': 1, 'Hip_X': 1, 'Hip_Y': 2.5, 'Knee': 4, 'Ankle_Y': 1, 'Ankle_X': 0.6
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.5
        action_scale = {'Waist': 0.1, 'Shoulder_Z': 0.05, 'Shoulder_X': 0.02, 'Shoulder_Y': 0.5, 'Elbow': 0.3,
                        'Hip_Z': 0.05, 'Hip_X': 0.1, 'Hip_Y': 0.35, 'Knee': 0.5, 'Ankle_Y': 0.2, 'Ankle_X': 0.1}

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/WukongIV/urdf/WuKongIV_preciseMass.urdf'
        name = "Wukong4"
        foot_name = 'FOOT'
        terminate_after_contacts_on = ['TORSO', 'UPPERARM', 'ILIUM', 'ISCHIUM', 'THIGH']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

        angular_damping = 0.
        max_angular_velocity = 2800.
        armature = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [-0.2, 0.7]
        randomize_base_mass = True
        added_mass_range = [-3., 12.5]
        com_displacement_range_x = [-0.1, 0.1]
        com_displacement_range_yz = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]
        rand_interval_s = 8
        push_robots = False
        randomize_init_state = True
        randomize_lag_timesteps = True
        lag_timesteps = 2

        class camera:
            randomize_camera = True
            cam_pos_base = [0.10489, 0.0175, 0.16049]
            cam_ori_base = [90, -90, 49]  # XZY euler angle, in degree
            transform_p_x = [-0.01, 0.01]
            transform_p_y = [-0.01, 0.01]
            transform_p_z = [-0.01, 0.01]
            transform_r_x = [-1., 1.]
            transform_r_y = [-1., 1.]
            transform_r_z = [-1., 1.]
            horizontal_fov = [-1, 1]

    class rewards:
        class item:
            lin_Vel = GaussianRewardParam(0.2, 0.6, 0)
            ang_Vel = GaussianRewardParam(0.1, 0.4, 0)

            bRot = GaussianRewardParam(0.1, 0.1, 0)
            bTwist = GaussianRewardParam(0.1, 0.4, 0)

            zVel = GaussianRewardParam(0.0, 6, 0)
            bHgt = GaussianRewardParam(0.1, 0.0707, 0)
            cotr = CauchyRewardParam(0.05, 0.2, 1, 0)
            eVel = CauchyRewardParam(0.1, 0.5, 1, 0)
            eFrc = CauchyRewardParam(0.1, 0.2, 1, 0)
            ipct = CauchyRewardParam(0.05, 0.16, 3, 0)

            cnct_forces = CauchyRewardParam(0.0, 0.3, 3, 1)
            smth = CauchyRewardParam(0.05, 0.6, 1, 0)
            jPos = CauchyRewardParam(0.0, 0.2, 1, 0)
            jVel = CauchyRewardParam(0.05, 16, 2, 0)

            # stumble = CauchyRewardParam(0.02, 0.1, 1, 0)
            # ftEdge = CauchyRewardParam(0.03, 0.3, 1, 0)

        class item_finetune(item):
            bTwist = GaussianRewardParam(0.05, 0.4, 0)

            stumble = CauchyRewardParam(0.02, 0.1, 1, 0)
            ftEdge = CauchyRewardParam(0.03, 0.3, 1, 0)

        only_positive_rewards = True
        # if true negative total rewards are clipped at zero (avoids early termination problems)

        soft_dof_pos_limit = 0.7  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        base_height_target = 0.71
        max_contact_force = 600.  # forces above this value are penalized
        mean_vel_window = 1.0  # window length for computing average body velocity (in sec)

    class gait(LeggedRobotCfg.gait):
        frequency: float = 1.2
        contactTolerance = 0.05
        state_type: str = 'adaptive_sine'

        name: str = "walk"
        swingRatio: float = 0.45
        offset = None
        symmetricity: str = "NONE"

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5
            base_height = 30.
            foot_height = 30.
            depth_image = 0.5

        clip_observations = 100.
        clip_actions = 100.
        depth_mean = 1.
        whole_body_dof_order = [
            'Waist',
            'Shoulder_Z_R', 'Shoulder_X_R', 'Shoulder_Y_R', 'Elbow_R',
            'Shoulder_Z_L', 'Shoulder_X_L', 'Shoulder_Y_L', 'Elbow_L',
            'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_R', 'Ankle_Y_R', 'Ankle_X_R',
            'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_L', 'Ankle_Y_L', 'Ankle_X_L'
        ]

        obs_dof_order = [
            # 'Waist',
            # 'Shoulder_Z_R', 'Shoulder_X_R', 'Shoulder_Y_R', 'Elbow_R',
            # 'Shoulder_Z_L', 'Shoulder_X_L', 'Shoulder_Y_L', 'Elbow_L',
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

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class Wukong4VisualCfgPPO(LeggedRobotCfgPPO):
    seed = time.time()
    runner_class_name = 'OnPolicyRunner'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        encoder_hidden_dims = [1024, 256]
        num_contrast_feature = 32
        activation = 'elu'  # can be gelu, elu, relu, selu, crelu, lrelu, tanh, sigmoid
        token_channels = 64

        # only for 'ActorCriticRecurrent':
        rnn_type = 'gru'
        rnn_hidden_size = 128
        rnn_num_layers = 1

        estimation_dims = {"vel": 3, "implicit": 16}

        latent_dims = 16 + 3  # 16: implicit encoding; 3: velocity estimation;
        # estimation_dims = {"vel": 3, "foot_height": 2, "implicit": 16}
        # estimation_dims = {"foot_height": 2, "implicit": 16}

        visual_encode_hidden_dims = [512, 128, 64]
        visual_decode_hidden_dims = [64, 128, 512]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * nsteps / num_mini_batches
        learning_rate = 5.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 0.5

        # est_loss_coeff = {"vel": 1, "foot_hgt": 0.4}
        est_loss_coeff = {"vel": 5}
        # est_loss_coeff = {"foot_hgt": 0.4}
        height_reconstruct_coeff = 1
        obs_mse_coeff = 2
        vae_kl_coeff = 2
        # SwAV_coeff = 1

        blend_decay = 0.002
        expert_path = f"{LEGGED_GYM_ROOT_DIR}/logs/rough_wk4/exported/240724_181221_v_25h2.10000/policy.pt"

    class runner(LeggedRobotCfgPPO.runner):
        # policy_class_name = 'AsymActorCritic'
        # algorithm_class_name = 'PPO_DW'
        # policy_class_name = 'RecurrentTransActorCritic'
        # algorithm_class_name = 'ActionDistill'
        # algorithm_class_name = 'PPO_Contrast'
        policy_class_name = 'MultiModalActorCritic'
        algorithm_class_name = 'PPO_MM'

        num_steps_per_env = 48  # per iteration
        max_iterations = 20000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'visual_wk4'
        run_name = 'addStumbleGap'
        # load and resume
        resume = False
        load_run = '240820_150257_Proper_MH_VAE'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wk4_vis_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/Wukong4Visual.py"]
