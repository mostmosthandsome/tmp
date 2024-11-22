from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, BaseConfig
from legged_gym.envs.WukongIV.wk4_leg_config import Wk4RoughCfg, Wk4RoughCfgPPO
import time

from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam


class Dr01LocoCfg(Wk4RoughCfg):
    class env(Wk4RoughCfg.env):
        num_envs = 3072
        num_history_frames = 50
        history_stride = 1
        num_actions = 12
        num_observations = 9 + num_actions * 3 + 4
        num_observations_history = num_observations * num_history_frames
        episode_length_s = 20  # episode length in seconds
        num_privileged_obs = num_observations + 3 + 1 + 18 + 81  # base vel, base height and foot height
        estimation_terms = ["vel"]

        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm

    class commands(Wk4RoughCfg.commands):
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(Wk4RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.4, 0.4]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            heading = [-1.57, 1.57]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.72]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'Waist': 0.,
            'Shoulder_Z_R': -0.1,
            'Shoulder_X_R': 0.,
            'Shoulder_Y_R': 0.3,
            'Elbow_R': 0.525,
            'Wrist_X_R': 0.,
            'Wrist_Z_R': 0.,
            'Wrist_Y_R': 0.,
            'Shoulder_Z_L': -0.1,
            'Shoulder_X_L': 0.,
            'Shoulder_Y_L': 0.,
            'Elbow_L': 0.525,
            'Wrist_X_L': 0.,
            'Wrist_Z_L': 0.,
            'Wrist_Y_L': 0.,

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
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'Waist': 400.0, 'Shoulder_Z': 200, 'Shoulder_X': 200, 'Shoulder_Y': 200, 'Elbow': 200, 'Wrist': 60,
                     'Hip_Z': 100, 'Hip_X': 80, 'Hip_Y': 240, 'Knee': 240, 'Ankle_Y': 80, 'Ankle_X': 80
                     }  # [N*m/rad]
        # orgin PD of ankle: P: 'Ankle_Y': 50, 'Ankle_X': 50; D:'Ankle_Y': 0.5, 'Ankle_X': 0.3
        damping = {'Waist': 6., 'Shoulder_Z': 4, 'Shoulder_X': 4, 'Shoulder_Y': 4, 'Elbow': 4, 'Wrist': 0.5,
                   'Hip_Z': 1, 'Hip_X': 1, 'Hip_Y': 2.5, 'Knee': 4, 'Ankle_Y': 1.6, 'Ankle_X': 1.6
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.5
        action_scale = {'Waist': 0.1, 'Shoulder_Z': 0.05, 'Shoulder_X': 0.02, 'Shoulder_Y': 0.5,
                        'Elbow': 0.3, 'Wrist_X': 0.7, 'Wrist_Z': 0.2 ,'Wrist_Y': 0.2,
                        'Hip_Z': 0.05, 'Hip_X': 0.1, 'Hip_Y': 0.35, 'Knee': 0.5, 'Ankle_Y': 0.2, 'Ankle_X': 0.1}

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(Wk4RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/DR01/urdf/DR01.urdf'
        name = "Dr01"
        foot_name = 'FOOT'
        terminate_after_contacts_on = ['TORSO', 'UPPERARM', 'ILIUM', 'ISCHIUM', 'THIGH']
        flip_visual_attachments = True
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter


    class rewards(Wk4RoughCfg.rewards):
        base_height_target = 0.73

    class normalization(Wk4RoughCfg.normalization):
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


class Dr01LocoCfgPPO(Wk4RoughCfgPPO):
    seed = time.time()

    class runner(Wk4RoughCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'

        max_iterations = 10000  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'Dr01_test'
        experiment_name = 'Dr01_blind'

        save_interval = 100  # check for potential saves every this many iterations
        resume = False
        load_run = '240708_090625_key2_exp'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/Dr01_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wukong4_leg.py"]

