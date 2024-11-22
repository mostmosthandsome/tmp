from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, BaseConfig
from legged_gym.envs.WukongIV.wk4_leg_config import Wk4RoughCfg, Wk4RoughCfgPPO
import time

from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam


class H1LocoCfg(Wk4RoughCfg):
    class env(Wk4RoughCfg.env):
        num_envs = 3072
        num_history_frames = 50
        history_stride = 1
        num_actions = 10
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
        pos = [0.0, 0.0, 1.02]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'torso_joint': 0,
            'right_shoulder_pitch_joint': 0,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': -0.1,
            'right_elbow_joint': 0.525,
            'left_shoulder_pitch_joint': 0,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0.1,
            'left_elbow_joint': 0.525,
            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': -0.08,
            'right_hip_pitch_joint': -0.4,
            'right_knee_joint': 0.95,
            'right_ankle_joint': -0.44,
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': -0.08,
            'left_hip_pitch_joint': -0.4,
            'left_knee_joint': 0.95,
            'left_ankle_joint': -0.44
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'torso': 200.0, 'shoulder': 60, 'elbow': 30,
                     'hip_yaw': 100, 'hip_roll': 80, 'hip_pitch': 240, 'knee': 240, 'ankle': 100
                     }  # [N*m/rad]
        # orgin PD of ankle: P: 'Ankle_Y': 50, 'Ankle_X': 50; D:'Ankle_Y': 0.5, 'Ankle_X': 0.3
        damping = {'torso': 4., 'shoulder': 0.5, 'elbow': 0.5,
                   'hip_yaw': 1, 'hip_roll': 1, 'hip_pitch': 2.5, 'knee': 4, 'ankle': 2
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.5
        action_scale = {'torso': 0.1, 'shoulder_pitch': 0.5, 'shoulder_roll': 0.02, 'shoulder_yaw': 0.5, 'elbow': 0.3,
                        'hip_yaw': 0.05, 'hip_roll': 0.1, 'hip_pitch': 0.35, 'knee': 0.5, 'ankle': 0.2}

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(Wk4RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Unitree_H1/urdf/h1.urdf'
        name = "H1"
        foot_name = 'ankle'
        terminate_after_contacts_on = ['torso', 'pelvis', 'shoulder', 'hip']
        flip_visual_attachments = True
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter


    class rewards(Wk4RoughCfg.rewards):
        base_height_target = 0.99

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
            'torso_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_joint'
        ]

        obs_dof_order = [
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_joint'
        ]

        action_dof_order = [
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_joint'
        ]


class H1LocoCfgPPO(Wk4RoughCfgPPO):
    seed = time.time()

    class runner(Wk4RoughCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'

        max_iterations = 10000  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'h1_test'
        experiment_name = 'h1_blind'

        save_interval = 100  # check for potential saves every this many iterations
        resume = False
        load_run = '240708_090625_key2_exp'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/h1_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wukong4_leg.py"]

