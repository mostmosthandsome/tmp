from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, BaseConfig
from legged_gym.envs.WukongIV.wk4_leg_config import Wk4RoughCfg, Wk4RoughCfgPPO
import time

from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam


class DancerLocoCfg(Wk4RoughCfg):
    class env(Wk4RoughCfg.env):
        num_envs = 2048
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

    class terrain(Wk4RoughCfg.terrain):
        mesh_type = "plane"
        # mesh_type = "trimesh"

    class commands(Wk4RoughCfg.commands):
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(Wk4RoughCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.2, 0.2]  # min max [rad/s]
            heading = [-1.57, 1.57]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [-0.09983342, 0., 0., 0.99500417]  # x,y,z,w [quat]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw' : 0.041147,
            'left_hip_roll' : 0.142089,
            'left_hip_pitch' : 0.978417,
            'left_knee' : 1.242302,
            'left_ankle_pitch' : 0.546813,
            'left_ankle_roll' : 0.147888,
            'right_hip_yaw' : -0.041147,
            'right_hip_roll' : 0.142089,
            'right_hip_pitch' : 0.978417,
            'right_knee' : 1.242302,
            'right_ankle_pitch' : 0.546813,
            'right_ankle_roll' : 0.147888,
            'left_arm_upper' : 0.000000,
            'left_arm_lower' : 3.054326,
            'right_arm_upper' : 0.000000,
            'right_arm_lower' : 3.054326
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        # stiffness = {'arm': 61.0, 'hip_yaw': 61.0, 'hip_roll': 61.0, 'hip_pitch': 301.0,
        #              'knee': 301.0, 'ankle_pitch': 301.0, 'ankle_roll':  61.0,}
        # damping = {'arm': 1.24, 'hip_yaw':1.24, 'hip_roll':1.24, 'hip_pitch':  6.001,
        #            'knee': 6.001, 'ankle_pitch':  6.001,  'ankle_roll':1.24}
        stiffness = {'left': 0, 'right': 0}
        damping = {'left': 0, 'right': 0}
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.5
        action_scale = {'arm': 1, 'hip_yaw':0.4, 'hip_roll':0.2, 'hip_pitch':  1.5,
                        'knee': 2, 'ankle_pitch':  1.5,  'ankle_roll':0.5}

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(Wk4RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/ZJU_Dancer/urdf/humanoid.urdf'
        name = "Dancer"
        foot_name = 'foot'
        terminate_after_contacts_on = ['chest', 'leg_pitch']
        flip_visual_attachments = True
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class sim(Wk4RoughCfg.sim):
        substeps = 2
        class physx(Wk4RoughCfg.sim.physx):
            num_position_iterations = 20
            num_velocity_iterations = 1
            contact_offset = 0.000005
            friction_offset_threshold = 0.00005

    class rewards:
        class item:
            lin_Vel = GaussianRewardParam(0.24, 0.3, 0)
            ang_Vel = GaussianRewardParam(0.1, 0.4, 0)
            bRot = GaussianRewardParam(0.03, 0.1, 0)
            bTwist = GaussianRewardParam(0.07, 0.4, 0)
            bHgt = GaussianRewardParam(0.01, 0.0707, 0)
            cotr = CauchyRewardParam(0.05, 1.2, 1, 0)
            eVel = CauchyRewardParam(0.125, 0.25, 1, 0)
            eFrc = CauchyRewardParam(0.125, 0.2, 1, 0)
            # ipct = CauchyRewardParam(0.05, 0.16, 3, 0)
            # cnct_forces = CauchyRewardParam(0.05, 0.12, 3, 1)
            smth = CauchyRewardParam(0.1, 0.4, 1, 0)
            jVel = CauchyRewardParam(0.05, 10, 2, 0)
        base_height_target = 0.48
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.7  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        foot_height_target = 0.12
        max_contact_force = 600.  # forces above this value are penalized
        mean_vel_window = 1.0  # window length for computing average body velocity (in sec)

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
            'left_hip_yaw','left_hip_roll','left_hip_pitch', 'left_knee','left_ankle_pitch','left_ankle_roll',
            'right_hip_yaw','right_hip_roll','right_hip_pitch', 'right_knee','right_ankle_pitch','right_ankle_roll',
            'left_arm_upper','left_arm_lower', 'right_arm_upper','right_arm_lower'
        ]

        obs_dof_order = [
            'left_hip_yaw','left_hip_roll','left_hip_pitch', 'left_knee','left_ankle_pitch','left_ankle_roll',
            'right_hip_yaw','right_hip_roll','right_hip_pitch', 'right_knee','right_ankle_pitch','right_ankle_roll'
        ]

        action_dof_order = [
            'left_hip_yaw','left_hip_roll','left_hip_pitch', 'left_knee','left_ankle_pitch','left_ankle_roll',
            'right_hip_yaw','right_hip_roll','right_hip_pitch', 'right_knee','right_ankle_pitch','right_ankle_roll'
        ]

    class domain_rand(Wk4RoughCfg.domain_rand):
        randomize_init_state = False
        randomize_init_dof = False
        randomize_Kp_factor = False
        randomize_Kd_factor = False


class DancerLocoCfgPPO(Wk4RoughCfgPPO):
    seed = 10086

    class runner(Wk4RoughCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'

        max_iterations = 10000  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'dancer_test'
        experiment_name = 'dancer_blind'

        save_interval = 100  # check for potential saves every this many iterations
        resume = False
        load_run = '240708_090625_key2_exp'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/dancer_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wukong4_leg.py"]

