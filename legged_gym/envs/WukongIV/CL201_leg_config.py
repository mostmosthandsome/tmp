from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, BaseConfig
from legged_gym.envs.WukongIV.wk4_leg_config import Wk4RoughCfg, Wk4RoughCfgPPO
import time

from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam


class CL201LocoCfg(Wk4RoughCfg):
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
        pos = [0.0, 0.0, 1.24]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'J_waist_yaw': 0,
            'J_waist_roll': 0,
            'J_hip_r_roll': -0.087,
            'J_hip_r_yaw': -0.003,
            'J_thigh_r_pitch': 0.5,
            'J_shins_r_pitch': -0.94,
            'J_backsole_r_pitch': -0.44,
            'J_backsole_r_roll': 0.087,
            'J_hip_l_roll': 0.087,
            'J_hip_l_yaw': 0.003,
            'J_thigh_l_pitch': 0.5,
            'J_shins_l_pitch': -0.94,
            'J_backsole_l_pitch': -0.44,
            'J_backsole_l_roll': 0.087
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'waist_yaw': 400, 'waist_roll': 400,
                     'hip_r_roll': 140, 'hip_r_yaw': 100, 'thigh_r_pitch': 240, 'shins_r_pitch': 240, 'backsole_r_pitch': 100,  'backsole_r_roll': 100,
                     'hip_l_roll': 140, 'hip_l_yaw': 100, 'thigh_l_pitch': 240, 'shins_l_pitch': 240, 'backsole_l_pitch': 100,  'backsole_l_roll': 100
                     }  # [N*m/rad]
        # orgin PD of ankle: P: 'Ankle_Y': 50, 'Ankle_X': 50; D:'Ankle_Y': 0.5, 'Ankle_X': 0.3
        damping = {'waist_yaw': 6, 'waist_roll': 6,
                   'hip_r_roll': 1.2, 'hip_r_yaw': 1, 'thigh_r_pitch': 2.5, 'shins_r_pitch': 4, 'backsole_r_pitch': 2, 'backsole_r_roll': 2,
                   'hip_l_roll': 1.2, 'hip_l_yaw': 1, 'thigh_l_pitch': 2.5, 'shins_l_pitch': 4, 'backsole_l_pitch': 2, 'backsole_l_roll': 2
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = {'waist_yaw': 0.1, 'waist_roll': 0.1,
                        'hip_r_roll': 0.1, 'hip_r_yaw': 0.1, 'thigh_r_pitch': 0.35, 'shins_r_pitch': 0.5, 'backsole_r_pitch': 0.3,  'backsole_r_roll': 0.2,
                        'hip_l_roll': 0.1, 'hip_l_yaw': 0.1, 'thigh_l_pitch': 0.35, 'shins_l_pitch': 0.5, 'backsole_l_pitch': 0.3,  'backsole_l_roll': 0.2
                        }

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(Wk4RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/CL201/urdf/CL201.urdf'
        name = "CL201"
        foot_name = 'backsole_roll'
        terminate_after_contacts_on = ['base_link', 'Link_waist']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter


    class rewards(Wk4RoughCfg.rewards):
        class item:
            lin_Vel = GaussianRewardParam(0.2, 0.4, 0)
            ang_Vel = GaussianRewardParam(0.1, 0.4, 0)
            bRot = GaussianRewardParam(0.1, 0.1, 0)
            bTwist = GaussianRewardParam(0.05, 0.4, 0)
            bHgt = GaussianRewardParam(0.1, 0.0707, 0)
            cotr = CauchyRewardParam(0.05, 0.3, 1, 0)
            eVel = CauchyRewardParam(0.12, 0.5, 1, 0)
            eFrc = CauchyRewardParam(0.12, 0.2, 1, 0)
            ipct = CauchyRewardParam(0.05, 0.16, 3, 0)
            smth = CauchyRewardParam(0.05, 0.6, 1, 0)
            jVel = CauchyRewardParam(0.06, 12, 2, 0)
        base_height_target = 1.2

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

        whole_body_dof_order = [ 'J_waist_yaw', 'J_waist_roll',
            'J_hip_r_roll', 'J_hip_r_yaw', 'J_thigh_r_pitch', 'J_shins_r_pitch', 'J_backsole_r_pitch', 'J_backsole_r_roll',
            'J_hip_l_roll', 'J_hip_l_yaw', 'J_thigh_l_pitch', 'J_shins_l_pitch', 'J_backsole_l_pitch', 'J_backsole_l_roll']

        obs_dof_order = [
            'J_hip_r_roll', 'J_hip_r_yaw', 'J_thigh_r_pitch', 'J_shins_r_pitch', 'J_backsole_r_pitch', 'J_backsole_r_roll',
            'J_hip_l_roll', 'J_hip_l_yaw', 'J_thigh_l_pitch', 'J_shins_l_pitch', 'J_backsole_l_pitch', 'J_backsole_l_roll']

        action_dof_order = [
            'J_hip_r_roll', 'J_hip_r_yaw', 'J_thigh_r_pitch', 'J_shins_r_pitch', 'J_backsole_r_pitch', 'J_backsole_r_roll',
            'J_hip_l_roll', 'J_hip_l_yaw', 'J_thigh_l_pitch', 'J_shins_l_pitch', 'J_backsole_l_pitch', 'J_backsole_l_roll']


class CL201LocoCfgPPO(Wk4RoughCfgPPO):
    seed = time.time()

    class runner(Wk4RoughCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'

        max_iterations = 10000  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'CL201_test'
        experiment_name = 'CL201_blind'

        save_interval = 100  # check for potential saves every this many iterations
        resume = False
        load_run = '240708_090625_key2_exp'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/CL201_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wukong4_leg.py"]

