from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, BaseConfig
from legged_gym.envs.WukongIV.wk4_leg_config import Wk4RoughCfg, Wk4RoughCfgPPO
import time

from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam


class HuV3LocoCfg(Wk4RoughCfg):
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

    class terrain(Wk4RoughCfg.terrain):
        # mesh_type = "plane"
        mesh_type = "trimesh"
    
    class commands(Wk4RoughCfg.commands):
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(Wk4RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.4, 0.4]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            heading = [-1.57, 1.57]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.86]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_hip_yaw_joint": 0.08,
            "left_hip_roll_joint": 0.06,
            "left_hip_pitch_joint": -0.42,
            "left_knee_joint": 0.94,
            "left_ankle_joint": -0.52,
            "left_toe_joint": -0.06,
            "right_hip_yaw_joint": -0.08,
            "right_hip_roll_joint": -0.06,
            "right_hip_pitch_joint": -0.42,
            "right_knee_joint": 0.94,
            "right_ankle_joint": -0.52,
            "right_toe_joint": 0.06,
            "torso_joint": 0,
            "left_shoulder_pitch_joint": 0.38,
            "left_shoulder_roll_joint": 0,
            "left_shoulder_yaw_joint": 0,
            "left_elbow_pitch_joint": 0.45,
            "left_elbow_roll_joint": 0,
            "left_wrist_pitch_joint": 0,
            "left_wrist_yaw_joint": 0,
            "right_shoulder_pitch_joint": 0.38,
            "right_shoulder_roll_joint": 0,
            "right_shoulder_yaw_joint": 0,
            "right_elbow_pitch_joint": 0.45,
            "right_elbow_roll_joint": 0,
            "right_wrist_pitch_joint": 0,
            "right_wrist_yaw_joint": 0,
            "zneck_joint": 0
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {"hip_yaw": 120,  "hip_roll": 100,  "hip_pitch": 200,  "knee": 200,  "ankle": 80,  "toe": 80,
                     "torso": 800,  "shoulder": 180,  "elbow_pitch": 180,  "elbow_roll": 60,  "wrist": 60,  "zneck": 400
                     }  # [N*m/rad]
        # orgin PD of ankle: P: 'Ankle_Y': 50, 'Ankle_X': 50; D:'Ankle_Y': 0.5, 'Ankle_X': 0.3
        damping = {"hip_yaw": 1, "hip_roll": 1, "hip_pitch": 3.2, "knee": 4, "ankle": 2, "toe": 1,
                        "torso": 16, "shoulder": 3, "elbow_pitch": 3, "elbow_roll": 3, "wrist": 1.2, "zneck": 6
                   }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = {"hip_yaw": 0.2, "hip_roll": 0.25, "hip_pitch": 0.5, "knee": 0.7, "ankle": 0.4, "toe": 0.2,
                        "torso": 0.2, "shoulder": 0.2, "elbow_pitch": 0.3, "elbow_roll": 0.2, "wrist": 0.2, "zneck": 0.6 }

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(Wk4RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/HuV3/hu_v3.urdf'
        name = "HuV3"
        foot_name = 'toe'
        terminate_after_contacts_on = ['torso_link', 'hip_pitch_link']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter


    class rewards(Wk4RoughCfg.rewards):
        class item:
            lin_Vel = GaussianRewardParam(0.2, 0.4, 0)
            ang_Vel = GaussianRewardParam(0.15, 0.4, 0)
            bRot = GaussianRewardParam(0.12, 0.1, 0)
            bTwist = GaussianRewardParam(0.05, 0.4, 0)
            bHgt = GaussianRewardParam(0.03, 0.1, 0)
            cotr = CauchyRewardParam(0.05, 0.3, 1, 0)
            eVel = CauchyRewardParam(0.12, 0.5, 1, 0)
            eFrc = CauchyRewardParam(0.12, 0.2, 1, 0)
            ipct = CauchyRewardParam(0.0, 0.16, 3, 0)
            smth = CauchyRewardParam(0.1, 0.6, 1, 0)
            jVel = CauchyRewardParam(0.06, 8, 2, 0)
        base_height_target = 0.825

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
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint", "left_toe_joint",
             "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint", "right_toe_joint",
             "torso_joint",
             "left_shoulder_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_elbow_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
             "right_shoulder_pitch_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
            "right_elbow_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
             "zneck_joint"]

        obs_dof_order = [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint", "left_toe_joint",
             "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint", "right_toe_joint"]

        action_dof_order = [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint", "left_toe_joint",
             "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint", "right_toe_joint"]


class HuV3LocoCfgPPO(Wk4RoughCfgPPO):
    seed = time.time()

    class runner(Wk4RoughCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'

        max_iterations = 10000  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'less_hgt_rough'
        experiment_name = 'HU_blind'

        save_interval = 100  # check for potential saves every this many iterations
        resume = False
        load_run = '240708_090625_key2_exp'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/HuV3_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wukong4_leg.py"]

