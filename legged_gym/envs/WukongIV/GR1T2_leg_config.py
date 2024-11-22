from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, BaseConfig
from legged_gym.envs.WukongIV.wk4_leg_config import Wk4RoughCfg, Wk4RoughCfgPPO
import time

from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam


class GR1T2LocoCfg(Wk4RoughCfg):
    class env(Wk4RoughCfg.env):
        num_envs = 2048
        num_history_frames = 50
        history_stride = 1
        num_actions = 12
        num_observations = 9 + num_actions * 3 + 4
        num_observations_history = num_observations * num_history_frames
        episode_length_s = 20  # episode length in seconds
        # num_privileged_obs = num_observations + 3 + 1 + 18 + 9  # base vel, base height and foot height
        num_privileged_obs = num_observations + 3 + 1 + 18 + 81  # base vel, base height and foot height
        estimation_terms = ["vel"]

        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm

    class terrain(Wk4RoughCfg.terrain):
        measure_heights = True
        # mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        # measured_points_x = [-0.1, 0., 0.1]  # 1mx1m rectangle (without center line)
        # measured_points_y = [-0.1, 0., 0.1]
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]

    class commands(Wk4RoughCfg.commands):
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(Wk4RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.25, 0.25]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            heading = [-1.57, 1.57]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.85]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "right_hip_roll_joint": -0.08,
            "right_hip_yaw_joint": -0.02,
            "right_hip_pitch_joint": -0.37,
            "right_knee_pitch_joint": 0.94,
            "right_ankle_pitch_joint": -0.55,
            "right_ankle_roll_joint": -0.08,
            "left_hip_roll_joint": 0.08,
            "left_hip_yaw_joint": 0.02,
            "left_hip_pitch_joint": -0.37,
            "left_knee_pitch_joint": 0.94,
            "left_ankle_pitch_joint": -0.55,
            "left_ankle_roll_joint": -0.08,

            "left_shoulder_pitch_joint": 0.17,
            "left_shoulder_roll_joint": 0,
            "left_shoulder_yaw_joint": 0,
            "left_elbow_pitch_joint": -0.525,
            "right_shoulder_pitch_joint": 0.17,
            "right_shoulder_roll_joint": 0,
            "right_shoulder_yaw_joint": 0,
            "right_elbow_pitch_joint": -0.525,

            "right_wrist_yaw_joint": 0,
            "right_wrist_roll_joint": 0,
            "right_wrist_pitch_joint": 0,
            "left_wrist_yaw_joint": 0,
            "left_wrist_roll_joint": 0,
            "left_wrist_pitch_joint": 0,

            "waist_yaw_joint": 0,
            "waist_pitch_joint": 0,
            "waist_roll_joint": 0,

            "head_yaw_joint": 0,
            "head_pitch_joint": 0,
            "head_roll_joint": 0,

            "L_thumb_proximal_yaw_joint": 0,
            "L_thumb_proximal_pitch_joint": 0,
            "L_thumb_intermediate_joint": 0,
            "L_thumb_distal_joint": 0,
            "L_index_proximal_joint": 0,
            "L_index_intermediate_joint": 0,
            "L_middle_proximal_joint": 0,
            "L_middle_intermediate_joint": 0,
            "L_ring_proximal_joint": 0,
            "L_ring_intermediate_joint": 0,
            "L_pinky_proximal_joint": 0,
            "L_pinky_intermediate_joint": 0,

            "R_thumb_proximal_yaw_joint": 0,
            "R_thumb_proximal_pitch_joint": 0,
            "R_thumb_intermediate_joint": 0,
            "R_thumb_distal_joint": 0,
            "R_index_proximal_joint": 0,
            "R_index_intermediate_joint": 0,
            "R_middle_proximal_joint": 0,
            "R_middle_intermediate_joint": 0,
            "R_ring_proximal_joint": 0,
            "R_ring_intermediate_joint": 0,
            "R_pinky_proximal_joint": 0,
            "R_pinky_intermediate_joint": 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {"hip_roll": 120, "hip_yaw": 100, "hip_pitch": 200,
                     "knee_pitch": 200, "ankle_pitch": 50, "ankle_roll": 80,
                     "waist_yaw": 600, "waist_pitch": 600, "waist_roll": 600,
                     "head_yaw": 300, "head_pitch": 300, "head_roll": 300,
                     "shoulder_pitch": 200, "shoulder_roll": 200, "shoulder_yaw": 200, "elbow_pitch": 200,
                     "intermediate": 10, "proximal": 10, "distal": 10, "proximal_yaw": 10, "proximal_pitch": 10,
                     "wrist_yaw": 50, "wrist_roll": 10, "wrist_pitch": 10
                     }  # [N*m/rad]
        # orgin PD of ankle: P: 'Ankle_Y': 50, 'Ankle_X': 50; D:'Ankle_Y': 0.5, 'Ankle_X': 0.3
        damping = {"hip_roll": 1, "hip_yaw": 1, "hip_pitch": 3.2,
                   "knee_pitch": 4, "ankle_pitch": 1, "ankle_roll": 2,
                   "waist_yaw": 8, "waist_pitch": 8, "waist_roll": 8,
                   "head_yaw": 4, "head_pitch": 4, "head_roll": 4,
                   "shoulder_pitch": 4, "shoulder_roll": 4, "shoulder_yaw": 4, "elbow_pitch": 4,
                   "intermediate": 0.2, "proximal": 0.2, "distal": 0.2, "proximal_yaw": 0.2, "proximal_pitch": 0.2,
                   "wrist_yaw": 1, "wrist_roll": 0.1, "wrist_pitch": 0.1
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.5
        action_scale = {"hip_roll": 0.2, "hip_yaw": 0.15, "hip_pitch": 0.5,
                        "knee_pitch": 0.7, "ankle_pitch": 0.3, "ankle_roll": 0.2,
                        "waist_yaw": 0.1, "waist_pitch": 0.1, "waist_roll": 0.1,
                        "head_yaw": 0.1, "head_pitch": 0.1, "head_roll": 0.1,
                        "shoulder_pitch": 0.2, "shoulder_roll": 0.2, "shoulder_yaw": 0.2, "elbow_pitch": 0.4,
                        "intermediate": 0.1, "proximal": 0.1, "distal": 0.1, "proximal_yaw": 0.1, "proximal_pitch": 0.1,
                        "wrist_yaw": 0.2, "wrist_roll": 0.2, "wrist_pitch": 0.2 }

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5

    class asset(Wk4RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T2/urdf/GR1T2_limbs.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T2/urdf/GR1T2_inspire_hand.urdf'
        name = "GR1T2"
        foot_name = 'foot_roll_link'
        terminate_after_contacts_on = ['torso', 'thigh', 'base', 'waist', 'head', 'upper_arm']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(Wk4RoughCfg.rewards):
        class item:
            lin_Vel = GaussianRewardParam(0.2, 0.3, 0)
            ang_Vel = GaussianRewardParam(0.1, 0.4, 0)
            bRot = GaussianRewardParam(0.05, 0.1, 0)
            bTwist = GaussianRewardParam(0.05, 0.4, 0)
            bHgt = GaussianRewardParam(0.05, 0.0707, 0)
            cotr = CauchyRewardParam(0.05, 1.2, 1, 0)
            eVel = CauchyRewardParam(0.125, 0.25, 1, 0)
            eFrc = CauchyRewardParam(0.125, 0.2, 1, 0)
            # ipct = CauchyRewardParam(0.05, 0.16, 3, 0)
            # cnct_forces = CauchyRewardParam(0.05, 0.12, 3, 1)
            smth = CauchyRewardParam(0.1, 0.4, 1, 0)
            jVel = CauchyRewardParam(0.05, 10, 2, 0)
        base_height_target = 0.83

    class sim(Wk4RoughCfg.sim):
        dt = 0.002
        substeps = 3

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
            "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
            "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
            "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]

        obs_dof_order = [
            "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
            "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
            "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        ]

        action_dof_order = [
            "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
            "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
            "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        ]


class GR1T2LocoCfgPPO(Wk4RoughCfgPPO):
    seed = time.time()

    class runner(Wk4RoughCfgPPO.runner):
        policy_class_name = 'AsymActorCritic'
        algorithm_class_name = 'PPO_DW'

        max_iterations = 10000  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'higher_500Hz'
        experiment_name = 'GR1T2_blind'

        save_interval = 100  # check for potential saves every this many iterations
        resume = False
        load_run = '240708_090625_key2_exp'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/GR1T2_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wukong4_leg.py"]

