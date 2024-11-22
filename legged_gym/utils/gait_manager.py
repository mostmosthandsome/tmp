# created on 2023/09/07
# 0907: 最简版本，暂不考虑变步态的情况和四足适应性，仿人能用就行
# 2023/09/11 (willw): Add quadruped compatibility and phase swapping function, optimize running speed
import random
import numpy as np
import collections

import torch

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class GaitParam:
    frequency: float
    swing_ratio: float
    offset: list
    symmetry: str

    def __init__(self, freq: float = 2, sw_rate: float = 0.5, ofs=None, sym: str = "NONE"):
        if ofs is None:
            ofs = [0, 0, 0, 0]
        self.frequency = freq
        self.offset = ofs
        self.symmetry = sym
        self.swing_ratio = sw_rate


quadruped_gait_library = {
    "pronk": GaitParam(freq=2, sw_rate=0.7, ofs=[0, 0, 0, 0], sym="None"),
    "pace": GaitParam(freq=2, sw_rate=0.5, ofs=[0, 0.5, 0, 0.5], sym="SAGITTAL"),
    "trot": GaitParam(freq=2, sw_rate=0.5, ofs=[0, 0.5, 0.5, 0], sym="CORONAL"),
    "walk": GaitParam(freq=1.2, sw_rate=0.25, ofs=[0, 0.25, 0.75, 0.5], sym="DOUBLE_SPIRAL"),
    "gallop": GaitParam(freq=2.5, sw_rate=0.75, ofs=[0, 0.2, 0.6, 0.8], sym="DOUBLE_SPIRAL"),
    "bound": GaitParam(freq=2, sw_rate=0.5, ofs=[0, 0, 0.5, 0.5], sym="SAGITTAL"),
    "tripod": GaitParam(freq=2, sw_rate=0.33, ofs=[0, 0.333, 0.667, 0], sym="DOUBLE_SPIRAL")
}

humanoid_gait_library = {
    "leap": GaitParam(freq=1.8, sw_rate=0.25, ofs=[0, 0], sym="NONE"),
    "walk": GaitParam(freq=1.2, sw_rate=0.5, ofs=[0, 0.5], sym="CORONAL"),
    "stance_walk": GaitParam(freq=1.2, sw_rate=0.45, ofs=[0, 0.5], sym="CORONAL"),
    "run": GaitParam(freq=2, sw_rate=0.65, ofs=[0, 0.5], sym="CORONAL"),
    "stand": GaitParam(freq=0., sw_rate=1e-8, ofs=[0.5, 0.5], sym="NONE")
}


def piecewise_2var(x, r, condlist, funclist, *args, **kw):
    """
    This function is derived from numpy's original numpy.piecewise()
    x : ndarray or scalar
        The input domain.
    condlist : list of bool arrays or bool scalars
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x)` is used as the output value.

        Each boolean array in `condlist` selects a piece of `x`,
        and should therefore be of the same shape as `x`.

        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if
        ``len(funclist) == len(condlist) + 1``, then that extra function
        is the default value, used wherever all conditions are false.
    funclist : list of callables, f(x,*args,**kw), or scalars
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take a 1d array as input and give an 1d
        array or a scalar value as output.  If, instead of a callable,
        a scalar is provided then a constant function (``lambda x: scalar``) is
        assumed.
    args : tuple, optional
        Any further arguments given to `piecewise` are passed to the functions
        upon execution, i.e., if called ``piecewise(..., ..., 1, 'a')``, then
        each function is called as ``f(x, 1, 'a')``.
    kw : dict, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(..., ..., alpha=1)``, then each function is called as
        ``f(x, alpha=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have a default value of 0.
    """
    x = np.asanyarray(x)
    n2 = len(funclist)

    # undocumented: single condition is promoted to a list of one condition
    if np.isscalar(condlist) or (
            not isinstance(condlist[0], (list, np.ndarray)) and x.ndim != 0):
        condlist = [condlist]

    condlist = np.asarray(condlist, dtype=bool)
    n = len(condlist)

    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected"
            .format(n, n, n + 1)
        )

    y = np.zeros_like(x)
    for cond, func in zip(condlist, funclist):
        if not isinstance(func, collections.abc.Callable):
            y[cond] = func
        else:
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, r[cond], *args, **kw)

    return y


class GaitManager:
    """
    Gait signal generator. Also is in charge of computing the gait reward coefficients.
    """
    PhaseTypes = ['RAMP', 'BALANCED_SINE', 'ADAPTIVE_SINE', 'STEP']
    Symmetries = ['NONE', 'SAGITTAL', 'CORONAL', 'Z_AXIAL', 'CENTRAL', 'SPIRAL', 'DOUBLE_SPIRAL']

    # * Definition of gait symmetry
    # * NONE: Cannot do any mirroring or spinning
    # * SAGITTAL: Can do Left/Right mirroring
    # * CORONAL: Can Front/Back mirroring
    # * Z_AXIAL: Can do both SAGITTAL and CORONAL flip
    # * CENTRAL: Can spin 180deg
    # * SPIRAL: Can spin any x*90deg clock
    # * DOUBLE_SPIRAL:  Can do both SPIRAL and SAGITTAL

    num_legs: int = 2
    num_robots: int = 1
    symmetry: str = "NONE"
    # swingRatio: float = 0
    contactTolerance: float = 0
    signalType: str = ""

    def __init__(self, cfg: LeggedRobotCfg.gait, num_robots: int = 1, num_legs: int = 2, dt: float = 1e-2) -> None:
        """
        Constructor of GaitManager.
        :param cfg: config class from LeggedRobotCfg, should contain gait name, contactTolerance and state_type
        :param num_robots: number of parallel environments
        :param num_legs: 2:humanoid. 4:quadruped
        :param dt: how much time would move forward for every self.run()
        """
        self.cfg = cfg
        self.num_robots = num_robots
        self.num_legs = num_legs
        self.time_step = dt
        self.phaseVal = np.zeros([self.num_robots, 1], dtype=float)
        self.transitionCountDown = np.zeros([self.num_robots, 1], dtype=int)
        self.isStance = np.ones([self.num_robots, self.num_legs], dtype=bool)
        self.onStartingUp = np.zeros([self.num_robots, self.num_legs], dtype=bool)
        self.onStopping = np.zeros([self.num_robots, self.num_legs], dtype=bool)
        self.footPhases = np.zeros([self.num_robots, self.num_legs], dtype=float)
        self.offset = np.zeros([self.num_robots, self.num_legs], dtype=float)
        self.swingRatio = np.ones([self.num_robots, self.num_legs], dtype=float) / 2
        self.frequency = np.ones([self.num_robots, self.num_legs], dtype=float)
        self.swap_offset = np.zeros([self.num_robots], dtype=bool)

        self.frequency_range = None
        self.swing_ratio_range = None
        self.init_offset = None

        self.load_config()

        self.rwd_coeff_condf = [lambda x, r: x <= self.contactTolerance,
                                lambda x, r: x >= 1 - self.contactTolerance,
                                lambda x, r: abs(x - r) <= self.contactTolerance,
                                lambda x, r: np.logical_and(x <= 1 - self.contactTolerance,
                                                            r + self.contactTolerance <= x)
                                ]
        self.rwd_coeff_funcs = [lambda x, r: (0.5 + x / 2 / self.contactTolerance),
                                lambda x, r: (0.5 + (x - 1) / 2 / self.contactTolerance),
                                lambda x, r: (0.5 - (x - r) / 2 / self.contactTolerance),
                                0,
                                1
                                ]
        self.adaptive_sig_condf = [lambda x, r: x >= r]
        self.adaptive_sig_funcs = [lambda x, r: 1 + (x - 1) / 2 / (1 - r), lambda x, r: x / 2 / r]

    def _generate_sample_phase(self):
        self.phaseVal[:, 0] = np.mod(np.arange(self.num_robots) * self.time_step * self.frequency[:, 0], 1.0)
        self.footPhases = np.mod(self.offset + self.phaseVal, 1.0)

    def load_config(self):
        if self.num_legs == 2:
            gait_preset = humanoid_gait_library[self.cfg.name]
        else:
            gait_preset = quadruped_gait_library[self.cfg.name]
        self.symmetry = gait_preset.symmetry
        self.init_offset = np.array(gait_preset.offset, dtype=float)
        self.offset = np.tile(np.array(gait_preset.offset, dtype=float), (self.num_robots, 1))
        self.contactTolerance = self.cfg.contactTolerance
        if self.cfg.state_type.upper() in self.PhaseTypes:
            self.signalType = self.cfg.state_type.upper()
        else:
            self.signalType = 'BALANCED_SINE'
            print("[GaitManager.load_config] Unspecified Gait Signal Type, using balanced sine wave mode.")

        if self.cfg.frequency is None or self.cfg.frequency == "default":
            self.frequency[:, :] = gait_preset.frequency
        elif isinstance(self.cfg.frequency, list):
            self.frequency_range = self.cfg.frequency
            self.generate_random_frequency()
        elif np.isscalar(self.cfg.frequency):
            self.frequency[:, :] = self.cfg.frequency

        if self.cfg.swingRatio is None or self.cfg.swingRatio == "default":
            self.swingRatio[:, :] = gait_preset.swing_ratio
        elif isinstance(self.cfg.swingRatio, list):
            self.swing_ratio_range = self.cfg.swingRatio
            self.generate_random_swing_ratio()
        elif np.isscalar(self.cfg.swingRatio):
            self.swingRatio[:, :] = self.cfg.swingRatio
        self.swingRatio = np.clip(self.swingRatio, a_min=0, a_max=1)

    def reset(self, env_ids):
        """
        Reset all instant member variables. Called when the environment is calling reset().
        If the gait has symmetry features, this method will randomly flip it before usage.
        :param env_ids: a tensor with multiple entries, containing index of all envs whose dones==True
        :return:
        """
        if len(env_ids) == 0:
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.cpu().numpy()
        self.phaseVal[env_ids] = 0.
        self.transitionCountDown[env_ids, :] = 0
        self.isStance[env_ids, :] = np.ones(self.num_legs, dtype=bool)
        self.onStartingUp[env_ids, :] = np.zeros(self.num_legs, dtype=bool)
        self.onStopping[env_ids, :] = np.zeros(self.num_legs, dtype=bool)
        self.footPhases[env_ids, :] = 0

        if self.frequency_range is not None:
            self.generate_random_frequency(env_ids=env_ids)
        if self.swing_ratio_range is not None:
            self.generate_random_swing_ratio(env_ids=env_ids)

        # TODO: Random gait type and frequency
        # TODO: Tripod gait, abnormal leg selection

        # swap offset
        for env_id in env_ids:
            if self.symmetry == "SAGITTAL" or self.symmetry == "Z_AXIAL" or self.symmetry == "DOUBLE_SPIRAL":
                if random.random() < 0.5:
                    new_offset = np.zeros(self.num_legs)
                    new_offset[::2], new_offset[1::2] = self.offset[env_id, 1::2], self.offset[env_id, ::2]
                    self.offset[env_id, :] = new_offset
            if self.symmetry == "CORONAL" or self.symmetry == "Z_AXIAL":
                if random.random() < 0.5:
                    new_offset = np.zeros(self.num_legs)
                    new_offset[:self.num_legs // 2], new_offset[self.num_legs // 2:] = (
                        self.offset[env_id, self.num_legs // 2:], self.offset[env_id, :self.num_legs // 2])
                    self.offset[env_id, :] = new_offset
            if self.symmetry == "CENTRAL":
                if random.random() < 0.5:
                    new_offset = np.array([self.offset[env_id, i + self.num_legs // 2] for i in range(self.num_legs)])
                    self.offset[env_id, :] = new_offset
            if self.symmetry.endswith("SPIRAL") and self.num_legs == 4:
                for i in range(int(np.floor(random.random() * self.num_legs))):
                    new_offset = self.offset[env_id, [2, 0, 3, 1]]
                    self.offset[env_id, :] = new_offset
            if self.symmetry == "ZX_AXIAL":
                if random.random() < 0.5:
                    new_offset = self.offset[env_id, [1, 0, 3, 2]]
                    self.offset[env_id, :] = new_offset
            comp =  self.offset[env_id, :] == self.init_offset
            self.swap_offset[env_id] = False if comp.all() else True
            # print("offset:", self.offset)
            # print("comp:", comp)
            # print("flag:", self.swap_offset)

    def run(self, cmd: np.ndarray = None):
        """

        Step forward using determined gait parameters.
        Should be called every timestep of simulation.
        :param cmd:
        :return:
        """

        if cmd is not None:
            vel_present = np.repeat(np.expand_dims(np.abs(np.linalg.norm(cmd[:, :3], axis=-1)) > 0.1732, 1),
                                    self.num_legs,
                                    1)

            self.onStartingUp[~vel_present & self.onStartingUp] = False
            self.onStopping[vel_present & self.onStopping] = False

            self.onStartingUp[vel_present & self.isStance] = True
            self.onStopping[~(vel_present | self.isStance)] = True
        else:
            self.onStartingUp[self.isStance] = True
            self.onStopping[:, :] = False

        prev_footPhase = np.fmod(self.offset + self.phaseVal, 1.)
        self.phaseVal = np.fmod(self.phaseVal + self.time_step * self.frequency, 1.)
        self.footPhases = np.fmod(self.offset + self.phaseVal, 1.)

        stance_center = 0.5 + self.swingRatio / 2
        on_stance_center = (self.footPhases >= stance_center) & (prev_footPhase < stance_center)

        self.isStance &= ~(self.onStartingUp & on_stance_center)
        self.onStartingUp &= ~on_stance_center

        self.isStance |= self.onStopping & on_stance_center
        self.onStopping &= ~on_stance_center

        self.footPhases[self.isStance] = stance_center[self.isStance]

        swing_center = self.swingRatio / 2
        sw_arm_idx = np.where(self.offset == -1)
        self.footPhases[sw_arm_idx] = swing_center[sw_arm_idx]


        # TODO: Gait transition, offset blending
        # TODO: Tripod gait, abnormal leg phase maintaining

    def set_frequency(self, new_freq, env_ids=None):
        if env_ids is None:
            self.frequency[:, :] = new_freq
        else:
            self.frequency[env_ids, :] = new_freq

    def set_swing_ratio(self, new_ratio, env_ids=None):
        if env_ids is None:
            self.swingRatio[:, :] = new_ratio
        else:
            self.swingRatio[env_ids, :] = new_ratio

    def generate_random_frequency(self, env_ids=None):
        new_freq = np.random.uniform(size=(self.num_robots, 1) if env_ids is None else (len(env_ids), 1),
                                     low=self.frequency_range[0],
                                     high=self.frequency_range[1])
        new_foot_freq = np.tile(new_freq, reps=(1, self.num_legs))
        self.set_frequency(new_freq=new_foot_freq, env_ids=env_ids)

    def generate_random_swing_ratio(self, env_ids=None):
        new_ratio = np.random.uniform(size=(self.num_robots, 1) if env_ids is None else (len(env_ids), 1),
                                      low=self.swing_ratio_range[0],
                                      high=self.swing_ratio_range[1])
        new_foot_ratio = np.tile(new_ratio, reps=(1, self.num_legs))
        self.set_swing_ratio(new_ratio=new_foot_ratio, env_ids=env_ids)

    def get_frc_penalty_coeff(self):
        """
        Calculate force penalty coeff, speed penalty coeff doesn't need to be calculated,
        because their sum is 1
        Use numpy.piecewise() for speeding up larger-scale computation
        numpy.piecewise() is faster than simple for loop and numpy ufunc (numpy.frompyfunc())
        :return: np.ndarray with shape [num_robots, 2]
        """

        rwd_coeff_conds = [f(self.footPhases, self.swingRatio) for f in self.rwd_coeff_condf]
        frc_coeff = piecewise_2var(self.footPhases, self.swingRatio, rwd_coeff_conds, self.rwd_coeff_funcs)
        check = np.isnan(frc_coeff) | np.isinf(frc_coeff)
        if np.sum(check) > 0:
            print("Wrong frc Rwd Weight in env:", np.unique(np.where(check[0])))
        # print(self.phaseVal[:2,:], self.footPhases[:2,:], frc_coeff[:2, :])
        return frc_coeff

    def get_phase_states(self):
        if self.signalType == 'RAMP':
            states = self.footPhases
        elif self.signalType.endswith("SINE"):
            states = np.zeros([self.num_robots, self.num_legs * 2], dtype=float)
            if self.signalType.startswith("ADAPTIVE"):
                adaptive_sig_conds = [f(self.footPhases, self.swingRatio) for f in self.adaptive_sig_condf]
                internal_phase = piecewise_2var(self.footPhases, self.swingRatio, adaptive_sig_conds,
                                                self.adaptive_sig_funcs) * 2 * np.pi
            else:
                internal_phase = self.footPhases * 2 * np.pi
            states[:, ::2] = np.sin(internal_phase)
            states[:, 1::2] = np.cos(internal_phase)
        elif self.signalType == 'STEP':
            states = piecewise_2var(self.footPhases, self.swingRatio, [self.footPhases < self.swingRatio], [1, -1])
        else:
            states = None
            print("[GaitManager.get_state] Gait signal type unspecified. Might be changed half-way.")
        return states

    def get_swap_offset(self):
        return self.swap_offset
