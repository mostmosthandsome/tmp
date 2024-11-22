# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .lite3.lite3_config import Lite3RoughCfg, Lite3RoughCfgPPO
from .WukongIV.wukong4_leg import WukongLegbase
from .WukongIV.wk4_leg_config import Wk4RoughCfg, Wk4RoughCfgPPO
from .WukongIV.Wukong4Visual import Wukong4Visual
from .WukongIV.wk4_vis_config import Wukong4VisualCfg, Wukong4VisualCfgPPO
from .WukongIV.h1_leg_config import H1LocoCfg, H1LocoCfgPPO
from .WukongIV.Dr01_leg_config import Dr01LocoCfg, Dr01LocoCfgPPO
from .WukongIV.CL201_leg_config import CL201LocoCfg, CL201LocoCfgPPO
from .WukongIV.GR1T2_leg_config import GR1T2LocoCfg, GR1T2LocoCfgPPO
from .WukongIV.HuV3_leg_config import HuV3LocoCfg, HuV3LocoCfgPPO
from .WukongIV.dancer_leg_config import DancerLocoCfgPPO, DancerLocoCfg
from .lite3.lite3_gait import Lite3Gait
from .lite3.lite3_gait_config import Lite3GaitCfg, Lite3GaitCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register("lite3", LeggedRobot, Lite3RoughCfg(), Lite3RoughCfgPPO())
task_registry.register("wk4_leg", WukongLegbase, Wk4RoughCfg(), Wk4RoughCfgPPO())
task_registry.register("dr01_leg", WukongLegbase, Dr01LocoCfg(), Dr01LocoCfgPPO())
task_registry.register("cl201_leg", WukongLegbase, CL201LocoCfg(), CL201LocoCfgPPO())
task_registry.register("gr1t2_leg", WukongLegbase, GR1T2LocoCfg(), GR1T2LocoCfgPPO())
task_registry.register("h1_leg", WukongLegbase, H1LocoCfg(), H1LocoCfgPPO())
task_registry.register("HuV3_leg", WukongLegbase, HuV3LocoCfg(), HuV3LocoCfgPPO())
task_registry.register("Dancer_leg", WukongLegbase, DancerLocoCfg(), DancerLocoCfgPPO())
task_registry.register("wk4_vis", Wukong4Visual, Wukong4VisualCfg(), Wukong4VisualCfgPPO())
task_registry.register("lite3_gait", Lite3Gait, Lite3GaitCfg(), Lite3GaitCfgPPO())
