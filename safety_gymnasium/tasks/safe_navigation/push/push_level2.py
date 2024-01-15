# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Push level 2."""
from safety_gymnasium.assets.geoms import Hazards, Pillars
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.assets.free_geoms import Vases, PushBox
from safety_gymnasium.assets.geoms import Buttons, Goal
from safety_gymnasium.tasks.safe_navigation.push.push_level1 import PushLevel1


class PushLevel2(PushLevel1):
    """An agent must push a box to a goal while avoiding more hazards and pillars."""

    def __init__(self, config, reward_goal, reward_distance, num_steps=1000, action_noise=0.) -> None:
        super().__init__(config=config, reward_goal=reward_goal, reward_distance=reward_distance)

        self.placements_conf.extents = [-1.8, -1.8, 1.8, 1.8]
        self._add_geoms(Hazards(num=4, keepout=0.18))
        self._add_mocaps(Gremlins(num=4, travel=0.35, keepout=0.4))
        self._add_free_geoms(Vases(num=1, is_constrained=False))
        self._add_geoms(Pillars(num=1, is_constrained=False))
        self._add_geoms(Buttons(num=4, is_constrained=True,))
