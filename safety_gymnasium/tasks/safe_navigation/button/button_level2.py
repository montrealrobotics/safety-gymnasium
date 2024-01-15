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
"""Button task 2."""
from safety_gymnasium.assets.geoms import Hazards, Pillars
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.assets.free_geoms import Vases, PushBox
from safety_gymnasium.tasks.safe_navigation.button.button_level1 import ButtonLevel1


class ButtonLevel2(ButtonLevel1):
    """An agent must press a goal button while avoiding more hazards and gremlins.

    And while not pressing any of the wrong buttons.
    """

    def __init__(self, config, reward_goal=1., reward_distance=1., num_steps=1000, action_noise=0.) -> None:
        super().__init__(config=config, reward_goal=reward_goal, reward_distance=reward_distance, num_steps=num_steps, action_noise=action_noise)

        self.placements_conf.extents = [-1.8, -1.8, 1.8, 1.8]

        self._add_geoms(Hazards(num=4, keepout=0.18))
        self._add_mocaps(Gremlins(num=4, travel=0.35, keepout=0.4))
        self._add_free_geoms(Vases(num=1, is_constrained=False))
        self._add_geoms(Pillars(num=1, is_constrained=False))
        self._add_free_geoms(PushBox(null_dist=0))
        self.buttons.is_constrained = True  # pylint: disable=no-member
