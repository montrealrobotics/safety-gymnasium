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
"""Button task 0."""

import gymnasium
import mujoco
import numpy as np

from safety_gymnasium.assets.geoms import Buttons, Goal
from safety_gymnasium.bases.base_task import BaseTask


# pylint: disable-next=too-many-instance-attributes
class ButtonLevel0(BaseTask):
    """An agent must press a goal button."""

    def __init__(self, config, reward_goal=1., reward_distance=1.) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]

        self._add_geoms(Buttons(num=4, is_constrained=False, reward_goal=reward_goal, reward_distance=reward_distance))
        self._add_geoms(Goal(size=self.buttons.size * 2, alpha=0.1))  # pylint: disable=no-member

        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = 0.0
        dist_goal = self.dist_goal()
        # pylint: disable-next=no-member
        reward += (self.last_dist_goal - dist_goal) * self.buttons.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.buttons.reward_goal  # pylint: disable=no-member

        return reward

    def specific_reset(self):
        """Reset the buttons timer."""
        self.buttons.timer = 0  # pylint: disable=no-member

    def specific_step(self):
        """Clock the buttons timer."""
        self.buttons.timer_tick()  # pylint: disable=no-member

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # pylint: disable-next=no-member
        assert self.buttons.num > 0, 'Must have at least one button.'
        self.build_goal_button()
        self.last_dist_goal = self.dist_goal()
        self.buttons.reset_timer()  # pylint: disable=no-member

    def build_goal_button(self):
        """Pick a new goal button, maybe with resampling due to hazards."""
        # pylint: disable-next=no-member
        self.buttons.goal_button = self.random_generator.choice(self.buttons.num)
        new_goal_pos = self.buttons.pos[self.buttons.goal_button]  # pylint: disable=no-member
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = new_goal_pos[:2]
        self._set_goal(new_goal_pos[:2])
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def obs(self):
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        all_lidar = None
        lidar_class = []
        obstacles = ["goal", "buttons", "gremlins", "hazards", "vases", "push_box", "pillars"]


        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                if obstacle.name == "buttons" and self.buttons.timer != 0:
                    lidar = np.zeros(self.lidar_conf.num_bins)
                else:
                    lidar = self._obs_lidar(obstacle.pos, obstacle.group)
                if obstacle.name == "goal":
                    obs["goal_lidar"] = lidar
                lidar = np.expand_dims(lidar, 1)
                all_lidar = lidar if all_lidar is None else np.hstack((all_lidar, lidar))
                lidar_class.append([obstacles.index(obstacle.name)]*lidar.shape[0])
                # del obs[obstacle.name+'_lidar']
            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos)

        all_lidar, lidar_class = np.array(all_lidar), np.transpose(np.array(lidar_class), (1,0))
        min_lidar_idx = np.argmax(all_lidar, axis=1)
        min_lidar_idx += np.array(list(range(0, np.array(all_lidar.shape).prod(), all_lidar.shape[1])))

        obs["joint_lidar"] = np.take(all_lidar, min_lidar_idx)
        obs["joint_lidar_class"] = np.take(lidar_class, min_lidar_idx)


        if self.observe_vision:
            obs['vision'] = self._obs_vision()

        # assert self.obs_info.obs_space_dict.contains(
        #     obs,
        # ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'

        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            # pylint: disable-next=no-member
            if any(n == f'button{self.buttons.goal_button}' for n in geom_names) and any(
                n in self.agent.body_info.geom_names for n in geom_names
            ):
                return True
        return False
