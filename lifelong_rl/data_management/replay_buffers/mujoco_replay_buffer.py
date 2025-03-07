import numpy as np

import copy

from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.util.visualize_mujoco import visualize_mujoco_from_states


class MujocoReplayBuffer(EnvReplayBuffer):

    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )

        self.body_xpos_shape = env.sim.data.body_xpos.shape
        self._body_xpos = np.zeros((max_replay_buffer_size, *self.body_xpos_shape))

        self.qpos_shape = env.sim.data.qpos.shape
        self._qpos = np.zeros((max_replay_buffer_size, *self.qpos_shape))

        self.env_states = []
        self.desired_goals = []

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, env_state, desired_goal = None, **kwargs):
        self._body_xpos[self._top] = self.env.sim.data.body_xpos
        self._qpos[self._top] = self.env.sim.data.qpos
        # if len(self.env_states) >= self.max_replay_buffer_size():
        #     self.env_states[self._top] = self.env.sim.get_state()
        # else:
        #     self.env_states.append(copy.deepcopy(self.env.sim.get_state()))
        if len(self.env_states) >= self.max_replay_buffer_size():
            self.env_states[self._top] = env_state
        else:
            self.env_states.append(copy.deepcopy(env_state))

        if len(self.desired_goals) >= self.max_replay_buffer_size():
            self.desired_goals[self._top] = desired_goal
        else:
            self.desired_goals.append(copy.deepcopy(desired_goal))


        # print(env_state, self.env_states)

        return super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def get_snapshot(self):
        # print("hoge", self.env.sim.get_state())
        snapshot = super().get_snapshot()
        snapshot.update(dict(
            body_xpos=self._body_xpos[:self._size],
            qpos=self._qpos[:self._size],
            env_states=self.env_states[:self._size],
            desired_goals = self.desired_goals[:self._size],
        ))
        return snapshot

    def visualize_agent(self, start_idx, end_idx):
        visualize_mujoco_from_states(self.env, self.env_states[start_idx:end_idx])
