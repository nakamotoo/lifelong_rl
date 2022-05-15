import numpy as np
import gtimer as gt
import copy
from collections import OrderedDict

from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.util.eval_util import create_stats_ordered_dict


class PPOHERTrainer(TorchTrainer):

    """
    Hindsight Experience Replay (Andrychowicz et al. 2017).
    Duplicates transitions using different goals with particular reward function.
    """

    def __init__(
            self,
            policy_trainer,
            replay_buffer,
            replay_size,
            replay_k = 1,
            path_len = None,
            reward_scale=1.,             # Typically beneficial to multiply rewards
            reward_bounds=(-10, 10),     # Clip intrinsic rewards for stability
    ):
        super().__init__()

        self.policy_trainer = policy_trainer
        self.replay_buffer = replay_buffer
        self._ptr = 0
        self._cur_replay_size = 0
        self.replay_size = replay_size
        self.relabel_prob = 1 - (1. / (1 + replay_k)) # あるサンプルに対してどれくらいの確率でrelabelを実行するか

        self.obs_dim = replay_buffer.obs_dim()
        self.action_dim = replay_buffer.action_dim()

        self._obs = np.zeros((replay_size, self.obs_dim))
        self._next_obs = np.zeros((replay_size, self.obs_dim))
        self._actions = np.zeros((replay_size, self.action_dim))
        self._rewards = np.zeros((replay_size, 1))
        self._terminals = np.zeros((replay_size, 1))
        self._oracle_rewards = np.zeros((replay_size, 1))

        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        self.path_len = path_len
        self.reward_scale = reward_scale
        self.reward_bounds = reward_bounds

    def add_sample(self, obs, next_obs, action, reward, oracle_reward, terminal):
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._oracle_rewards[self._ptr] = oracle_reward
        self._terminals[self._ptr] = terminal

        self._ptr = (self._ptr + 1) % self.replay_size
        self._cur_replay_size = min(self._cur_replay_size+1, self.replay_size)

    def train_from_paths(self, paths):

        """
        Path processing
        """

        paths = copy.deepcopy(paths)
        for path in paths:
            obs, next_obs = path['observations'], path['next_observations']
            achieved_goals = path['achieved_goals']
            actions = path['actions']
            terminals = path['terminals']  # this is probably always False, but might want it?
            path_len = len(obs)
            relabeled_desired_goals = []
            oracle_rewards = path["rewards"]

            if self.policy_trainer.env.reward_type == "initial_distance_sparse":
                oracle_rewards[:-1] = 0

            # Relabel goals
            # for i in range(path_len):
            #     goal_ind = np.random.randint(i, path_len)
            #     obs[i, -3:] = achieved_goals[goal_ind]
            #     next_obs[i, -3:] = achieved_goals[goal_ind]
            #     relabeled_desired_goals.append(achieved_goals[goal_ind])
            # relabeled_desired_goals = np.array(relabeled_desired_goals)


            # relabel rewards |next
            # 距離がある程度近づいたら1, それじゃないなら0のバイナリreward
            # distance = np.linalg.norm(achieved_goals - relabeled_desired_goals, axis=-1)
            # relabeled_rewards = -(distance > 0.05).astype(np.float32)

            for t in range(path_len):
                self.add_sample(
                    obs[t],
                    next_obs[t],
                    actions[t],
                    # reward = relabeled_rewards[t],
                    reward = oracle_rewards[t],
                    oracle_reward = oracle_rewards[t],
                    terminal = terminals[t]
                )
        gt.stamp('policy training', unique=False)
        self.train_ppo()

    def reward_postprocessing(self, rewards):
        # Some scaling of the rewards can help; it is very finicky though
        rewards = rewards * self.reward_scale
        rewards = np.clip(rewards, *self.reward_bounds)  # stabilizes training
        return rewards


    def train_ppo(self):

        oracle_rewards = self._oracle_rewards[:self._cur_replay_size].squeeze()
        rewards = self._rewards[:self._cur_replay_size]
        rewards = self.reward_postprocessing(rewards)
        ppo_paths = [{
            "observations": self._obs[:self._cur_replay_size],
            "next_observations": self._next_obs[:self._cur_replay_size],
            "actions": self._actions[:self._cur_replay_size],
            "rewards": rewards,
            "terminals": self._terminals[:self._cur_replay_size],
        }]

        self.policy_trainer.train_from_paths(ppo_paths, path_len = self.path_len)
        gt.stamp('policy training', unique=False)

        if self._need_to_update_eval_statistics:
            self.eval_statistics.update(self.policy_trainer.eval_statistics)

        self.eval_statistics.update(create_stats_ordered_dict(
                'Rewards (Processed)',
                rewards,
            ))

        self.eval_statistics.update(create_stats_ordered_dict(
                'Oracle Rewards',
                oracle_rewards,
            ))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.policy_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self.policy_trainer.networks

    def get_snapshot(self):
        snapshot = dict()
        for k, v in self.policy_trainer.get_snapshot().items():
            snapshot['policy_trainer/' + k] = v

        return snapshot
