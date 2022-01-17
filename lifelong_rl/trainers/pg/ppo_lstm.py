import torch

from lifelong_rl.trainers.pg.pg import PGTrainer
import numpy as np
import torch.optim as optim

from collections import OrderedDict
import copy

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.envs.env_utils import get_dim
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.samplers.utils.path_functions as path_functions
import lifelong_rl.util.pythonplusplus as ppp
import sys


class PPOLSTMTrainer(PGTrainer):

    """
    Proximal Policy Optimization (Schulman et al. 2016).
    Policy gradient algorithm with clipped surrogate loss.
    """

    def __init__(
            self,
            ppo_epsilon=0.2,    # Epsilon for clipping
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ppo_epsilon = ppo_epsilon

    def policy_objective(self, obs, actions, advantages, old_policy):
        log_probs = torch.squeeze(self.policy.get_log_probs(obs, actions), dim=-1)
        log_probs_old = torch.squeeze(old_policy.get_log_probs(obs, actions), dim=-1)

        ratio = torch.exp(log_probs - log_probs_old)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1-self.ppo_epsilon, 1+self.ppo_epsilon)

        if torch.any(torch.isinf(ratio)):
            print("ratio is inf...")
            # import pdb;pdb.set_trace()
            objective = policy_loss_2.mean() # policy_loss_1に -inf が入ったりするので、policy_loss_2を使用するように変更した
        else:
            objective = torch.min(policy_loss_1, policy_loss_2).mean()

        if torch.any(torch.isnan(ratio)):
            print("ratio is nan...")
            sys.exit()


        # objective = torch.min(policy_loss_1, policy_loss_2).mean()
        # objective = policy_loss_2.mean() # policy_loss_1に -inf が入ったりするので、policy_loss_2を使用するように変更した
        objective += self.entropy_coeff * (-log_probs).mean()

        kl = (log_probs_old - log_probs).mean()

        return objective, kl

    def train_from_paths(self, paths, path_len = None):

        """
        Path preprocessing; have to copy so we don't modify when paths are used elsewhere
        """

        paths = copy.deepcopy(paths)
        for path in paths:
            # Other places like to have an extra dimension so that all arrays are 2D
            path['rewards'] = np.squeeze(path['rewards'], axis=-1)
            path['terminals'] = np.squeeze(path['terminals'], axis=-1)

        obs, actions = [], []
        for path in paths:
            obs.append(path['observations'])
            actions.append(path['actions'])
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)

        total_steps = obs.shape[0]

        obs_tensor, act_tensor = ptu.from_numpy(obs), ptu.from_numpy(actions)
        """
        Policy training loop
        """

        old_policy = copy.deepcopy(self.policy)
        # old_policy = ptu.clone_module(self.policy)
        with torch.no_grad():
            log_probs_old = old_policy.get_log_probs(obs_tensor, act_tensor).squeeze(dim=-1)

        rem_value_epochs = self.num_epochs

        num_p = 0
        num_v = 0
        for epoch in range(self.num_policy_epochs):

            """
            Recompute advantages at the beginning of each epoch. This allows for advantages
                to utilize the latest value function.
            Note: while this is not present in most implementations, it is recommended
                  by Andrychowicz et al. 2020.
            """

            path_functions.calculate_baselines(paths, self.value_func)
            path_functions.calculate_returns(paths, self.discount)
            path_functions.calculate_advantages(
                paths, self.discount, self.gae_lambda, self.normalize_advantages,
            )

            advantages, returns, baselines = [], [], []
            for path in paths:
                advantages = np.append(advantages, path['advantages'])
                returns = np.append(returns, path['returns'])

            if epoch == 0 and self._need_to_update_eval_statistics:
                with torch.no_grad():
                    values = torch.squeeze(self.value_func(obs_tensor), dim=-1)
                    values_np = ptu.get_numpy(values)
                first_val_loss = ((returns - values_np) ** 2).mean()

            old_params = self.policy.get_param_values()

            # lstmなので、path(200step)ずつ入力
            # num_policy_steps = len(advantages) // self.policy_batch_size
            for i in range(0, total_steps, path_len):

                batch = dict(
                    observations=obs[i:i+path_len, :],
                    actions=actions[i:i+path_len, :],
                    advantages=advantages[i:i+path_len],
                )
                num_p += 1
                policy_loss, kl = self.train_policy(batch, old_policy)
                print("policy_loss, kl", policy_loss, kl)

            with torch.no_grad():
                log_probs = self.policy.get_log_probs(obs_tensor, act_tensor).squeeze(dim=-1)
            kl = (log_probs_old - log_probs).mean()

            # もともとabs(kl)ではなかったけど、マイナスに行きすぎても嫌なのでabs加える
            if (self.target_kl is not None and abs(kl) > 1.5 * self.target_kl) or (kl != kl):
                if epoch > 0 or kl != kl:  # nan check
                    self.policy.set_param_values(old_params)
                break


        # ここもvalue funcにlstm組み込んだらかえるべし
            num_value_steps = len(advantages) // self.value_batch_size
            for i in range(num_value_steps):
                batch = ppp.sample_batch(
                    self.value_batch_size,
                    observations=obs,
                    targets=returns,
                )
                value_loss = self.train_value(batch)
                num_v += 1
            rem_value_epochs -= 1

        # Ensure the value function is always updated for the maximum number
        # of epochs, regardless of if the policy wants to terminate early.
        for _ in range(rem_value_epochs):
            num_value_steps = len(advantages) // self.value_batch_size
            for i in range(num_value_steps):
                batch = ppp.sample_batch(
                    self.value_batch_size,
                    observations=obs,
                    targets=returns,
                )
                value_loss = self.train_value(batch)

        if self._need_to_update_eval_statistics:
            with torch.no_grad():
                _, _, _, log_pi, *_ = self.policy(obs_tensor, return_log_prob=True)
                values = torch.squeeze(self.value_func(obs_tensor), dim=-1)
                values_np = ptu.get_numpy(values)

            errors = returns - values_np
            explained_variance = 1 - (np.var(errors) / np.var(returns))
            value_loss = errors ** 2

            self.eval_statistics['Num Epochs'] = epoch + 1

            print("Policy Loss:", ptu.get_numpy(policy_loss).mean())
            print(ptu.get_numpy(policy_loss))

            self.eval_statistics['Policy Loss'] = ptu.get_numpy(policy_loss).mean()
            self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl).mean()
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantages',
                advantages,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Returns',
                returns,
            ))

            self.eval_statistics['Value Loss'] = value_loss.mean()
            self.eval_statistics['First Value Loss'] = first_val_loss
            self.eval_statistics['Value Explained Variance'] = explained_variance
            self.eval_statistics.update(create_stats_ordered_dict(
                'Values',
                ptu.get_numpy(values),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Value Squared Errors',
                value_loss,
            ))
