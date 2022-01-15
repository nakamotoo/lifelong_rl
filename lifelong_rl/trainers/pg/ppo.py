import torch

from lifelong_rl.trainers.pg.pg import PGTrainer

import sys


class PPOTrainer(PGTrainer):

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

        if torch.any(torch.isnan(ratio)):
            print("ratio is nan...")
            sys.exit()


        # objective = torch.min(policy_loss_1, policy_loss_2).mean()
        objective = policy_loss_2.mean() # policy_loss_1に -inf が入ったりするので、policy_loss_2を使用するように変更した
        objective += self.entropy_coeff * (-log_probs).mean()

        kl = (log_probs_old - log_probs).mean()

        return objective, kl
