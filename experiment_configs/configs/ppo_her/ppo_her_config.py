import torch

from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.models.networks import FlattenMlp
from lifelong_rl.trainers.pg.ppo import PPOTrainer
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.trainers.ppo_her.ppo_her import PPOHERTrainer


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):

    M = variant['policy_kwargs']['layer_size']
    layer_division = variant['policy_kwargs']['layer_division']

    policy_hidden_sizes = []
    for i in range(variant['policy_kwargs']['layer_num']):
        if i == 0:
            policy_hidden_sizes.append(M)
        else:
            policy_hidden_sizes.append(M // layer_division)

    print("policy_hidden_sizes", policy_hidden_sizes)


    # PPO is very finicky with weight initializations

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=policy_hidden_sizes,
        hidden_activation=torch.tanh,
        b_init_value=0,
        w_scale=1.41,
        init_w=0.01,
        final_init_scale=0.01,
        std=0.5,
        hidden_init=ptu.orthogonal_init,
    )

    # M = variant['value_kwargs']['layer_size']

    value_func = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=policy_hidden_sizes,
        hidden_activation=torch.tanh,
        hidden_init=ptu.orthogonal_init,
        b_init_value=0,
        final_init_scale=1,
    )

    # memo これなぜeval env?
    policy_trainer = PPOTrainer(
        env=expl_env,
        policy=policy,
        value_func=value_func,
        **variant['policy_trainer_kwargs'],
    )

    trainer = PPOHERTrainer(
        policy_trainer = policy_trainer,
        replay_buffer=replay_buffer,
        replay_size=variant['replay_buffer_size'],
        **variant['trainer_kwargs'],
    )

    config = dict()
    config.update(dict(
        trainer=trainer,
        exploration_policy=policy,
        evaluation_policy=MakeDeterministic(policy),
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
    ))
    config['algorithm_kwargs'] = variant['algorithm_kwargs']

    return config
