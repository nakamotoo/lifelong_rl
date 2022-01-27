import torch

from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.lstm_gaussian_policy import LSTMGaussianPolicy
from lifelong_rl.policies.base.lstm_memory_policy import LSTMMemoryPolicy
from lifelong_rl.models.networks import FlattenLSTMMlp
from lifelong_rl.trainers.pg.ppo_lstm import PPOLSTMTrainer
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

    control_policy = LSTMGaussianPolicy(
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

    policy = LSTMMemoryPolicy(
        policy=control_policy,
        latent_dim=M // layer_division,
    )

    # M = variant['value_kwargs']['layer_size']

    value_func = FlattenLSTMMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=policy_hidden_sizes,
        hidden_activation=torch.tanh,
        hidden_init=ptu.orthogonal_init,
        b_init_value=0,
        final_init_scale=1,
    )

    # memo これなぜeval env?
    policy_trainer = PPOLSTMTrainer(
        env=expl_env,
        policy=control_policy,
        value_func=value_func,
        **variant['policy_trainer_kwargs'],
    )

    trainer = PPOHERTrainer(
        policy_trainer = policy_trainer,
        replay_buffer=replay_buffer,
        replay_size=variant['replay_buffer_size'],
        path_len = variant['algorithm_kwargs']['max_path_length'],
        **variant['trainer_kwargs'],
    )

    config = dict()
    config.update(dict(
        trainer=trainer,
        exploration_policy=policy,
        # evaluation_policy=MakeDeterministic(policy),
        evaluation_policy = policy,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
    ))
    config['algorithm_kwargs'] = variant['algorithm_kwargs']

    return config
