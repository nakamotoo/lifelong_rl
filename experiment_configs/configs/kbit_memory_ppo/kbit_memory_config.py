import torch
import numpy as np

from lifelong_rl.policies.base.kbit_memory_policy import KbitMemoryPolicy
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.models.networks import FlattenMlp
from lifelong_rl.trainers.kbit_memory.kbit_memory import KbitMemoryTrainer
from lifelong_rl.trainers.kbit_memory.state_predictor import StatePredictor
from lifelong_rl.trainers.pg.ppo import PPOTrainer
import lifelong_rl.torch.pytorch_util as ptu
import lifelong_rl.util.pythonplusplus as ppp


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):

    """
    Policy construction
    """

    M = variant['policy_kwargs']['layer_size']
    latent_dim = variant['policy_kwargs']['latent_dim']
    restrict_dim = variant['discriminator_kwargs']['restrict_input_size']

    hidden_state_dim = expl_env.hidden_state_dim

    print("obs_dim, action_dim, hidden_state_dim = ", obs_dim, action_dim, hidden_state_dim)

    control_policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim + latent_dim,
        hidden_sizes=[M] * variant['policy_kwargs']['layer_num'],
        restrict_obs_dim=restrict_dim,
        hidden_activation=torch.tanh,
        b_init_value=0,
        w_scale=1.41,
        init_w=0.01,
        final_init_scale=0.01,
        std=0.5,
        hidden_init=ptu.orthogonal_init,
    )

    # k bit binary prior にする
    # randomで初期化
    # prior = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]*latent_dim))
    # 全部0で初期化
    prior = torch.distributions.bernoulli.Bernoulli(ptu.from_numpy(np.array([0.0]*latent_dim)))

    # prior = torch.distributions.uniform.Uniform(
    #     -ptu.ones(latent_dim), ptu.ones(latent_dim),
    # )

    policy = KbitMemoryPolicy(
        policy=control_policy,
        prior=prior,
        latent_dim=latent_dim,
    )

    value_func = FlattenMlp(
        input_size=obs_dim + latent_dim,
        output_size=1,
        hidden_sizes=[M] * variant['policy_kwargs']['layer_num'],
        hidden_activation=torch.tanh,
        hidden_init=ptu.orthogonal_init,
        b_init_value=0,
        final_init_scale=1,
    )
    """
    Discriminator
    """

    discrim_kwargs = variant['discriminator_kwargs']
    discriminator = StatePredictor(
        observation_size=obs_dim,
        latent_size=latent_dim,
        hidden_state_size = hidden_state_dim,
        normalize_observations=discrim_kwargs.get('normalize_observations', True),
        fix_variance=discrim_kwargs.get('fix_variance', True),
        fc_layer_params=[discrim_kwargs['layer_size']] * discrim_kwargs['num_layers'],
    )

    """
    Policy trainer
    """

    policy_trainer = PPOTrainer(
        env=expl_env,
        policy=control_policy,
        value_func=value_func,
        **variant['policy_trainer_kwargs'],
    )

    trainer = KbitMemoryTrainer(
        control_policy=control_policy,
        discriminator=discriminator,
        replay_buffer=replay_buffer,
        replay_size=variant['generated_replay_buffer_size'],
        policy_trainer=policy_trainer,
        restrict_input_size=restrict_dim,
        hidden_state_dim = hidden_state_dim,
        algorithm = variant["algorithm"],
        **variant['trainer_kwargs'],
    )

    """
    Create config dict
    """

    config = dict()
    config.update(dict(
        trainer=trainer,
        exploration_policy=policy,
        evaluation_policy=policy,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
        prior=prior,
        control_policy=control_policy,
        latent_dim=latent_dim,
        policy_trainer=policy_trainer,
    ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())
    config['offline_kwargs'] = variant.get('offline_kwargs', dict())

    return config
