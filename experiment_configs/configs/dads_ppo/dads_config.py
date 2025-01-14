import torch

from lifelong_rl.policies.base.latent_prior_policy import PriorLatentPolicy
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.models.networks import FlattenMlp
from lifelong_rl.trainers.dads.dads import DADSTrainer
from lifelong_rl.trainers.dads.skill_dynamics import SkillDynamics
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

    control_policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        restrict_obs_dim=restrict_dim,
        hidden_activation=torch.tanh,
        b_init_value=0,
        w_scale=1.41,
        init_w=0.01,
        final_init_scale=0.01,
        std=0.5,
        hidden_init=ptu.orthogonal_init,
    )

    prior = torch.distributions.uniform.Uniform(
        -ptu.ones(latent_dim), ptu.ones(latent_dim),
    )

    policy = PriorLatentPolicy(
        policy=control_policy,
        prior=prior,
        unconditional=True,
    )

    # M = variant['value_kwargs']['layer_size']

    value_func = FlattenMlp(
        input_size=obs_dim + latent_dim,
        output_size=1,
        hidden_sizes=[M, M],
        hidden_activation=torch.tanh,
        hidden_init=ptu.orthogonal_init,
        b_init_value=0,
        final_init_scale=1,
    )

    """
    Discriminator
    """

    discrim_kwargs = variant['discriminator_kwargs']
    discriminator = SkillDynamics(
        observation_size=obs_dim if restrict_dim == 0 else restrict_dim,
        action_size=action_dim,
        latent_size=latent_dim,
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

    """
    Setup of intrinsic control
    """

    dads_type = variant.get('dads_type', 'onpolicy')
    if dads_type == 'onpolicy':
        trainer_class = DADSTrainer
    else:
        raise NotImplementedError('dads_type not recognized')

    trainer = trainer_class(
        control_policy=control_policy,
        discriminator=discriminator,
        replay_buffer=replay_buffer,
        replay_size=variant['generated_replay_buffer_size'],
        policy_trainer=policy_trainer,
        restrict_input_size=restrict_dim,
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
