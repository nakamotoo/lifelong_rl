from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.dads_ppo.dads_config import get_config
from experiment_configs.algorithms.batch import get_algorithm
import os

# ENV_NAME = 'Gridworld'
ENV_NAME = 'HalfCheetah'
experiment_kwargs = dict(
    exp_name='dads-ppo-cheetah',
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    variant = dict(
        algorithm='DADS-PPO',
        collector_type='batch_latent',
        replay_buffer_size=int(5000),   # for DADS, only used to store past history
        generated_replay_buffer_size=int(5000),   # off-policy replay buffer helps learning
        env_name=ENV_NAME,
        env_kwargs=dict(
            # grid_files=['blank'],  # specifies which file to load for gridworld
            terminates=False,
        ),
        policy_kwargs=dict(
            layer_size=512,
            latent_dim=2,
        ),
        discriminator_kwargs=dict(
            layer_size=512,
            num_layers=2,
            restrict_input_size=0,
        ),
        trainer_kwargs=dict(
            num_prior_samples=100,
            num_discrim_updates=16,
            num_policy_updates=128,
            discrim_learning_rate=3e-4,
            policy_batch_size=512,
            reward_bounds=(-50, 50),
            reward_scale=1,  # increasing reward scale helps learning signal
        ),
        policy_trainer_kwargs=dict(
            discount=0.99,
            gae_lambda=0.97,
            ppo_epsilon=0.2,
            policy_lr=3e-4,
            value_lr=3e-4,
            target_kl=None,
            num_epochs=16,
            policy_batch_size=64,
            value_batch_size=64,
            normalize_advantages=True,
        ),
        algorithm_kwargs=dict(
            num_epochs=50000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=5000,
            min_num_steps_before_training=0,
            max_path_length=200,
            save_snapshot_freq=50,
        ),
    )

    sweep_values = {
    }

    launch_experiment(
        get_config=get_config,
        get_algorithm=get_algorithm,
        variant=variant,
        sweep_values=sweep_values,
        **experiment_kwargs
    )
