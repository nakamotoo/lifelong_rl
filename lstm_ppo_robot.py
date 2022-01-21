from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.lstm_ppo.lstm_ppo_config import get_config
from experiment_configs.algorithms.batch import get_algorithm
import os

num_epochs = 4
policy_layer_size = 256
discrim_layer_size = 256
horizon = int(4000)
policy_num_layer = 3
discrim_num_layer = 3

# ENV_NAME = 'Gridworld'
ENV_NAME = 'PartialFetchPickAndPlace'

experiment_kwargs = dict(
    exp_name='lstm-memory-ppo-{}-p{}-d{}-{}'.format(str(ENV_NAME), str(policy_layer_size), str(discrim_layer_size), str(policy_num_layer)),
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    variant = dict(
        algorithm='LSTM-Memory-PPO',
        collector_type='lstm_memory',
        replay_buffer_size=horizon,   # for DADS, only used to store past history
        generated_replay_buffer_size=horizon,   # off-policy replay buffer helps learning
        env_name=ENV_NAME,
        env_kwargs=dict(
            terminates=False,
        ),
        policy_kwargs=dict(
            layer_size=policy_layer_size,
            latent_dim=policy_layer_size,
            layer_num = policy_num_layer,
        ),
        discriminator_kwargs=dict(
            layer_size=discrim_layer_size,
            num_layers=discrim_num_layer,
            restrict_input_size=0,
        ),
        trainer_kwargs=dict(
            num_prior_samples=256,
            num_discrim_updates=8,
            num_policy_updates=num_epochs,
            discrim_learning_rate=3e-4,
            policy_batch_size=512,
            reward_bounds=(-50, 50),
            reward_scale=1,  # increasing reward scale helps learning signal
        ),
        policy_trainer_kwargs=dict(
            discount=0.99,
            gae_lambda=0.97,
            ppo_epsilon=0.1,
            policy_lr=3e-5,
            value_lr=3e-5,
            target_kl=0.01,
            num_epochs=num_epochs,
            policy_batch_size=512,
            value_batch_size=512,
            normalize_advantages=True,
        ),
        algorithm_kwargs=dict(
            num_epochs=20000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=horizon,
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
