from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.pg.ppo_config import get_config
from experiment_configs.algorithms.batch import get_algorithm
import os

num_epochs = 8
reward_mode = "back"
ENV_NAME = 'HalfCheetah'
experiment_kwargs = dict(
    exp_name='ppo-oracle-cheetah-{}'.format(reward_mode),
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)

horizon = int(2000)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    variant = dict(
        algorithm='PPO',
        collector_type='batch',
        env_name=ENV_NAME,
        env_kwargs=dict(
            reward_mode = reward_mode
        ),
        replay_buffer_size=horizon,
        policy_kwargs=dict(
            layer_size=512,
            layer_num = 2,
            layer_division = 1
        ),
        value_kwargs=dict(
            layer_size=512,
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
            num_epochs=10000,
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
