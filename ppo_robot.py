from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.ppo_her.ppo_her_config import get_config
# from experiment_configs.configs.pg.ppo_config import get_config

from experiment_configs.algorithms.batch import get_algorithm
import os

num_epochs =  4
horizon = int(2000)
policy_layer_size = 512
policy_num_layer = 2
layer_division = 2

# PickAndPlace, PartialPickAndPlace
# Push, PartialPush
# Slide, PartialSlide
ENV_NAME = 'PartialPush'

experiment_kwargs = dict(
    exp_name='ppo-oracle-{}-p{}-num{}-div{}'.format(ENV_NAME, str(policy_layer_size), str(policy_num_layer), str(layer_division)),
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    variant = dict(
        algorithm='PPO',
        # collector_type='batch',
        collector_type='lstm_memory',
        env_name=ENV_NAME,
        env_kwargs=dict(
            terminates=False,
            use_desired_goal = True,
            reward_type = "else"
        ),
        replay_buffer_size=horizon,
        policy_kwargs=dict(
            layer_size=policy_layer_size,
            layer_num = policy_num_layer,
            layer_division = layer_division
        ),
        trainer_kwargs = dict(

        ),
        value_kwargs=dict(
            layer_size=policy_layer_size,
        ),
        policy_trainer_kwargs=dict(
            discount=0.99,
            gae_lambda=0.97,
            ppo_epsilon=0.1,
            policy_lr=3e-4,
            value_lr=3e-4,
            target_kl=0.01,
            num_epochs=num_epochs,
            policy_batch_size=200,
            value_batch_size=200,
            normalize_advantages=True,
        ),
        algorithm_kwargs=dict(
            num_epochs=10000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=horizon,
            min_num_steps_before_training=0,
            max_path_length=100,
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
