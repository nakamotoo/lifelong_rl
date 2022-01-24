from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.lstm_ppo.lstm_ppo_config import get_config
from experiment_configs.algorithms.batch import get_algorithm
import os

num_epochs = 4
policy_layer_size = 512
discrim_layer_size = 512
horizon = int(4000)
policy_num_layer = 2
discrim_num_layer = 2
layer_division = 2

# PartialPickAndPlace
# PartialPush
# PartialSlide

ENV_NAME = 'PartialPickAndPlace'
intrinsic_reward_scale= 1  # increasing reward scale helps learning signal
oracle_reward_scale = 0

is_downstream = True
# robin1 PickAndPlace blending
load_model_path = "/data/local/mitsuhiko/lifelong_rl/01-23-lstm-memory-ppo-PartialPickAndPlace-p512-2-d512-div2-blend-i3-o1/01-23-lstm-memory-ppo-PartialPickAndPlace-p512-2-d512-div2-blend-i3-o1_2022_01_23_22_33_11_0000--s-59912257/itr_2499"


if policy_num_layer == 1:
    assert layer_division == 1

if is_downstream:
    use_desired_goal = True
    exp_name='lstm-memory-ppo-{}-downstream'.format(str(ENV_NAME))
elif oracle_reward_scale > 0:
    exp_name='lstm-memory-ppo-{}-p{}-{}-d{}-div{}-blend-i{}-o{}'.format(str(ENV_NAME), str(policy_layer_size), str(policy_num_layer), str(discrim_layer_size), str(layer_division), str(intrinsic_reward_scale), str(oracle_reward_scale))
    use_desired_goal = True
else:
    exp_name='lstm-memory-ppo-{}-p{}-{}-d{}-div{}'.format(str(ENV_NAME), str(policy_layer_size), str(policy_num_layer), str(discrim_layer_size), str(layer_division))
    use_desired_goal = True



experiment_kwargs = dict(
    exp_name=exp_name,
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
            use_desired_goal = use_desired_goal
        ),
        policy_kwargs=dict(
            layer_size=policy_layer_size,
            latent_dim=policy_layer_size // layer_division,
            layer_num = policy_num_layer,
            layer_division = layer_division
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
            reward_scale=intrinsic_reward_scale,  # increasing reward scale helps learning signal
            oracle_reward_scale = oracle_reward_scale,
            is_downstream = is_downstream, # downstream: fintune with oracle reward
            load_model_path = load_model_path
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
