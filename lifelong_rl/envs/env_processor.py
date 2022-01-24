import gym

from lifelong_rl.envs.wrappers import NormalizedBoxEnv, NonTerminatingEnv, SwapColorEnv


gym.logger.set_level(40)  # stop annoying Box bound precision error


def make_env(env_name, terminates=True, **kwargs):
    env = None
    base_env = None
    env_infos = dict()

    """
    Episodic reinforcement learning
    """
    if env_name == 'HalfCheetah':
        from gym.envs.mujoco import HalfCheetahEnv
        base_env = HalfCheetahEnv
        env_infos['mujoco'] = True
    elif env_name == 'Ant':
        from lifelong_rl.envs.environments.ant_env import AntEnv
        base_env = AntEnv
        env_infos['mujoco'] = True
    elif env_name == 'Hopper':
        from gym.envs.mujoco import HopperEnv
        base_env = HopperEnv
        env_infos['mujoco'] = True
    elif env_name == 'InvertedPendulum':
        from gym.envs.mujoco import InvertedPendulumEnv
        base_env = InvertedPendulumEnv
        env_infos['mujoco'] = True
    elif env_name == 'Humanoid':
        from lifelong_rl.envs.environments.humanoid_env import HumanoidEnv
        base_env = HumanoidEnv
        env_infos['mujoco'] = True
    elif env_name == 'PickAndPlace':
        from lifelong_rl.envs.environments.fetch_pick_and_place_env import FetchPickAndPlaceEnv
        base_env = FetchPickAndPlaceEnv
        env_infos['mujoco'] = True
    elif env_name == 'Push':
        from lifelong_rl.envs.environments.push_env import FetchPushEnv
        base_env = FetchPushEnv
        env_infos['mujoco'] = True
    elif env_name == 'Slide':
        from lifelong_rl.envs.environments.slide_env import FetchSlideEnv
        base_env = FetchSlideEnv
        env_infos['mujoco'] = True


    """
    Lifelong reinforcement learning
    """
    if env_name == 'LifelongHopper':
        from lifelong_rl.envs.environments.hopper_env import LifelongHopperEnv
        base_env = LifelongHopperEnv
        env_infos['mujoco'] = True
    elif env_name == 'LifelongAnt':
        from lifelong_rl.envs.environments.ant_env import LifelongAntEnv
        base_env = LifelongAntEnv
        env_infos['mujoco'] = True
    elif env_name == 'Gridworld':
        from lifelong_rl.envs.environments.continuous_gridworld.cont_gridworld import ContinuousGridworld
        base_env = ContinuousGridworld
        env_infos['mujoco'] = False

    """
    Partially Observable
    """
    if env_name == 'PartialHalfCheetah':
        from lifelong_rl.envs.environments.cheetah_env import PartialHalfCheetahEnv
        base_env = PartialHalfCheetahEnv
        env_infos['mujoco'] = True
    elif env_name == 'PartialPickAndPlace':
        from lifelong_rl.envs.environments.fetch_pick_and_place_env import PartialFetchPickAndPlaceEnv
        # from gym.envs.robotics import FetchPickAndPlaceEnv
        base_env = PartialFetchPickAndPlaceEnv
        env_infos['mujoco'] = True
    elif env_name == 'PartialPush':
        from lifelong_rl.envs.environments.push_env import PartialFetchPushEnv
        base_env = PartialFetchPushEnv
        env_infos['mujoco'] = True
    elif env_name == 'PartialSlide':
        from lifelong_rl.envs.environments.slide_env import PartialFetchSlideEnv
        base_env = PartialFetchSlideEnv
        env_infos['mujoco'] = True
    elif env_name == 'PartialAnt':
        from lifelong_rl.envs.environments.ant_env import PartialAntEnv
        base_env = PartialAntEnv
        env_infos['mujoco'] = True
    elif env_name == 'PartialHumanoid':
        from lifelong_rl.envs.environments.humanoid_env import PartialHumanoidEnv
        base_env = PartialHumanoidEnv
        env_infos['mujoco'] = True

    if env is None and base_env is None:
        raise NameError('env_name not recognized')

    if env is None:
        env = base_env(**kwargs)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        env = NormalizedBoxEnv(env)

    if not terminates:
        env = NonTerminatingEnv(env)

    return env, env_infos
