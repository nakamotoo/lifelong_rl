import numpy as np


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    o = obs_process(o, env)

    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )

def obs_process(o, env):
    if isinstance(o, dict):
        if env.use_desired_goal:
            o = np.concatenate([o["observation"], o["desired_goal"]])
        else:
            o = o["observation"]
    return o


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        latent_dim = None
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    env_states = []
    o = env.reset()
    # robotics環境の時
    achieved_goals = []
    o = obs_process(o, env)

    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        if latent_dim is not None:
            a = a[:-latent_dim]
        next_o, r, d, env_info = env.step(a)
        if hasattr(env, "use_desired_goal"):
            ag = next_o["achieved_goal"]
            achieved_goals.append(ag)
        next_o = obs_process(next_o, env)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        env_states.append(env.sim.get_state())
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    achieved_goals = np.array(achieved_goals)
    if len(achieved_goals.shape) == 1:
        achieved_goals = np.expand_dims(achieved_goals, 1)
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        env_states = env_states,
        achieved_goals = achieved_goals
    )


def rollout_with_latent(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        sample_latent_every=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    latents = []
    env_states = []
    o = env.reset()
    o = obs_process(o, env)

    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        if sample_latent_every is not None and path_length % sample_latent_every == 0:
            agent.sample_latent()
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        next_o = obs_process(next_o, env)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        latents.append(agent.get_current_latent())
        env_states.append(env.sim.get_state())
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        latents=np.array(latents),
        env_states = env_states
    )

def rollout_with_kbit_memory(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):

    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    writes = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    latents = []
    next_latents = [] # memoryは毎回変わるので、TD estimateの際に必要
    hidden_states = []
    env_states = []
    o = env.reset()
    o = obs_process(o, env)
    desired_goals = []

    agent.reset()
    latent_dim = agent._latent_dim
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        if path_length == 0:
            agent.sample_latent()
        hidden_s = env._get_hidden_state()
        a, agent_info = agent.get_action(o)
        a, w = a[:-latent_dim], a[-latent_dim:] # split a into a_env and a_memory
        next_o, r, d, env_info = env.step(a)
        next_o = obs_process(next_o, env)
        observations.append(o)
        hidden_states.append(hidden_s)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        latents.append(agent.get_current_latent())
        # kbit memory を書き換える m_t → m_t+1
        writes.append(w) # pathに保存するwritesを連続値のままに
        desired_goals.append(None)

        write = np.where(w <= 0.0, 0.0, 1.0)
        next_m = agent.write_memory(write)
        next_latents.append(next_m)
        # writes.append(write)

        env_states.append(env.sim.get_state())
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    writes = np.array(writes)
    if len(writes.shape) == 1:
        writes = np.expand_dims(writes, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    hidden_states = np.array(hidden_states)
    if len(hidden_states.shape) == 1:
        hidden_states = np.expand_dims(hidden_states, 1)

    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        latents=np.array(latents),
        env_states = env_states,
        writes = writes,
        hidden_states = hidden_states,
        next_latents = next_latents,
        desired_goals = desired_goals
    )

def rollout_with_lstm(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):

    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    writes = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    latents = []
    next_latents = []
    hidden_states = []
    env_states = []
    o = env.reset()
    achieved_goals = []
    desired_goals = []

    o = obs_process(o, env)

    agent.reset()
    latent_dim = agent._latent_dim
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        if path_length == 0:
            agent.reset_lstm_hidden()
        if hasattr(env, "_get_hidden_state"):
            hidden_s = env._get_hidden_state()
            hidden_states.append(hidden_s)
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)

        dg = None
        if hasattr(env, "use_desired_goal"):
            ag = next_o["achieved_goal"]
            dg = next_o["desired_goal"]
            achieved_goals.append(ag)
            desired_goals.append(dg)

        next_o = obs_process(next_o, env)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        latents.append(agent.get_current_latent())
        # next_m = agent.get_current_latent()
        # next_latents.append(next_m)

        env_states.append(env.sim.get_state())
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    writes = np.array(writes)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    hidden_states = np.array(hidden_states)
    if len(hidden_states.shape) == 1:
        hidden_states = np.expand_dims(hidden_states, 1)
    achieved_goals = np.array(achieved_goals)
    if len(achieved_goals.shape) == 1:
        achieved_goals = np.expand_dims(achieved_goals, 1)
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        latents=np.array(latents),
        env_states = env_states,
        hidden_states = hidden_states,
        next_latents = next_latents,
        achieved_goals = achieved_goals,
        desired_goals = desired_goals
    )

