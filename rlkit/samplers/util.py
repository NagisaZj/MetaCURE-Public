import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch

def rollout(env, agent, max_path_length=np.inf, accum_context=True, resample_z=False, animated=False,metaworld_sparse=False):
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

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param accum_context: if True, accumulate the collected context
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    z_means = []
    z_vars = []
    o = env.reset()
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        if metaworld_sparse:
            sr = r if env_info['success'] else 0
            env_info['sparse_reward'] = sr
            r = r if env_info['success'] else 0
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        z_means.append(agent.z_means.cpu().data.numpy())
        z_vars.append(agent.z_vars.cpu().data.numpy())
        path_length += 1
        if d:
            break

        o = next_o
        if animated:
            env.render()

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
        z_means=z_means,
        z_vars=z_vars
    )

def seedrollout(env, agent, max_path_length=np.inf, accum_context=True, resample_z=False, animated=False,random_seed=None,sample_interval=5):
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

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param accum_context: if True, accumulate the collected context
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    z_means = []
    z_vars = []
    o = env.reset()
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        if agent.context is None:
            agent.z = ptu.FloatTensor(random_seed)
        else:
            if path_length % sample_interval ==0:
                agent.infer_posterior(agent.context)
                agent.z = agent.z_means + torch.sqrt(agent.z_vars)*ptu.FloatTensor(random_seed)
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        z_means.append(np.mean(agent.z_means.cpu().data.numpy()))
        z_vars.append(np.mean(agent.z_vars.cpu().data.numpy()))
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()

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
        z_means=z_means,
        z_vars=z_vars
    )

def exprolloutsimple(env, agent, max_path_length,  max_trajs, accum_context_for_agent=False, context_agent = None,rsample=1,metaworld_sparse=False):
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

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param accum_context: if True, accumulate the collected context
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    z_means = []
    z_vars = []
    sparse_rewards=[]
    num_trajs = 0
    i=0
    while num_trajs<max_trajs:
        o = env.reset()

        a = np.zeros(agent.action_dim, dtype=np.float32)
        r= 0
        info = {"sparse_reward":0}
        agent.update_context([o,a,r,info])
        next_o = None
        path_length = 0
        while path_length < max_path_length:

            if i % rsample==0:
                agent.infer_posterior(agent.context)
            i = i + 1

            a, agent_info = agent.get_action(o)
            #print(a)
            next_o, r, d, env_info = env.step(a)
            if metaworld_sparse:
                sr = r if env_info['success'] else 0
                env_info['sparse_reward'] = sr
                r = r if env_info['success'] else 0
            #print(o, a, r)
            agent.update_context([next_o,a,r,env_info])

            #r = env_info['sparse_reward']
            # update the agent's current context

            #r = agent.infer_reward()

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            z_means.append(agent.z.cpu().data.numpy())


            sparse_rewards.append(env_info.get('sparse_reward', 0))
            path_length += 1
            if d:
                break
            o = next_o

        num_trajs+=1

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
        z_means=z_means

    )

def exprollout_splitsimple(env, agent, max_path_length,  max_trajs, accum_context_for_agent=False, context_agent = None,rsample=1,metaworld_sparse=False):
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

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param accum_context: if True, accumulate the collected context
    :return:
    """
    paths = []
    num_trajs = 0
    i=0
    while num_trajs<max_trajs:
        o = env.reset()
        a = np.zeros(agent.action_dim,dtype=np.float32)
        r= 0
        info = {"sparse_reward":0}
        agent.update_context([o,a,r,info])
        agent.infer_posterior(agent.context)
        if accum_context_for_agent:
            context_agent.update_context([o,a,r,info])
        next_o = None
        path_length = 0
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        z_means = []
        z_vars = []
        sparse_rewards = []
        while path_length < max_path_length:
            if i % rsample==0:
                agent.infer_posterior(agent.context)
            i = i+1
            a, agent_info = agent.get_action(o)
            #print(o,a,agent.context,agent.z_means,agent.z_vars)
            next_o, r, d, env_info = env.step(a)
            if metaworld_sparse:
                sr = r if env_info['success'] else 0
                env_info['sparse_reward'] = sr
                r = r if env_info['success'] else 0


            agent.update_context([next_o, a, r, env_info])
            if accum_context_for_agent:
                context_agent.update_context([next_o, a, r, info])
            #print(np.mean(ptu.get_numpy(agent.z_means)),np.mean(ptu.get_numpy(agent.z_vars)))

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            sparse_rewards.append(env_info.get('sparse_reward', 0))
            z_means.append(agent.z.cpu().data.numpy())

            path_length += 1
            if d:
                break
            o = next_o

        num_trajs+=1

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
        paths.append( dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            z_means=z_means
        ))
        #print(len(paths))

    return paths


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
