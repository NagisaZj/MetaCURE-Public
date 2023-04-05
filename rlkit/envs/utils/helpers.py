import os
import pickle
import gym
# import pickle5 as pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reset_env(env, args, tasks=None, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset(task=tasks).float().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        if tasks is not None:
            raise NotImplementedError()
        for i in indices:
            state[i] = env.reset(index=i)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).float().to(device) if args.pass_task_to_policy else None
        
    return state, belief, task


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):

    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None

    if args.add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if args.sample_embeddings:
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1:
        latent = latent.squeeze(0)

    return latent

def squash_action(action, args):
    if args.norm_actions_post_sampling:
        return torch.tanh(action)
    else:
        return action

def env_step(env, action, args):
    act = squash_action(action.detach(), args)
    next_obs, reward, done, infos = env.step(act)

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).float().to(device) if (args.pass_task_to_policy or args.decode_task) else None

    return [next_obs, belief, task], reward, done, infos
