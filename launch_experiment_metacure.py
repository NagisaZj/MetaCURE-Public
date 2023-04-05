"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import SnailPolicy, TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder,SnailEncoder, RNN
from rlkit.torch.sac.sac import PEARLSoftActorCritic, ExpSACFinSubtract3, ExpSACFinSubtract4
from rlkit.torch.sac.agent import  ExpAgentFin, PEARLAgent2
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config_exp as default_config
from metaworld import ML1


def experiment(variant):

    # create multi-task environment and sample tasks
    if variant['env_name'] !='metaworld':
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
        tasks = env.get_all_task_idx()
        #print(tasks)
    else:
        '''available_tasks=ML1.available_tasks()
        print(available_tasks)
        env = NormalizedBoxEnv(ML1.get_train_tasks(available_tasks[variant['env_params']['task_id']]))
        print(env)
        tasks = env.sample_tasks(variant['env_params']['n_tasks'])
        print(tasks)
        env.tasks_pool = tasks
        tasks = list(range(variant['env_params']['n_tasks']))'''
        print(ML1.ENV_NAMES)
        name = ML1.ENV_NAMES[variant['env_params']['task_id']]
        ml1 = ML1(name)  # Construct the benchmark, sampling tasks

        env = ml1.train_classes[name]()  # Create an environment with task `pick_place`
        tasks = ml1.train_tasks+ml1.test_tasks
        env = NormalizedBoxEnv(env)
        print(env)
        env.tasks_pool = tasks
        tasks = list(range(variant['env_params']['n_tasks']))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    #print(env.action_space.low,env.action_space.high)
    reward_dim = 1
    latent_dim = variant['latent_size']
    context_encoder_output_dim = latent_dim * 2
    pie_hidden_dim = variant['algo_params']['pie_hidden_dim']
    # instantiate networks
    net_size = variant['net_size']
    input_length = variant['algo_params']['embedding_mini_batch_size']
    num_train_tasks=variant['n_train_tasks']
    encoder = SnailEncoder(
        hidden_sizes=[64, 64, 64],
        input_size=obs_dim + action_dim + 1,
        output_size=context_encoder_output_dim,
        input_length= input_length
    )
    if 'metaworld' in variant['env_name']:
        encoder = RNN(
        hidden_sizes=[64, 64, 64],
        input_size=obs_dim + action_dim + 1,
        output_size=context_encoder_output_dim
    )
    # encoder = RNN(
    #    hidden_sizes=[64, 64, 64],
    #    input_size=obs_dim + action_dim + 1,
    #    output_size=context_encoder_output_dim
    # )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent2(latent_dim,
        encoder,
        policy,
        **variant['algo_params'])
    qf1_exp = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size= obs_dim + action_dim + latent_dim*2 ,
        output_size=1,
    )
    qf2_exp = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size= obs_dim + action_dim + latent_dim*2 ,
        output_size=1,
    )
    vf_exp = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size= obs_dim  + latent_dim*2 ,
        output_size=1,
    )
    policy_exp = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim= obs_dim  + latent_dim*2 ,
        latent_dim=pie_hidden_dim,
        action_dim=action_dim,
    )
    rew_decoder = FlattenMlp( hidden_sizes=[net_size, net_size, net_size],
        input_size= latent_dim*2+obs_dim+action_dim ,
        output_size=1,)
    transition_decoder = FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                             input_size=latent_dim*2 + obs_dim + action_dim,
                             output_size=obs_dim, )

    reward_predictor = FlattenMlp(
        hidden_sizes=[64, 64, 64],
        input_size=obs_dim + action_dim + num_train_tasks,
        output_size=1
    )
    transition_predictor = FlattenMlp(
        hidden_sizes=[64, 64, 64],
        input_size=obs_dim + action_dim + num_train_tasks,
        output_size=obs_dim
    )
    baseline_reward_predictors = []
    baseline_trans_predictors = []
    for i in range(variant['n_train_tasks']):
        baseline_reward_predictors.append(FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                             input_size= obs_dim + action_dim,
                             output_size=1, ))
        baseline_trans_predictors.append(FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                                                     input_size= obs_dim + action_dim,
                                                     output_size=obs_dim, ))

    agent_exp = ExpAgentFin(latent_dim,
                            encoder,
                            policy_exp,
                            None,
                            None,
                            action_dim,
                            **variant['algo_params'])
    if 'vel' in variant['env_name'] or 'point' in variant['env_name']:
    #if 'point' in variant['env_name']:
        print('36!')
        algorithm = ExpSACFinSubtract3(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets = [agent, qf1, qf2, vf],
        latent_dim = latent_dim,
        nets_exp=[agent_exp, qf1_exp, qf2_exp, vf_exp,rew_decoder,transition_decoder],
        encoder=encoder,
        baseline_reward_predictors=baseline_reward_predictors,
        baseline_trans_predictors=baseline_trans_predictors,
        **variant['algo_params']
    )
    else:
        algorithm = ExpSACFinSubtract4(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets = [agent, qf1, qf2, vf],
        latent_dim = latent_dim,
        nets_exp=[agent_exp, qf1_exp, qf2_exp, vf_exp,rew_decoder,transition_decoder],
        encoder=encoder,
        reward_predictor = reward_predictor,
        transition_predictor = transition_predictor,
        **variant['algo_params']
    )

    # optionally load pre-trained weights


    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        device = torch.device('cuda:0')
        print(device)
        algorithm.to(device)


    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    if gpu < 0:
        variant['util_params']['use_gpu'] = False

    experiment(variant)

if __name__ == "__main__":
    main()

