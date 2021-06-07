from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm,ExpAlgorithmFin2


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(ptu.get_numpy(self.agent.z_means))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot




class ExpSACFinSubtract3(ExpAlgorithmFin2):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            nets_exp,
            encoder,
            latent_dim,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            entropy_weight=1e-2,
            intrinsic_reward_weight=1e-1,
            use_kl_div_intrinsic=False,
            prediction_reward_scale=1,
            intrinsic_reward_decay = 1,
            consider_dynamics=0,
            prediction_transition_scale=1,
            baseline_reward_predictors=None,
            baseline_trans_predictors=None,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            agent_exp=nets_exp[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            encoder=encoder,
            **kwargs
        )
        self.baseline_reward_predictors = baseline_reward_predictors
        self.baseline_trans_predictors = baseline_trans_predictors
        self.use_kl_div_intrinsic = use_kl_div_intrinsic
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.entropy_weight = entropy_weight
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.latent_dim = latent_dim
        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.qf_exp_criterion = nn.MSELoss()
        self.vf_exp_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.pred_loss = nn.MSELoss()
        self.kl_lambda = kl_lambda
        self.prediction_reward_scale = prediction_reward_scale
        self.consider_dynamics = consider_dynamics
        self.prediction_transition_scale = prediction_transition_scale

        self.use_information_bottleneck = use_information_bottleneck
        self.use_next_obs_in_context = use_next_obs_in_context
        self.intrinsic_reward_decay = intrinsic_reward_decay

        self.qf1, self.qf2, self.vf = nets[1:]
        self.qf1_exp, self.qf2_exp, self.vf_exp, self.rew_decoder, self.transition_decoder = nets_exp[1:]
        self.reward_predictor = self.rew_decoder
        self.transition_predictor = self.transition_decoder
        self.target_exp_vf = self.vf_exp.copy()
        self.target_vf = self.vf.copy()


        self.policy_exp_optimizer = optimizer_class(
            self.exploration_agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_exp_optimizer = optimizer_class(
            self.qf1_exp.parameters(),
            lr=qf_lr,
        )
        self.qf2_exp_optimizer = optimizer_class(
            self.qf2_exp.parameters(),
            lr=qf_lr,
        )
        self.vf_exp_optimizer = optimizer_class(
            self.vf_exp.parameters(),
            lr=vf_lr,
        )
        self.policy_optimizer = optimizer_class(
            self.agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=context_lr,
        )
        self.rew_optimizer = optimizer_class(
            self.rew_decoder.parameters(),
            lr=context_lr,
        )
        self.transition_optimizer = optimizer_class(
            self.transition_decoder.parameters(),
            lr=context_lr,
        )
        self.rew_predictor_optimizer = optimizer_class(
            self.reward_predictor.parameters(),
            lr=context_lr,
        )
        self.transition_predictor_optimizer = optimizer_class(
            self.transition_predictor.parameters(),
            lr=context_lr,
        )
        self.baseline_reward_optimizers = []
        self.baseline_trans_optimizers=[]
        for i in range(len(self.baseline_reward_predictors)):
            self.baseline_reward_optimizers.append(optimizer_class(
            self.baseline_reward_predictors[i].parameters(),
            lr=context_lr,
        ))
            self.baseline_trans_optimizers.append(optimizer_class(
                self.baseline_trans_predictors[i].parameters(),
                lr=context_lr,
            ))

    ###### Torch stuff #####
    @property
    def networks(self):
        return  [self.context_encoder] + [self.exploration_agent.policy] + [self.qf1_exp, self.qf2_exp, self.vf_exp, self.target_exp_vf,self.rew_decoder,self.transition_decoder] + [self.agent.policy,self.qf1, self.qf2, self.vf, self.target_vf]+[self.reward_predictor, self.transition_predictor] + self.baseline_trans_predictors + self.baseline_reward_predictors

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]

        sr = batch['rewards'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t, sr]

    def unpack_batch_context(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        sparse_r = batch['rewards'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        info = batch['env_infos'][None, ...]
        # print(o[0,:5],a[0,:5],r[0],sparse_r[0],no[0,:5])
        return [o, a, sparse_r, no, t, info, r]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for
                       idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        return unpacked

    def sample_context(self, indices, sequence=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx
                   in indices]
        context = [self.unpack_batch_context(batch) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        context = torch.cat(context[:-4], dim=2)
        return context, context_unbatched

    def pred_context(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0], 1, context[2].shape[2])
        tmp = torch.cat([r_0, context[2]], dim=1)
        a_0 = ptu.zeros(context[1].shape[0], 1, context[1].shape[2])
        tmp2 = torch.cat([a_0, context[1]], dim=1)
        tmp3 = torch.cat([torch.unsqueeze(context[0][:, 0, :], 1), context[3]], dim=1)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([tmp3, tmp2, tmp], dim=2)
        return contextr

    def pred_context_rnn(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0], 1, context[2].shape[2])
        tmp = torch.cat([r_0, context[2][:,:-1,:]], dim=1)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([context[0], context[1], tmp], dim=2)
        return contextr

    def sample_exp(self, indices,sequence=True):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.exp_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch_context(batch) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        context = torch.cat(context[:-4], dim=2)
        return  context,context_unbatched


    ##### Training #####
    def _do_training(self, indices,it_):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        #_,exp_context_unbatched = self.sample_exp(indices,True)
        #exp_context_pred = self.pred_context(exp_context_unbatched)
        _, context_unbatched = self.sample_context(indices, True)
        context_pred = self.pred_context(context_unbatched)
        context = self.sample_sac(indices)
        context_rnn = self.pred_context_rnn(context)
        #context_new_pred = self.pred_context(context)
        context_new_pred = context_pred
        # zero out context and hidden encoder state

        '''indice2 = self.train_tasks
        for i in range(1):
            context2 = self.sample_sac(indices)
            self._take_step_baseline(indice2,context2)'''



        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):

            #self._take_step(indices, context_unbatched,context_pred)
            #self._take_step_exp(indices, context_unbatched,context_pred,context,context_rnn,context_new_pred)
            self._take_step_exp(indices, context_unbatched, context_pred, context, context_rnn)

            # stop backprop


    def _min_q_exp(self,  obs,actions,z_mean,z_var):
        #print(obs.shape,actions.shape)

        q1 = self.qf1_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        q2 = self.qf2_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        min_q = torch.min(q1, q2)
        return min_q


    def _min_q(self, obs, actions,z):
        #print(obs.shape,actions.shape)

        q1 = self.qf1(torch.cat([obs, actions,z],dim=1))
        q2 = self.qf2(torch.cat([obs, actions,z],dim=1))
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network_exp(self):
        ptu.soft_update_from_to(self.vf_exp, self.target_exp_vf, self.soft_target_tau)

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def compute_kl(self,means,vars):
        std_mean = ptu.zeros(means.size())
        std_var = ptu.ones(means.size())
        tem = vars / std_var
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((std_mean - means) / std_var) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        kl_div = torch.mean(kl_div)
        return kl_div

    def compute_intrinsic(self,z_mean_prev, z_var_prev,z_mean_post,z_var_post):
        tem = z_var_post / z_var_prev
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((z_mean_prev - z_mean_post) / z_var_prev) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        return kl_div



    def _take_step_exp(self, indices,context_unbatched,context_pred,context,context_rnn):

        t, b, _ = context_pred.size()
        b = b - 1
        context_pred_pre = context_pred [:,:-1,:]
        #context_pred = context_pred.contiguous()
        z_s = self.context_encoder.forward_seq(context_pred_pre)
        #z_s = z_s.view(t, b, -1)
        z_mean = z_s[:, :self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:, self.latent_dim:])
        # print(z_mean.shape,z_var.shape)
        z_dis = torch.distributions.Normal(z_mean, torch.sqrt(z_var))
        z_sample = z_dis.rsample()
        z_sample_pearl = z_sample

        obs, actions, agent_rew, next_obs, terms, sr = context

        pred_rewardss = agent_rew

        t, b, _ = obs.size()
        #agent_rew = agent_rew.contiguous()
        #pred_rewardss = pred_rewardss.contiguous()
        agent_rew = agent_rew.view(t * b, -1)
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        pred_rewardss = pred_rewardss.view(t * b, -1)

        rewards_flat = agent_rew.detach()

        q1_pred = self.qf1(torch.cat([obs, actions, z_sample_pearl], dim=1))
        q2_pred = self.qf2(torch.cat([obs, actions, z_sample_pearl], dim=1))
        v_pred = self.vf(torch.cat([obs, z_sample_pearl.detach()], dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            target_v_values = self.target_vf(torch.cat([next_obs, z_sample_pearl], dim=1))

        # KL constraint on z if probabilistic


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        self.context_optimizer.zero_grad()
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)

        kl_div = self.compute_kl(z_mean, z_var)
        kl_loss = kl_div * self.kl_lambda
        kl_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        policy_outputs, _ = self.agent(obs, z_sample_pearl.detach())

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q(obs, new_actions, z_sample_pearl.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        rew_pred = self.rew_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
        self.rew_optimizer.zero_grad()
        rew_loss = self.pred_loss(pred_rewardss, rew_pred) * self.prediction_reward_scale
        rew_loss.backward()
        self.rew_optimizer.step()

        '''self.reward_predictor.reset(num_tasks=t)
        reward_pred_rnn = self.reward_predictor.forward_seq(context_rnn)
        self.rew_predictor_optimizer.zero_grad()
        rew_predict_loss = self.pred_loss(pred_rewardss, reward_pred_rnn) * self.prediction_reward_scale
        rew_predict_loss.backward()
        self.rew_predictor_optimizer.step()'''


        if self.consider_dynamics:
            self.transition_optimizer.zero_grad()
            trans_pred = self.transition_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
            trans_loss = self.pred_loss(next_obs, trans_pred) * self.prediction_transition_scale
            trans_loss.backward()
            self.transition_optimizer.step()

            '''self.transition_predictor.reset(num_tasks=t)
            trans_pred_rnn = self.transition_predictor.forward_seq(context_rnn)
            self.transition_predictor_optimizer.zero_grad()
            trans_predict_loss = self.pred_loss(next_obs, trans_pred_rnn) * self.prediction_reward_scale
            trans_predict_loss.backward()
            self.transition_predictor_optimizer.step()'''
        for _ in range(1):
            for number, id in enumerate(indices):
                self.baseline_reward_optimizers[id].zero_grad()
                reward_pred = self.baseline_reward_predictors[id].forward(obs[number * b:(number + 1) * b],
                                                                          actions[(number) * b:(number + 1) * b])
                rew_loss_3 = self.pred_loss(pred_rewardss[number * b:(number + 1) * b],
                                          reward_pred) * self.prediction_reward_scale
                rew_loss_3.backward()
                self.baseline_reward_optimizers[id].step()
                if self.consider_dynamics:
                    self.baseline_trans_optimizers[id].zero_grad()
                    trans_pred = self.baseline_trans_predictors[id].forward(obs[number * b:(number + 1) * b],
                                                                            actions[number * b:(number + 1) * b])
                    trans_loss = self.pred_loss(next_obs[number * b:(number + 1) * b],
                                                trans_pred) * self.prediction_transition_scale
                    trans_loss.backward()
                    self.baseline_trans_optimizers[id].step()

        policy_outputs, _ = self.exploration_agent(obs, z_mean.detach(), z_var.detach())

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        context_post = context_pred[:, 1:, :]
        context_post = context_post.contiguous()
        z_s_post = self.context_encoder.forward_seq(context_post)
        z_mean_post = z_s_post[:, :self.latent_dim]
        z_var_post = torch.nn.functional.softplus(z_s_post[:, self.latent_dim:])
        if self.intrinsic_reward_weight > 0:
            if self.use_kl_div_intrinsic:


                intrinsic_reward = self.compute_intrinsic(z_mean, z_var, z_mean_post, z_var_post).detach()
            else:
                reward_pred_rnn = self.rew_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                intrinsic_reward = (reward_pred_rnn - pred_rewardss) ** 2

                for number, id in enumerate(indices):
                    reward_pred = self.baseline_reward_predictors[id].forward(obs[number * b:(number + 1) * b],
                                                                              actions[number * b:(number + 1) * b])
                    rew_loss_2 = (pred_rewardss[number * b:(number + 1) * b] -
                                  reward_pred) ** 2
                    intrinsic_reward[number * b:(number + 1) * b] -= rew_loss_2
                    if self.consider_dynamics:
                        trans_pred = self.baseline_trans_predictors[id].forward(obs[number * b:(number + 1) * b],
                                                                                actions[number * b:(number + 1) * b])
                        trans_loss_2 = torch.mean((trans_pred - next_obs[number * b:(number + 1) * b]) ** 2, dim=1,
                                                  keepdim=True)
                        intrinsic_reward[number * b:(number + 1) * b] -= trans_loss_2

                # pred_rew = self.rew_decoder.forward(z_sample_pearl_post.detach(), obs, actions)
                # reward_pred_rnn = self.reward_predictor.forward_seq(context_rnn)
                # intrinsic_reward = (pred_rew - pred_rewardss) ** 2 - (reward_pred_rnn - pred_rewardss) ** 2
                # intrinsic_reward = - (reward_pred_rnn - pred_rewardss) ** 2
                if self.consider_dynamics:
                    trans_pred_rnn = self.transition_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                    intrinsic_reward = intrinsic_reward + torch.mean((trans_pred_rnn - next_obs) ** 2, dim=1,
                                                                     keepdim=True)
                    # intrinsic_reward = intrinsic_reward - torch.mean((trans_pred_rnn - next_obs) ** 2, dim=1, keepdim=True)

                '''pred_rew = self.rew_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                reward_pred_rnn = self.reward_predictor.forward_seq(context_rnn)
                intrinsic_reward = (pred_rew - pred_rewardss) ** 2 - (reward_pred_rnn - pred_rewardss) ** 2
                if self.consider_dynamics:
                    pred_trans = self.transition_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                    trans_pred_rnn = self.transition_predictor.forward_seq(context_rnn)
                    intrinsic_reward = intrinsic_reward + torch.mean((pred_trans - next_obs) ** 2, dim=1, keepdim=True) - torch.mean((trans_pred_rnn - next_obs) ** 2, dim=1, keepdim=True)'''

            intrinsic_reward = intrinsic_reward.view(t * b, -1)
            if self.intrinsic_reward_decay != 1:
                intrinsic_reward = intrinsic_reward * torch.unsqueeze(
                    ptu.from_numpy(self.intrinsic_reward_decay ** np.linspace(0, t * b - 1, t * b)), 1)
            rew = intrinsic_reward * self.intrinsic_reward_weight + agent_rew
        else:
            rew = agent_rew
        rew = rew.detach()
        # print(z_mean.shape, z_mean_next.shape, obs.shape, t, b)
        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred_exp = self.qf1_exp(torch.cat([obs, actions, z_mean.detach(), z_var.detach()], dim=1))
        q2_pred_exp = self.qf2_exp(torch.cat([obs, actions, z_mean.detach(), z_var.detach()], dim=1))
        v_pred_exp = self.vf_exp(torch.cat([obs, z_mean.detach(), z_var.detach()], dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            #print(next_obs.shape,z_mean_post.shape)
            target_v_values = self.target_exp_vf(torch.cat([next_obs, z_mean_post, z_var_post], dim=1))

        # KL constraint on z if probabilistic

        self.qf1_exp_optimizer.zero_grad()
        self.qf2_exp_optimizer.zero_grad()
        rewards_flat = rew
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss_exp = torch.mean((q1_pred_exp - q_target) ** 2) + torch.mean((q2_pred_exp - q_target) ** 2)
        qf_loss_exp.backward()

        self.qf1_exp_optimizer.step()
        self.qf2_exp_optimizer.step()


        # compute min Q on the new actions
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q_exp(obs, new_actions, z_mean.detach(), z_var.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss_exp = self.vf_exp_criterion(v_pred_exp, v_target.detach())
        self.vf_exp_optimizer.zero_grad()
        vf_loss_exp.backward()
        self.vf_exp_optimizer.step()
        self._update_target_network_exp()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss_exp = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss_exp = policy_loss_exp + policy_reg_loss

        self.policy_exp_optimizer.zero_grad()
        policy_loss_exp.backward()
        self.policy_exp_optimizer.step()

        if self.eval_statistics_2 is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics_2 = OrderedDict()

            self.eval_statistics_2['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics_2['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics_2['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics_2['QF Loss Exp'] = np.mean(ptu.get_numpy(qf_loss_exp))
            self.eval_statistics_2['VF Loss Exp'] = np.mean(ptu.get_numpy(vf_loss_exp))
            self.eval_statistics_2['Policy Loss Exp'] = np.mean(ptu.get_numpy(
                policy_loss_exp
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions Exp',
                ptu.get_numpy(q1_pred_exp),
            ))


            self.eval_statistics_2['KL Divergence'] = ptu.get_numpy(kl_div)
            self.eval_statistics_2['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions Exp',
                ptu.get_numpy(v_pred_exp),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics_2['Z mean train'] = np.mean(ptu.get_numpy(z_mean))
            self.eval_statistics_2['Z variance train'] = np.mean(ptu.get_numpy(z_var))
            self.eval_statistics_2['reward prediction loss'] = ptu.get_numpy(rew_loss)
            if self.consider_dynamics:
                self.eval_statistics_2['transisition prediction loss'] = ptu.get_numpy(trans_loss)


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            context_encoder=self.context_encoder.state_dict(),
            qf1_exp=self.qf1_exp.state_dict(),
            qf2_exp=self.qf2_exp.state_dict(),
            policy_exp=self.exploration_agent.state_dict(),
            vf_exp=self.vf_exp.state_dict(),
            target_vf_exp=self.target_exp_vf.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
        )
        return snapshot

class ExpSACFinSubtract4(ExpAlgorithmFin2):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            nets_exp,
            encoder,
            latent_dim,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            entropy_weight=1e-2,
            intrinsic_reward_weight=1e-1,
            use_kl_div_intrinsic=False,
            prediction_reward_scale=1,
            consider_dynamics=0,
            prediction_transition_scale=1,
            intrinsic_reward_decay=1,
            reward_predictor=None,
            transition_predictor=None,
            baseline_reward_predictors=None,
            baseline_trans_predictors=None,
            baseline_reward_predictor=None,
            baseline_trans_predictor=None,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            agent_exp=nets_exp[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            encoder=encoder,
            **kwargs
        )
        self.baseline_reward_predictor = baseline_reward_predictor
        self.baseline_trans_predictor = baseline_trans_predictor
        self.baseline_reward_predictors = baseline_reward_predictors
        self.baseline_trans_predictors = baseline_trans_predictors
        self.reward_predictor = reward_predictor
        self.transition_predictor = transition_predictor
        self.use_kl_div_intrinsic = use_kl_div_intrinsic
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.entropy_weight = entropy_weight
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.latent_dim = latent_dim
        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.qf_exp_criterion = nn.MSELoss()
        self.vf_exp_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.pred_loss = nn.MSELoss()
        self.kl_lambda = kl_lambda
        self.prediction_reward_scale = prediction_reward_scale
        self.consider_dynamics = consider_dynamics
        self.prediction_transition_scale = prediction_transition_scale

        self.use_information_bottleneck = use_information_bottleneck
        self.use_next_obs_in_context = use_next_obs_in_context
        self.intrinsic_reward_decay=intrinsic_reward_decay

        self.qf1, self.qf2, self.vf = nets[1:]
        self.qf1_exp, self.qf2_exp, self.vf_exp, self.rew_decoder, self.transition_decoder = nets_exp[1:]
        self.target_exp_vf = self.vf_exp.copy()
        self.target_vf = self.vf.copy()


        self.policy_exp_optimizer = optimizer_class(
            self.exploration_agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_exp_optimizer = optimizer_class(
            self.qf1_exp.parameters(),
            lr=qf_lr,
        )
        self.qf2_exp_optimizer = optimizer_class(
            self.qf2_exp.parameters(),
            lr=qf_lr,
        )
        self.vf_exp_optimizer = optimizer_class(
            self.vf_exp.parameters(),
            lr=vf_lr,
        )
        self.policy_optimizer = optimizer_class(
            self.agent.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.context_encoder.parameters(),
            lr=context_lr,
        )
        self.rew_optimizer = optimizer_class(
            self.rew_decoder.parameters(),
            lr=context_lr,
        )
        self.transition_optimizer = optimizer_class(
            self.transition_decoder.parameters(),
            lr=context_lr,
        )
        self.rew_predictor_optimizer = optimizer_class(
            self.reward_predictor.parameters(),
            lr=context_lr,
        )
        self.transition_predictor_optimizer = optimizer_class(
            self.transition_predictor.parameters(),
            lr=context_lr,
        )




    ###### Torch stuff #####
    @property
    def networks(self):
        return  [self.context_encoder] + [self.exploration_agent.policy] + [self.qf1_exp, self.qf2_exp, self.vf_exp, self.target_exp_vf,self.rew_decoder,self.transition_decoder] + [self.agent.policy,self.qf1, self.qf2, self.vf, self.target_vf]+[self.reward_predictor, self.transition_predictor]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        sr = batch['rewards'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t, sr]

    def unpack_batch_context(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        sparse_r = batch['rewards'][None, ...]
        r = batch['rewards'][None, ...]
        #sparse_r = r
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        info = batch['env_infos'][None, ...]
        # print(o[0,:5],a[0,:5],r[0],sparse_r[0],no[0,:5])
        return [o, a, sparse_r, no, t, info, r]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense

        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for
                       idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        return unpacked

    def sample_context(self, indices, sequence=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx
                   in indices]
        context = [self.unpack_batch_context(batch) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        context = torch.cat(context[:-4], dim=2)
        return context, context_unbatched

    def pred_context(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0], 1, context[2].shape[2])
        tmp = torch.cat([r_0, context[2]], dim=1)
        a_0 = ptu.zeros(context[1].shape[0], 1, context[1].shape[2])
        tmp2 = torch.cat([a_0, context[1]], dim=1)
        tmp3 = torch.cat([torch.unsqueeze(context[0][:, 0, :], 1), context[3]], dim=1)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([tmp3, tmp2, tmp], dim=2)
        return contextr

    def pred_context_rnn(self, context):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        r_0 = ptu.zeros(context[2].shape[0], 1, context[2].shape[2])
        tmp = torch.cat([r_0, context[2][:,:-1,:]], dim=1)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        contextr = torch.cat([context[0], context[1], tmp], dim=2)
        return contextr

    def sample_exp(self, indices,sequence=True):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.exp_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch_context(batch) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        context_unbatched = context
        context = torch.cat(context[:-4], dim=2)
        return  context,context_unbatched


    ##### Training #####
    def _do_training(self, indices,it_):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        #_,exp_context_unbatched = self.sample_exp(indices,True)
        #exp_context_pred = self.pred_context(exp_context_unbatched)
        _, context_unbatched = self.sample_context(indices, True)
        context_pred = self.pred_context(context_unbatched)
        context = self.sample_sac(indices)
        context_rnn = self.pred_context_rnn(context)
        #context_new_pred = self.pred_context(context)
        context_new_pred = context_pred
        # zero out context and hidden encoder state

        '''indice2 = self.train_tasks
        for i in range(1):
            context2 = self.sample_sac(indices)
            self._take_step_baseline(indice2,context2)'''



        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):

            #self._take_step(indices, context_unbatched,context_pred)
            #self._take_step_exp(indices, context_unbatched,context_pred,context,context_rnn,context_new_pred)
            self._take_step_exp(indices, context_unbatched, context_pred, context, context_rnn)

            # stop backprop


    def _min_q_exp(self,  obs,actions,z_mean,z_var):
        #print(obs.shape,actions.shape)

        q1 = self.qf1_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        q2 = self.qf2_exp(torch.cat([ obs,actions,z_mean,z_var],dim=1))
        min_q = torch.min(q1, q2)
        return min_q


    def _min_q(self, obs, actions,z):
        #print(obs.shape,actions.shape)

        q1 = self.qf1(torch.cat([obs, actions,z],dim=1))
        q2 = self.qf2(torch.cat([obs, actions,z],dim=1))
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network_exp(self):
        ptu.soft_update_from_to(self.vf_exp, self.target_exp_vf, self.soft_target_tau)

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def compute_kl(self,means,vars):
        std_mean = ptu.zeros(means.size())
        std_var = ptu.ones(means.size())
        tem = vars / std_var
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((std_mean - means) / std_var) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        kl_div = torch.mean(kl_div)
        return kl_div

    def compute_intrinsic(self,z_mean_prev, z_var_prev,z_mean_post,z_var_post):
        tem = z_var_post / z_var_prev
        kl_div = tem ** 2 - 2 * torch.log(tem) + ((z_mean_prev - z_mean_post) / z_var_prev) ** 2 - 1
        kl_div = torch.sum(kl_div, dim=1, keepdim=True) / 2
        return kl_div

    def _take_step_exp(self, indices,context_unbatched,context_pred,context,context_rnn):

        t, b, _ = context_pred.size()
        b = b - 1
        context_pred_pre = context_pred [:,:-1,:]
        #context_pred = context_pred.contiguous()
        z_s = self.context_encoder.forward_seq(context_pred_pre)
        #z_s = z_s.view(t, b, -1)
        z_mean = z_s[:, :self.latent_dim]
        z_var = torch.nn.functional.softplus(z_s[:, self.latent_dim:])
        # print(z_mean.shape,z_var.shape)
        z_dis = torch.distributions.Normal(z_mean, torch.sqrt(z_var))
        z_sample = z_dis.rsample()
        z_sample_pearl = z_sample

        obs, actions, agent_rew, next_obs, terms, sr = context
        pred_rewardss = agent_rew

        t, b, _ = obs.size()
        #agent_rew = agent_rew.contiguous()
        #pred_rewardss = pred_rewardss.contiguous()
        agent_rew = agent_rew.view(t * b, -1)
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        pred_rewardss = pred_rewardss.view(t * b, -1)

        rewards_flat = agent_rew.detach()

        q1_pred = self.qf1(torch.cat([obs, actions, z_sample_pearl], dim=1))
        q2_pred = self.qf2(torch.cat([obs, actions, z_sample_pearl], dim=1))
        v_pred = self.vf(torch.cat([obs, z_sample_pearl.detach()], dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            target_v_values = self.target_vf(torch.cat([next_obs, z_sample_pearl], dim=1))

        # KL constraint on z if probabilistic


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        self.context_optimizer.zero_grad()
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)

        kl_div = self.compute_kl(z_mean, z_var)
        kl_loss = kl_div * self.kl_lambda
        kl_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        policy_outputs, _ = self.agent(obs, z_sample_pearl.detach())

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q(obs, new_actions, z_sample_pearl.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        rew_pred = self.rew_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
        self.rew_optimizer.zero_grad()
        rew_loss = self.pred_loss(pred_rewardss, rew_pred) * self.prediction_reward_scale
        rew_loss.backward()
        self.rew_optimizer.step()

        '''self.reward_predictor.reset(num_tasks=t)
        reward_pred_rnn = self.reward_predictor.forward_seq(context_rnn)
        self.rew_predictor_optimizer.zero_grad()
        rew_predict_loss = self.pred_loss(pred_rewardss, reward_pred_rnn) * self.prediction_reward_scale
        rew_predict_loss.backward()
        self.rew_predictor_optimizer.step()'''


        if self.consider_dynamics:
            self.transition_optimizer.zero_grad()
            trans_pred = self.transition_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
            trans_loss = self.pred_loss(next_obs, trans_pred) * self.prediction_transition_scale
            trans_loss.backward()
            self.transition_optimizer.step()

            '''self.transition_predictor.reset(num_tasks=t)
            trans_pred_rnn = self.transition_predictor.forward_seq(context_rnn)
            self.transition_predictor_optimizer.zero_grad()
            trans_predict_loss = self.pred_loss(next_obs, trans_pred_rnn) * self.prediction_reward_scale
            trans_predict_loss.backward()
            self.transition_predictor_optimizer.step()'''
        indice_batch = ptu.zeros(obs.shape[0], len(self.train_tasks))
        for number, id in enumerate(indices):
            indice_batch[number * b:(number + 1) * b, id] = 1
        for _ in range(1):
            self.rew_predictor_optimizer.zero_grad()
            reward_pred = self.reward_predictor.forward(indice_batch,obs,actions)
            rew_loss_2 = self.pred_loss(pred_rewardss,
                                          reward_pred) * self.prediction_reward_scale
            rew_loss_2.backward()
            self.rew_predictor_optimizer.step()
            if self.consider_dynamics:
                self.transition_predictor_optimizer.zero_grad()
                trans_pred = self.transition_predictor.forward(indice_batch, obs, actions)
                trans_loss_2 = self.pred_loss(next_obs,trans_pred) * self.prediction_transition_scale
                trans_loss_2.backward()
                self.transition_predictor_optimizer.step()

        policy_outputs, _ = self.exploration_agent(obs, z_mean.detach(), z_var.detach())

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        context_post = context_pred[:, 1:, :]
        context_post = context_post.contiguous()
        z_s_post = self.context_encoder.forward_seq(context_post)
        z_mean_post = z_s_post[:, :self.latent_dim]
        z_var_post = torch.nn.functional.softplus(z_s_post[:, self.latent_dim:])
        if self.intrinsic_reward_weight > 0:
            if self.use_kl_div_intrinsic:


                intrinsic_reward = self.compute_intrinsic(z_mean, z_var, z_mean_post, z_var_post).detach()
            else:
                reward_pred_rnn = self.rew_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                intrinsic_reward = (reward_pred_rnn - pred_rewardss) ** 2


                reward_pred = self.reward_predictor.forward(indice_batch,obs,actions)
                rew_loss_2 = (pred_rewardss-
                                  reward_pred) ** 2
                intrinsic_reward -= rew_loss_2

                # pred_rew = self.rew_decoder.forward(z_sample_pearl_post.detach(), obs, actions)
                # reward_pred_rnn = self.reward_predictor.forward_seq(context_rnn)
                # intrinsic_reward = (pred_rew - pred_rewardss) ** 2 - (reward_pred_rnn - pred_rewardss) ** 2
                # intrinsic_reward = - (reward_pred_rnn - pred_rewardss) ** 2
                if self.consider_dynamics:
                    trans_pred = self.transition_predictor.forward(indice_batch, obs, actions)
                    trans_loss_2 = torch.mean((trans_pred - next_obs) ** 2, dim=1,
                                              keepdim=True)
                    intrinsic_reward -= trans_loss_2
                    trans_pred_rnn = self.transition_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                    intrinsic_reward = intrinsic_reward + torch.mean((trans_pred_rnn - next_obs) ** 2, dim=1,
                                                                     keepdim=True)
                    # intrinsic_reward = intrinsic_reward - torch.mean((trans_pred_rnn - next_obs) ** 2, dim=1, keepdim=True)

                '''pred_rew = self.rew_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                reward_pred_rnn = self.reward_predictor.forward_seq(context_rnn)
                intrinsic_reward = (pred_rew - pred_rewardss) ** 2 - (reward_pred_rnn - pred_rewardss) ** 2
                if self.consider_dynamics:
                    pred_trans = self.transition_decoder.forward(z_mean.detach(),z_var.detach(), obs, actions)
                    trans_pred_rnn = self.transition_predictor.forward_seq(context_rnn)
                    intrinsic_reward = intrinsic_reward + torch.mean((pred_trans - next_obs) ** 2, dim=1, keepdim=True) - torch.mean((trans_pred_rnn - next_obs) ** 2, dim=1, keepdim=True)'''

            intrinsic_reward = intrinsic_reward.view(t * b, -1)
            if self.intrinsic_reward_decay != 1:
                intrinsic_reward = intrinsic_reward * torch.unsqueeze(
                    ptu.from_numpy(self.intrinsic_reward_decay ** np.linspace(0, t * b - 1, t * b)), 1)
            rew = intrinsic_reward * self.intrinsic_reward_weight + agent_rew
        else:
            rew = agent_rew
        rew = rew.detach()
        # print(z_mean.shape, z_mean_next.shape, obs.shape, t, b)
        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred_exp = self.qf1_exp(torch.cat([obs, actions, z_mean.detach(), z_var.detach()], dim=1))
        q2_pred_exp = self.qf2_exp(torch.cat([obs, actions, z_mean.detach(), z_var.detach()], dim=1))
        v_pred_exp = self.vf_exp(torch.cat([obs, z_mean.detach(), z_var.detach()], dim=1))
        # get targets for use in V and Q updates

        with torch.no_grad():
            #print(next_obs.shape,z_mean_post.shape)
            target_v_values = self.target_exp_vf(torch.cat([next_obs, z_mean_post, z_var_post], dim=1))

        # KL constraint on z if probabilistic

        self.qf1_exp_optimizer.zero_grad()
        self.qf2_exp_optimizer.zero_grad()
        rewards_flat = rew
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(t * b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss_exp = torch.mean((q1_pred_exp - q_target) ** 2) + torch.mean((q2_pred_exp - q_target) ** 2)
        qf_loss_exp.backward()

        self.qf1_exp_optimizer.step()
        self.qf2_exp_optimizer.step()


        # compute min Q on the new actions
        new_actions = new_actions.view(t * b, -1)
        min_q_new_actions = self._min_q_exp(obs, new_actions, z_mean.detach(), z_var.detach())

        # vf update
        # print(min_q_new_actions)
        # print(log_pi)
        v_target = min_q_new_actions - log_pi
        vf_loss_exp = self.vf_exp_criterion(v_pred_exp, v_target.detach())
        self.vf_exp_optimizer.zero_grad()
        vf_loss_exp.backward()
        self.vf_exp_optimizer.step()
        self._update_target_network_exp()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss_exp = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss_exp = policy_loss_exp + policy_reg_loss

        self.policy_exp_optimizer.zero_grad()
        policy_loss_exp.backward()
        self.policy_exp_optimizer.step()

        if self.eval_statistics_2 is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics_2 = OrderedDict()

            self.eval_statistics_2['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics_2['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics_2['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics_2['QF Loss Exp'] = np.mean(ptu.get_numpy(qf_loss_exp))
            self.eval_statistics_2['VF Loss Exp'] = np.mean(ptu.get_numpy(vf_loss_exp))
            self.eval_statistics_2['Policy Loss Exp'] = np.mean(ptu.get_numpy(
                policy_loss_exp
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Q Predictions Exp',
                ptu.get_numpy(q1_pred_exp),
            ))


            self.eval_statistics_2['KL Divergence'] = ptu.get_numpy(kl_div)
            self.eval_statistics_2['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'V Predictions Exp',
                ptu.get_numpy(v_pred_exp),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics_2.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics_2['Z mean train'] = np.mean(ptu.get_numpy(z_mean))
            self.eval_statistics_2['Z variance train'] = np.mean(ptu.get_numpy(z_var))
            self.eval_statistics_2['reward prediction loss'] = ptu.get_numpy(rew_loss)
            #self.eval_statistics_2['reward prediction loss baseline'] = ptu.get_numpy(rew_loss_2)
            if self.consider_dynamics:
                self.eval_statistics_2['transisition prediction loss'] = ptu.get_numpy(trans_loss)
                #self.eval_statistics_2['transisition prediction loss baseline'] = ptu.get_numpy(trans_loss_2)


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            context_encoder=self.context_encoder.state_dict(),
            qf1_exp=self.qf1_exp.state_dict(),
            qf2_exp=self.qf2_exp.state_dict(),
            policy_exp=self.exploration_agent.state_dict(),
            vf_exp=self.vf_exp.state_dict(),
            target_vf_exp=self.target_exp_vf.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
        )
        return snapshot
