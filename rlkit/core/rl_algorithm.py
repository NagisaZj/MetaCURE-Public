import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer,PERMultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler, SeedInPlacePathSampler, ExpInPlacePathSamplerSimple
from rlkit.torch import pytorch_util as ptu
import torch


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            seed_sample=False,
            sample_interval=5,
            last_reward_only=False,
            metaworld_sparse=False
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.exploration_agent = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.last_reward_only=last_reward_only

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.metaworld_sparse = metaworld_sparse

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
            metaworld_sparse = metaworld_sparse
        )
        self.seed_sample = seed_sample
        self.sample_interval = sample_interval
        if self.seed_sample:
            self.seedsampler = SeedInPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
                sample_interval=sample_interval
        )
        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def reset_task(self,idx):
        if hasattr(self.env,'tasks_pool'):
            self.env.set_task(self.env.tasks_pool[idx])
        else:
            self.env.reset_task(idx)

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.reset_task(idx)
                    if not self.seed_sample:
                        self.collect_data(self.num_initial_steps, 1, np.inf)
                    else:
                        self.collect_data(int(self.num_initial_steps/2), 1, np.inf)
                        self.collect_data_seed(int(self.num_initial_steps / 2), 1, np.inf)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if not self.seed_sample:
                    if self.num_steps_prior > 0:
                        self.collect_data(self.num_steps_prior, 1, np.inf)
                    # collect some trajectories with z ~ posterior
                    if self.num_steps_posterior > 0:
                        self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                    if self.num_extra_rl_steps_posterior > 0:
                        self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                          add_to_enc_buffer=False)
                else:
                    if self.num_steps_prior > 0:
                        self.collect_data(int(self.num_steps_prior/2), 1, np.inf)
                        self.collect_data_seed(int(self.num_steps_prior / 2), 1, np.inf)
                    # collect some trajectories with z ~ posterior
                    if self.num_steps_posterior > 0:
                        self.collect_data(int(self.num_steps_posterior/2), 1, self.update_post_train)
                        self.collect_data_seed(int(self.num_steps_posterior / 2), 1, self.update_post_train)
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                    if self.num_extra_rl_steps_posterior > 0:
                        self.collect_data(int(self.num_extra_rl_steps_posterior/2), 1, self.update_post_train,
                                          add_to_enc_buffer=False)
                        self.collect_data_seed(int(self.num_extra_rl_steps_posterior / 2), 1, self.update_post_train,
                                         add_to_enc_buffer=False)

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def collect_data_seed(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.seedsampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            if not self.seed_sample:
                path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            else:
                path, num = self.seedsampler.obtain_samples(deterministic=self.eval_deterministic,
                                                        max_samples=self.num_steps_per_eval - num_transitions,
                                                        max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        if hasattr(self.env, 'tasks_pool'):
            p = paths[-1]
            done = np.sum(e['success'] for e in p['env_infos'])
            done = 1 if done >0 else 0
            p['done'] = done
        else:
            p = paths[-1]
            p['done'] = 0


        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        if hasattr(self.env, '_goal'):
            goal = self.env._goal
            for path in paths:
                path['goal'] = goal  # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        final_returns_last = []
        online_returns = []
        success_cnt=[]
        for idx in indices:
            all_rets = []
            success_single = 0
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                success_single = success_single + paths[-1]['done']
            final_returns_last.append(np.mean([a[-1] for a in all_rets]))
            final_returns.append(np.mean([np.mean(a) for a in all_rets]))
            success_cnt.append(success_single/self.num_evals)
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        #print(success_cnt, len(success_cnt))
        return final_returns, online_returns,final_returns_last, success_cnt

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 5,
                                                        accum_context=False,
                                                        resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns, train_final_returns_last,train_success_cnt = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns, test_final_returns_last, test_success_cnt = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        #if hasattr(self.env, "log_diagnostics"):
        #    self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_return_last = np.mean(train_final_returns_last)
        avg_test_return_last = np.mean(test_final_returns_last)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        self.eval_statistics['AverageReturn_all_train_tasks_last'] = avg_train_return_last
        self.eval_statistics['AverageReturn_all_test_tasks_last'] = avg_test_return_last
        if hasattr(self.env,'tasks_pool'):
            self.eval_statistics['AverageSuccessRate_all_train_tasks'] = np.mean(train_success_cnt)
            self.eval_statistics['AverageSuccessRate_all_test_tasks'] = np.mean(test_success_cnt)
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass




class ExpAlgorithmFin2(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            agent_exp,
            train_tasks,
            eval_tasks,
            encoder,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            load_SMM =False,
            use_history=False,
            SMM_path=None,
            num_skills = 1,
            meta_episode_len=10,
            num_trajs = 2,
            num_trajs_init=5,
            num_trajs_eval=1,
            use_all_trajs=False,
            flush=1000000000,
            rsample_rate=5,
            pie_hidden_dim=64,
            rsample_rate_eval = 1,
            metaworld_sparse=False
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        #print(len(eval_tasks))
        self.env = env
        self.agent = agent
        self.exploration_agent = agent_exp # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.context_encoder = encoder
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.metaworld_sparse=metaworld_sparse

        self.eval_statistics = None
        self.eval_statistics_2 = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.load_SMM = load_SMM
        self.use_history = use_history,
        self.SMM_path = SMM_path
        self.num_skills = num_skills
        self.meta_episode_len = meta_episode_len
        self.num_trajs = num_trajs
        self.num_trajs_init = num_trajs_init
        self.num_trajs_eval = num_trajs_eval
        self.use_all_trajs = use_all_trajs
        self.flush = flush
        self.rsample_rate = rsample_rate
        self.rsample_rate_eval = rsample_rate_eval

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
            metaworld_sparse = metaworld_sparse
        )
        self.expsampler = ExpInPlacePathSamplerSimple(
            env=env,
            policy=self.exploration_agent,
            encoder=self.context_encoder,
            max_path_length=self.max_path_length,
            metaworld_sparse = metaworld_sparse
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update

        self.replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )
        self.enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )


        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx


    def reset_task(self,idx):
        if hasattr(self.env,'tasks_pool'):
            self.env.set_task(self.env.tasks_pool[idx])
        else:
            self.env.reset_task(idx)

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.reset_task(idx)
                    for _ in range(self.num_trajs_init):
                        self.collect_data_exp(self.meta_episode_len)
                    self.collect_data(self.num_initial_steps, 1, np.inf)

            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.reset_task(idx)
                if (it_+1)% self.flush ==0:
                    self.enc_replay_buffer.task_buffers[idx].clear()
                #if (it_+1)%5==0:
                #    self.enc_replay_buffer.task_buffers[idx].clear()
                for _ in range(self.num_trajs):
                    self.collect_data_exp(self.meta_episode_len)
                    self.collect_data_exp_posterior(self.meta_episode_len, idx)
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                    # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                    # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                      add_to_enc_buffer=False)
            print('collect over')

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices,it_)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples
        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            #self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context,context_unbatched = self.sample_context(self.task_idx)
                context_pred = self.pred_context(context_unbatched)
                self.agent.infer_posterior(context_pred)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')


    def collect_data_exp(self, num_episodes):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.exploration_agent.clear_z()
        paths, n_samples = self.expsampler.obtain_samples(max_trajs=num_episodes,rsample=self.rsample_rate)


        self.enc_replay_buffer.add_paths(self.task_idx, paths)
        self.replay_buffer.add_paths(self.task_idx, paths)
        self._n_env_steps_total += n_samples
        gt.stamp('sample')

    def collect_data_exp_posterior(self, num_episodes,idx):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.exploration_agent.clear_z()
        context, context_unbatched = self.sample_context(idx)
        context_pred = self.pred_context(context_unbatched)
        self.exploration_agent.infer_posterior(context_pred)
        paths, n_samples = self.expsampler.obtain_samples(max_trajs=num_episodes,rsample=self.rsample_rate)


        self.enc_replay_buffer.add_paths(self.task_idx, paths)
        self.replay_buffer.add_paths(self.task_idx, paths)
        self._n_env_steps_total += n_samples
        gt.stamp('sample')



    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate(epoch):
            self.evaluate(epoch,self.num_trajs)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self,epoch):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True #if (epoch+1)%5==0 else False

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.reset_task(idx)
        self.agent.clear_z()
        self.exploration_agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0

        path, num = self.expsampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_trajs=self.meta_episode_len,split=True,accum_context_for_agent=False,context_agent=self.agent,rsample=self.rsample_rate_eval)
        num_transitions += num
        num_trajs +=self.meta_episode_len
        paths+=path
        self.agent.infer_posterior(self.exploration_agent.context)
        path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
                                                accum_context=False)
        num_transitions += num
        num_trajs += 1
        paths += path



        if hasattr(self.env, '_goal'):
            goal = self.env._goal
            for path in paths:
                path['goal'] = goal  # goal

        if hasattr(self.env, 'tasks_pool'):
            p = paths[-1]
            done = np.sum(e['success'] for e in p['env_infos'])
            done = 1 if done > 0 else 0
            p['done'] = done
        else:
            p = paths[-1]
            p['done'] = 0

            # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        final_returns_last = []
        online_returns = []
        success_cnt = []
        for idx in indices:
            all_rets = []
            success_single = 0
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                success_single = success_single + paths[-1]['done']
            final_returns_last.append(np.mean([a[-1] for a in all_rets]))
            final_returns.append(np.mean([np.mean(a) for a in all_rets]))
            success_cnt.append(success_single / self.num_evals)
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        # print(success_cnt,len(success_cnt))
        return final_returns, online_returns, final_returns_last, success_cnt

    def evaluate(self, epoch,num_trajs):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        if self.eval_statistics_2 is None:
            self.eval_statistics_2 = OrderedDict()
        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.reset_task(idx)
            paths = []
            self.agent.clear_z()
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context, context_unbatched = self.sample_context(idx)
                context_pred = self.pred_context(context_unbatched)
                self.agent.infer_posterior(context_pred)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf)
                paths += p



            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        
        
        '''self.reset_task(indices[0])
        o = self.env.reset()
        self.exploration_agent.clear_z()

        a = np.zeros(self.exploration_agent.action_dim, dtype=np.float32)
        r = 0
        info = {"sparse_reward": 0}
        for i in range(1):
            self.exploration_agent.update_context([o, a, r, info])
            self.exploration_agent.infer_posterior(self.exploration_agent.context)
            a, agent_info = self.exploration_agent.get_action(o)
            # print(a)
            next_o, r, d, env_info = self.env.step(a)
            o = next_o
        self.exploration_agent.update_context([o, a, r, info])
        self.exploration_agent.infer_posterior(self.exploration_agent.context)
        z_mean,z_var = self.exploration_agent.z_means, self.exploration_agent.z_vars
        z_mean,z_var = z_mean.repeat(900,1),z_var.repeat(900,1)
        x = np.arange(30)*0.1-1.45
        y = np.arange(30) * 0.1 - 1.45
        X,Y = np.meshgrid(x,y)
        test_inputs = np.concatenate((X[...,np.newaxis],Y[...,np.newaxis]),2)
        test_inputs = test_inputs.reshape(900,2)
        noises = np.zeros((900,1))
        for i in range(900):
            x = test_inputs[i,0]
            y = test_inputs[i,1]
            if (x ** 2 + (y + 1) ** 2) ** 0.5 <= 0.3:
                noise_variance = 2
                noises[i,0] = np.random.rand() * noise_variance * 2
        test_inputs_ori = np.concatenate((test_inputs, noises), 1)
        test_inputs = ptu.from_numpy(test_inputs_ori)
        actions,_ = self.exploration_agent(test_inputs, z_mean, z_var)
        actions = actions[0]

        next_obs = test_inputs.clone().detach()
        next_obs[:,:2] = test_inputs[:,:2]+actions/10
        next_obs[:,2] = 0
        next_obs_np = ptu.get_numpy(next_obs)
        print(ptu.get_numpy(test_inputs)[0, :],ptu.get_numpy(actions)[0, :], next_obs_np[0,:])
        for i in range(900):
            x, y = next_obs_np[i,0],next_obs_np[i,1]
            dist = (x ** 2 + (y+1) ** 2) ** 0.5
            if dist <= 0.3:
                noise_variance = 2
                next_obs_np[i, 2] = np.random.rand() * noise_variance * 3
        goals = self.env._goal
        dist = -((next_obs_np[:,0]-goals[0])**2 + (next_obs_np[:,1]-goals[1])**2)**0.5
        r_true = self.env.sparsify_rewards(dist)
        for i in range(900):
            if r_true[i]<0:
                r_true[i]+=1
        r_true = r_true[...,np.newaxis]
        logger.save_extra_data(r_true,
                               path='eval_trajectories/task{}-epoch{}-run{}-rew-true'.format(0, epoch, 0))
        label = np.ones((900, 1), dtype=np.float32) * float(indices[0]) / len(self.train_tasks)

        next_obs = ptu.from_numpy(next_obs_np)
        r_true = ptu.from_numpy(r_true)
        #actions = ptu.from_numpy(actions)
        label = ptu.from_numpy(label)


        
        print(test_inputs.shape,actions.shape,r_true.shape,next_obs.shape,label.shape,z_mean.shape,z_var.shape)
        rew_prediction = self.reward_predictor.forward( test_inputs, actions,z_mean,z_var)
        trans_prediction = self.transition_predictor.forward( test_inputs, actions,z_mean, z_var)
        #rew_baseline = self.reward_predictor.forward(label,test_inputs,actions)
        #trans_baseline = self.transition_predictor.forward(label, test_inputs, actions)
        rew_baseline = self.baseline_reward_predictors[indices[0]].forward(test_inputs,actions)
        trans_baseline = self.baseline_trans_predictors[indices[0]].forward( test_inputs, actions)
        print(rew_prediction.shape, trans_prediction.shape, rew_baseline.shape, trans_baseline.shape)

        intrinsic_reward_pred = (rew_prediction-r_true) **2 + torch.mean((trans_prediction - next_obs) ** 2, dim=1,
                                                                     keepdim=True)
        intrinsic_reward_baseline = (rew_baseline-r_true) **2 + torch.mean((trans_baseline - next_obs) ** 2, dim=1,
                                                         keepdim=True)
        print(intrinsic_reward_pred.shape, intrinsic_reward_baseline.shape)
        intrinsic_reward = intrinsic_reward_pred - intrinsic_reward_baseline


        intrinsic_reward,intrinsic_reward_pred,intrinsic_reward_baseline = ptu.get_numpy(intrinsic_reward),ptu.get_numpy(intrinsic_reward_pred),ptu.get_numpy(intrinsic_reward_baseline)
        rew_prediction,trans_prediction,rew_baseline,trans_baseline = ptu.get_numpy(rew_prediction),ptu.get_numpy(trans_prediction),ptu.get_numpy(rew_baseline),ptu.get_numpy(trans_baseline)
        logger.save_extra_data(intrinsic_reward, path='eval_trajectories/task{}-epoch{}-run{}-intr'.format(0, epoch, 0))
        logger.save_extra_data(intrinsic_reward_pred,
                               path='eval_trajectories/task{}-epoch{}-run{}-intr-pred'.format(0, epoch, 0))
        logger.save_extra_data(intrinsic_reward_baseline,
                               path='eval_trajectories/task{}-epoch{}-run{}-intr-baseline'.format(0, epoch, 0))
        logger.save_extra_data(self.env._goal,
                               path='eval_trajectories/task{}-epoch{}-run{}-goal'.format(0, epoch, 0))
        logger.save_extra_data(rew_prediction, path='eval_trajectories/task{}-epoch{}-run{}-rew-pred'.format(0, epoch, 0))
        logger.save_extra_data(trans_prediction,
                               path='eval_trajectories/task{}-epoch{}-run{}-trans-pred'.format(0, epoch, 0))
        logger.save_extra_data(rew_baseline,
                               path='eval_trajectories/task{}-epoch{}-run{}-rew-baseline'.format(0, epoch, 0))
        logger.save_extra_data(trans_baseline,
                               path='eval_trajectories/task{}-epoch{}-run{}-trans-baseline'.format(0, epoch, 0))'''
        
        
        '''for idx in indices:
            self.task_idx = idx
            self.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context,context_unbatched = self.sample_context(idx)
                context_pred = self.pred_context(context_unbatched)
                self.agent.infer_posterior(context_pred)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        train_returns_exp = []
        for idx in indices:
            self.task_idx = idx
            self.reset_task(idx)
            paths = []
            for _ in range(1):
                #context, context_unbatched = self.sample_context(idx)
                #context_pred = self.pred_context(context_unbatched)
                self.exploration_agent.infer_posterior(context_pred)
                p, _ = self.expsampler.obtain_samples(deterministic=self.eval_deterministic,
                                               max_trajs=self.meta_episode_len, split=True,rsample=self.rsample_rate_eval)

                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns_exp.append(eval_util.get_average_returns(paths))
        train_returns_exp = np.mean(train_returns_exp)'''
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns, train_final_returns_last,train_success_cnt = self._do_eval(indices, epoch)
        '''eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)
        eval_util.dprint('train online returns exp')
        eval_util.dprint(train_returns_exp)'''

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns, test_final_returns_last, test_success_cnt = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.exploration_agent.log_diagnostics(self.eval_statistics)
        self.agent.log_diagnostics(self.eval_statistics)
        #if hasattr(self.env, "log_diagnostics"):
        #    self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_return_last = np.mean(train_final_returns_last)
        avg_test_return_last = np.mean(test_final_returns_last)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        #self.eval_statistics['AverageExpTrainReturn_all_train_tasks'] = train_returns_exp
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        self.eval_statistics['AverageReturn_all_train_tasks_last'] = avg_train_return_last
        self.eval_statistics['AverageReturn_all_test_tasks_last'] = avg_test_return_last
        if hasattr(self.env, 'tasks_pool'):
            self.eval_statistics['AverageSuccessRate_all_train_tasks'] = np.mean(train_success_cnt)
            self.eval_statistics['AverageSuccessRate_all_test_tasks'] = np.mean(test_success_cnt)
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))


        for key, value in self.eval_statistics_2.items():
            logger.record_tabular(key, value)
        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)

        self.eval_statistics = None
        self.eval_statistics_2 = None
        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

