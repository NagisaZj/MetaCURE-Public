import numpy as np

from rlkit.samplers.util import rollout,seedrollout, exprolloutsimple, exprollout_splitsimple
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length,metaworld_sparse):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length
        self.metaworld_sparse = metaworld_sparse

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context,metaworld_sparse=self.metaworld_sparse)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total

class SeedInPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length,sample_interval=5):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length
        self.latent_dim = policy.latent_dim
        self.sample_interval = sample_interval
    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            self.random_seed = np.random.randn(1,self.latent_dim)
            path = seedrollout(
                self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context,random_seed=self.random_seed,sample_interval=self.sample_interval)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            #if n_trajs % resample == 0:
            #    policy.sample_z()
        return paths, n_steps_total

class ExpInPlacePathSamplerSimple(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length,encoder,metaworld_sparse=False):
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.max_path_length = max_path_length
        self.metaworld_sparse = metaworld_sparse

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context_for_agent=False, resample=1, context_agent = None,split=False,rsample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        policy.clear_z()
        #policy.reset_RNN()
        #self.env.reset_mask()
        if not split:
            path = exprolloutsimple(
            self.env, policy, max_path_length=self.max_path_length, max_trajs=max_trajs, accum_context_for_agent=accum_context_for_agent, context_agent = context_agent,rsample=rsample,metaworld_sparse = self.metaworld_sparse)
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1

            return paths, n_steps_total
        else:
            path = exprollout_splitsimple(
                self.env, policy, max_path_length=self.max_path_length, max_trajs=max_trajs,
                accum_context_for_agent=accum_context_for_agent, context_agent=context_agent,rsample=rsample,metaworld_sparse = self.metaworld_sparse)
            n_steps_total += self.max_path_length * max_trajs
            n_trajs += max_trajs

            return path, n_steps_total