import numpy as np
import random
import rlkit.data_management.sum_tree as sum_tree
from rlkit.data_management.replay_buffer import ReplayBuffer, PERReplayBuffer

class Experience(object):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, memory_size, batch_size, alpha):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha

    def add(self, data, priority):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority**self.alpha)

    def select(self, beta):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        
        Returns
        -------
        out : 
            list of samples
        weights: 
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """
        
        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating
            
        
        self.priority_update(indices, priorities) # Revert priorities

        weights /= max(weights) # Normalize for stability
        
        return out, weights, indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        
        Parameters
        ----------
        indices : 
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)
    
    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

        
            
        
class PERSimpleReplayBuffer(PERReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,info_dim=1, alpha=1,decay=0.9
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.info_dim = info_dim
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._env_infos = np.zeros((max_replay_buffer_size, info_dim))

        self._max_replay_buffer_size = max_replay_buffer_size
        self.alpha = alpha
        self.tree = sum_tree.SumTreeMine(max_replay_buffer_size)
        self._episode_starts = []
        self._cur_episode_start = 0
        self.decay = decay

    def add_sample(self, observation, action, reward, terminal,
                   next_observation,value, **kwargs):
        self._observations[self.tree.cursor] = observation
        self._actions[self.tree.cursor] = action
        self._rewards[self.tree.cursor] = reward
        self._terminals[self.tree.cursor] = terminal
        self._next_obs[self.tree.cursor] = next_observation
        self._sparse_rewards[self.tree.cursor] = kwargs['env_info'].get('sparse_reward', 0)
        self._env_infos[self.tree.cursor] = kwargs['env_info'].get('info', 0)
        self.tree.add(value**self.alpha)


    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self.tree.cursor

    def size(self):
        return self.tree.size

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p ** self.alpha)

    def random_batch(self, batch_size,beta):
        ''' batch of unordered transitions '''
        if self.tree.filled_size() < batch_size:
            return None, None, None


        indices = []
        weights = []
        priorities = []
        for _ in range(batch_size):
            r = random.random()
            priority, index = self.tree.find(r)
            priorities.append(priority*self.decay)
            #weights.append((1. / self._max_replay_buffer_size / priority) ** beta if priority > 1e-16 else 0)
            indices.append(index)

            self.priority_update([index], [0])  # To avoid duplicating
        #print([i<self.tree.size for i in indices])
        self.priority_update(indices, priorities)  # Revert priorities
        #m = max(weights)
        #weights = [i/m for i in weights]  # Normalize for stability
        return self.sample_data(indices), weights, indices


    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            #print(self._episode_starts)
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            sparse_rewards=self._sparse_rewards[indices],
            env_infos = self._env_infos[indices],
        )



    def num_steps_can_sample(self):
        return self.tree.size

