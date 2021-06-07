import numpy as np
from gym import spaces
from gym import Env

from . import register_env


@register_env('door-open')
class OpenDoorEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, n_tasks=2,**kwargs):
        np.random.seed(1337)
        goals = np.random.randint(0,3,n_tasks)
        self.goals = goals.tolist()

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=0, high=0.4, shape=(1,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.array([0,-1],dtype=np.float32)
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        action = np.sum(action //0.1)
        reward = 0
        if action ==3:
            self._state[1] = self._goal
        else:
            self._state[1] = -1
            reward = 1 if action == self._goal else -5
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(goal=self._goal)

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)



