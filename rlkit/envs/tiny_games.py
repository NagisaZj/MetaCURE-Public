
import numpy as np
from gym import spaces, Env
from . import register_env


@register_env('RPS')
class RockPaperScissorEnv(Env):
    def __init__(self,n_tasks, randomize_tasks=True):
        self.num_tasks = n_tasks
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.action_space = spaces.Box(low=-1, high=3, shape=(1,))
        self.goals = np.random.randint(0, 5, size=(n_tasks,)) * 0.5
        #super().__init__(self.num_tasks)


    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return  range(self.num_tasks)

    def reset(self):
        self._state = np.array([1],dtype=np.float32)
        return  self._state

    def _get_obs(self):
        return self._state

    def step(self, action):
        reward = 0
        #print(abs(action-self._goal))
        if abs(action-self._goal)<= 0.25:
            reward = 1
        else:
            reward = -1
            #reward = -1*abs(action-self._goal)[0]
        done = False
        ob = self._get_obs()
        #print(action,self._goal,reward)

        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)