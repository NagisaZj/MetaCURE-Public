import numpy as np
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv

from . import register_env


@register_env('hopper-rand-params')
class HopperRandParamsWrappedEnv(HopperRandParamsEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True):
        super(HopperRandParamsWrappedEnv, self).__init__()
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 0.0
        reward = (posafter - posbefore) / self.dt
        dist = abs(reward - 1.5)
        if dist > 0.5:
            reward = 0
        else:
            reward = 0.8 - dist
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}