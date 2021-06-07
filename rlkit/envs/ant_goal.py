import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal')
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):
        super(AntGoalEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        a = np.random.random(num_tasks) * 2 * np.pi
        r = 3 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

@register_env('ant-goal-sparse')
class AntGoalEnvSparse(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, goal_radius=0.8, **kwargs):
        self.goal_radius = goal_radius
        super(AntGoalEnvSparse, self).__init__(task, n_tasks, **kwargs)


    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        #mask = (r >= -self.goal_radius).astype(np.float32)
        #r = r * mask
        if r < -self.goal_radius:
            r = -2
        r = r + 2
        return r

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        sparse_reward = self.sparsify_rewards(reward) - ctrl_cost - contact_cost + survive_reward
        reward = sparse_reward
        # make sparse rewards positive
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            sparse_reward=sparse_reward
        )

    def sample_tasks(self, num_tasks):
        a = np.random.random(num_tasks) * 1 * np.pi
        r = 1
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])