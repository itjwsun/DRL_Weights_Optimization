# -*- coding: utf-8 -*-
"""
Create on 2022/2/22 18:17

@author: sjw
"""
import numpy as np
from weights_env import WeightsEnv
from ddpg import DDPG


class RunThis(object):

    def __init__(self):
        self._train_episode = 500  # 训练的episode数
        self._env = WeightsEnv()  # 初始化环境
        self._a_bound = 1  # 动作向量范围
        self._s_dim, self._a_dim, self._a_bound = self._env.s_dim, self._env.a_dim, self._a_bound
        self._train_steps = self._test_steps = 200  # 每个episode的训练步长
        self._exploration = 2  # 探索率
        self._models_n = self._env._models_n  # state向量长度
        self.test_episode = 10  # 测试集的次数
        self._state = []  # 存储状态值的矩阵
        self._reward = []  # 存储奖励值的列表
        self._all_reward = []
        self.test_reward = []  # 记录测试集的奖励
        self.test_state = []  # 记录累计状态
        self.test_all_reward = []  # 记录每个eposide的累计奖励
        self._train_agent = DDPG(self._s_dim, self._a_dim, self._a_bound)  # 初始化DDPG算法 -- 训练阶段

    def train(self):
        '''智能体和环境交互
        '''
        for episode in range(self._train_episode):
            ep_reward = 0.  # 一个episode内的总奖励
            ep_steps = 0  # 一个episode内的总步数
            state = self._env.reset()  # 随机初始化权重
            step_reward = []  # 记录每一步的奖励
            for step in range(self._train_steps):
                action = self._train_agent.choose_action(state)
                action = np.clip(np.random.normal(action, self._exploration), -self._a_bound, self._a_bound)
                next_state, reward, _ = self._env.step(action)
                # 向经验回放池里添加数据
                self._train_agent.store_transition(state, action, reward / 10, next_state)
                ep_reward += reward  # 记录累计奖励和步数
                if self._train_agent.pointer > self._train_agent.memory_size:
                    self._exploration *= 0.99995
                    self._train_agent.learn()  # 训练算法
                state = next_state  # 改变状态
                self._state.append(list(state))
                step_reward.append(reward)
                ep_steps += 1
            self._reward.append(ep_reward)
            self._all_reward.append(step_reward)
            print('【Episode】 ', episode, '【Reward】 %.3f' %
                  (ep_reward), '【Explore】 %.3f' % self._exploration)
        self._state = np.array(self._state)
        np.savetxt('../../data/' + self.v + '/state_24h.csv', self._state,
                   fmt='%.5f', encoding='utf-8', delimiter=',')
        np.savetxt('../../data/' + self.v + '/reward_24h.csv',
                   self._reward, fmt='%.5f', encoding='utf-8', delimiter=',')
        np.savetxt('../../data/' + self.v + '/step_reward_24h.csv', self._all_reward, fmt='%.5f', encoding='utf-8',
                   delimiter=',')

    def test(self):
        '''利用训练好的模型测试
        '''
        for i in range(self.test_episode):
            ep_reward_lst = []
            ep_reward = 0
            state = self._env.reset()  # 刷新环境
            for j in range(self._test_steps):
                action = self._train_agent.choose_action(state)
                state, reward, _ = self._env.step(action)
                ep_reward += reward
                ep_reward_lst.append(reward)
                self.test_state.append(list(state))
            self.test_all_reward.append(ep_reward)
            self.test_reward.append(ep_reward_lst)
            print('【Episode】 ', i, '【Reward】 %.5f' % ep_reward)
        np.savetxt("../../data/" + self.v + "/state_test_24h.csv",
                   np.array(self.test_state), fmt="%.5f", encoding='utf-8', delimiter=',')
        np.savetxt('../../data/' + self.v + '/reward_test_24h.csv',
                   np.array(self.test_all_reward), fmt='%.5f', encoding='utf-8', delimiter=',')
        np.savetxt('../../data/' + self.v + '/reward_steps_test_24h.csv',
                   np.array(self.test_all_reward), fmt='%.5f', encoding='utf-8', delimiter=',')

    def run(self):
        '''入口函数
        '''
        self.train()
        self.test()


if __name__ == '__main__':
    run_this = RunThis()
    run_this.run()
