# -*- coding: utf-8 -*-
"""
Create on 2022/2/20 15:37

@author: sjw
"""
import numpy as np
from scipy import signal
from sklearn.metrics import mean_absolute_percentage_error as MAPE, mean_squared_error as MSE, \
    mean_absolute_error as MAE


class WeightsEnv(object):

    def __init__(self):
        self.time_ = 24
        self.v = "filename"
        self._predict_actual_result = np.loadtxt('../../data/' + self.v + '/models_prediction_data_24.csv',
                                                 delimiter=',', encoding='utf-8')  # 各个模型预测的结果和真实结果
        self._predict_result = self._predict_actual_result[:, :-2][:self.time_]  # 各个模型的预测结果
        self._minimum_result = self._predict_actual_result[:, -2][:self.time_]  # 最小负荷
        self._actual_result = self._predict_actual_result[:, -1][:self.time_]  # 真实负荷
        self._models_n = self.s_dim = self.a_dim = self._predict_result.shape[1]  # 模型数目 = 状态维度 = 动作维度
        self.P_ = np.zeros([self._actual_result.shape[0]], dtype='float32')  # 状态更新下的热负荷
        self.min_w, self.max_w = -1, 1
        self.cal_best_ESR_CCI()

    def error(self):
        error = self._actual_result - self._minimum_result
        print(error)
        error_idx = np.argwhere(error > 0.5)[:, 0]
        for idx in error_idx:
            self._minimum_result[idx] = self._actual_result[idx]
        print(self._actual_result - self._minimum_result)

    def cal_best_ESR_CCI(self):
        self.best_ESR, self.best_CCI = self.cal_ESR_CCI(self._actual_result, self._minimum_result)

    def cal_ESR_CCI(self, y, y_hat):
        """计算预测曲线和实际曲线的节能率和互相关系数
        """
        y_hat_1, y_hat_N = y_hat[0], y_hat[-1]  # 获取到第一个预测值和最后一个预测值
        y_hat_middle = y_hat[1: -1]  # 获取到 (1-N) 之间的预测值
        y_1, y_N = y[0], y[-1]  # 获取到第一个真实值和最后一个真实值
        y_middle = y[1: -1]  # 获取到 (1-N) 之间的真实值
        E_hat = 0.5 * (y_hat_1 + y_hat_N) + np.sum(y_hat_middle)
        E = 0.5 * (y_1 + y_N) + np.sum(y_middle)
        ESR = 1 - (E_hat / E)
        ESR = np.round(ESR, 4)
        n = len(y)
        sum_x_y, sum_x, sum_y, sum_x_2, sum_y_2 = 0, 0, 0, 0, 0
        for x_i, y_i in zip(y_hat, y):
            sum_x_y += x_i * y_i
            sum_x += x_i
            sum_y += y_i
            sum_x_2 += x_i ** 2
            sum_y_2 += y_i ** 2
        fm1 = (n * sum_x_2) - sum_x ** 2
        fm2 = (n * sum_y_2) - sum_y ** 2
        CCI = (n * sum_x_y - sum_x * sum_y) / np.sqrt(fm1 * fm2)
        CCI = np.round(CCI, 4)
        return ESR, CCI

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return 1 * (x > 0) * x

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def reset(self):
        '''初始化函数
        :return: state: [w1, ..., w10]
        '''
        self.state = np.random.rand(self._models_n) * (self.max_w - self.min_w) + self.min_w
        self.state /= np.sum(self.state)
        return self.state

    def step(self, action):
        '''智能体和环境的一次交互
        P_ = w1 * P1 + ... + w6 * P6
        :param: action: [delta_w1, ..., delta_w10] -- action: [-1, 1]
        :return: [next_state, reward, done, info]
        '''
        self.state += action  # 更新state
        self.state = self.sigmoid(self.state)  # 将state归一化到0-1之间，且按照概率分布
        self.state /= np.sum(self.state)
        # 计算下一个状态下的实际热负荷
        for i in range(self._models_n):
            self.P_ += self.state[i] * self._predict_result[:, i]
        # 仅仅判断当前的动作 delta 向量取值是否为优选
        reward = - np.mean(np.square(self._minimum_result - self.P_))
        self.P_ = 0
        return self.state, reward, {}

    def choose_action(self):
        '''创建随机动作序列
        :return: action: [delta_w1, ..., delta_w6]
        '''
        return np.random.randn(self._models_n)


if __name__ == '__main__':
    env = WeightsEnv()
    state = env.reset()
    action = env.choose_action()
    for episode in range(100):
        ep_reward = 0
        for step in range(200):
            action = env.choose_action()
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
        print(ep_reward)
