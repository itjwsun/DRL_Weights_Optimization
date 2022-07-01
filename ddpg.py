# -*- coding: utf-8 -*-
"""
Create on 2022/2/20 15:35

@author: sjw
"""
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

class DDPG(object):
    def __init__(self, s_dim, a_dim, a_bound):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # --------------------------- hyper parameters ------------------------
        self.lr_a = 0.001
        self.lr_c = 0.002
        self.gamma = 0.9
        self.tau = 0.1
        self.memory_size = 5000
        self.batch_size = 32
        self.memory = np.zeros((self.memory_size, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session(config=config)

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # ------------ create actor and critic network ----------------
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval_a', trainable=True)
            a_ = self._build_a(self.S_, scope='target_a', trainable=True)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval_c', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target_c', trainable=True)

        # ---------- get actor and critic network paramerters ------------
        self.ae_prarms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_a')
        self.at_prarms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_a')
        self.ce_prarms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_c')
        self.ct_prarms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_c')

        # ------------------------ soft replacement ----------------------
        self.soft_replace_a = [tf.assign(t, self.tau * e + (1 - self.tau) * t) for t, e in zip(self.at_prarms, self.ae_prarms)]
        self.soft_replace_c = [tf.assign(t, self.tau * e + (1 - self.tau) * t) for t, e in zip(self.ct_prarms, self.ce_prarms)]

        # ------------------------ optimizer network --------------------
        q_target = self.R + self.gamma * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=self.ce_prarms)
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(-tf.reduce_mean(q), var_list=self.ae_prarms)

        # ------------------------ run global variables -----------------
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        # s = StandardScaler().fit_transform(s[None, :]).squeeze()
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # update network parameterts
        self.sess.run(self.soft_replace_a)
        self.sess.run(self.soft_replace_c)

        # choose batch data to learn
        indices = np.random.choice(self.memory_size, self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # assign self.a = a in memory when calculating q for td_error,
        # otherwise the self.a is from Actor when updating Actor
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_size
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            hid1 = tf.layers.dense(s, units=256, activation=tf.nn.relu, name='al1', trainable=trainable)
            hid2 = tf.layers.dense(hid1, units=128, activation=tf.nn.relu, name='al2', trainable=trainable)
            output = tf.layers.dense(hid2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(output, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            input = tf.concat([s, a], axis=1)
            hid1 = tf.layers.dense(input, 256, activation=tf.nn.relu, name='cl1', trainable=trainable)
            hid2 = tf.layers.dense(hid1, 128, activation=tf.nn.relu, name='cl2', trainable=trainable)
            return tf.layers.dense(hid2, 1, activation=None, name='q', trainable=trainable)

