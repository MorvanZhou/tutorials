"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example (based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20Actor%20Critic%20Solution.ipynb)

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.7.3
"""

import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


class ActorContinue(object):
    def __init__(self, n_features, action_range, lr=0.001):
        with tf.name_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [n_features, ], "state")
            state = tf.expand_dims(self.state, axis=0)
            self.act = tf.placeholder(tf.float32, name="act")
            self.advantage = tf.placeholder(tf.float32, name="adv")  # TD_error

        mu_ = tf.layers.dense(
            inputs=state,
            units=20,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu_'
        )
        mu = tf.layers.dense(
            inputs=mu_,
            units=1,  # number of hidden units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma_ = tf.layers.dense(
            inputs=state,
            units=20,  # output units
            activation=None,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='sigma_'
        )
        sigma = tf.layers.dense(
            inputs=sigma_,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(.5),  # biases
            name='sigma'
        )

        self.mu, self.sigma = tf.squeeze(mu), tf.squeeze(sigma+1e-2)
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_range[0], action_range[1])

        with tf.name_scope('loss'):
            neg_log_prob = -self.normal_dist.log_prob(self.act)  # loss without advantage
            self.loss = tf.reduce_mean(neg_log_prob * self.advantage)  # advantage (TD_error) guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def update(self, s, a, adv):
        feed_dict = {self.state: s, self.act: a, self.advantage: adv}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def choose_action(self, s):
        return self.sess.run([self.action, self.mu, self.sigma], {self.state: s})  # get probabilities for all actions


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        with tf.name_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [n_features, ], "state")
            state = tf.expand_dims(self.state, axis=0)
            self.act_index = tf.placeholder(tf.int32, name="act")
            self.advantage = tf.placeholder(tf.float32, name="adv")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=state,
                units=20,    # number of hidden units
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

        with tf.name_scope('loss'):
            neg_log_prob = -tf.log(self.acts_prob[0, self.act_index])   # loss without advantage
            self.loss = tf.reduce_mean(neg_log_prob * self.advantage)  # advantage (TD_error) guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def update(self, s, a, adv):
        feed_dict = {self.state: s, self.act_index: a, self.advantage: adv}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, {self.state: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, n_features, lr=0.01):
        with tf.name_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [n_features, ], "state")
            state = tf.expand_dims(self.state, axis=0)
            self.target = tf.placeholder(dtype=tf.float32, name="target")  # TD target=r+gamma*V_next

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=state,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # open end
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.eval = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.eval))    # TD_error = (r+gamma*V_next) - V_eval
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def update(self, s, target):
        _, loss = self.sess.run([self.train_op, self.loss], {self.state: s, self.target: target})
        return loss

    def evaluate(self, s):
        return self.sess.run(self.eval, {self.state: s})[0, 0]  # return a float

