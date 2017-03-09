"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.7.3
"""

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.act_index = tf.placeholder(tf.int32, name="act")
        self.td_error = tf.placeholder(tf.float32, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.state,
            units=20,    # number of hidden units
            activation=tf.nn.relu,
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
            name='acts_prob'
        )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.act_index])
            self.exp_r = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_r)  # minimize(-exp_v) = maximize(exp_v)

    def update(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.state: s, self.act_index: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_r], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.state: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_next = tf.placeholder(tf.float32, [1, 1], name="v_next")
        self.r = tf.placeholder(tf.float32, name='r')

        l1 = tf.layers.dense(
            inputs=self.state,
            units=20,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        self.v = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='V'
        )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_next - self.v)
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def update(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_next = self.sess.run(self.v, {self.state: s_})
        td_error, loss, _ = self.sess.run([self.td_error, self.loss, self.train_op],
                                          {self.state: s, self.v_next: v_next, self.r: r})
        return td_error, loss


OUTPUT_GRAPH = False
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
EPISODE_TIME_THRESHOLD = 1000
RENDER = False  # rendering wastes time
GAMMA = 0.9

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible

sess = tf.Session()

with tf.variable_scope('Actor'):
    actor = Actor(sess, n_features=env.observation_space.shape[0], n_actions=env.action_space.n, lr=0.001)
with tf.variable_scope('Critic'):
    critic = Critic(sess, n_features=env.observation_space.shape[0], lr=0.01)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(3000):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error, loss = critic.update(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.update(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= EPISODE_TIME_THRESHOLD:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

