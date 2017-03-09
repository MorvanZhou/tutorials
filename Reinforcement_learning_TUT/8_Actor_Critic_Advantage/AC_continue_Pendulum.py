"""
Actor-Critic with continuous action using TD-error as the Advantage, Reinforcement Learning.

The cart pole example (based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20Actor%20Critic%20Solution.ipynb)

Cannot converge!!!

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.7.3
"""

import tensorflow as tf
import numpy as np
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


class Actor(object):
    def __init__(self, n_features, action_range, lr=0.0001):
        with tf.name_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [n_features, ], "state")
            state = tf.expand_dims(self.state, axis=0)
            self.act = tf.placeholder(tf.float32, name="act")
            self.advantage = tf.placeholder(tf.float32, name="adv")  # TD_error

        l1 = tf.layers.dense(
            inputs=state,
            units=30,  # number of hidden units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        mu = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.relu,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma'
        )

        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+1e-2)
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_range[0], action_range[1])

        with tf.name_scope('loss'):
            neg_log_prob = -self.normal_dist.log_prob(self.act)  # loss without advantage
            self.loss = neg_log_prob * self.advantage  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def update(self, s, a, adv):
        feed_dict = {self.state: s, self.act: a, self.advantage: adv}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def choose_action(self, s):
        return self.sess.run([self.action, self.mu, self.sigma], {self.state: s})  # get probabilities for all actions


class Critic(object):
    def __init__(self, n_features, lr=0.01):
        with tf.name_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [n_features, ], "state")
            state = tf.expand_dims(self.state, axis=0)
            self.target = tf.placeholder(dtype=tf.float32, name="target")  # TD target=r+gamma*V_next

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=state,
                units=30,  # number of hidden units
                activation=None,
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
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(self.loss)

    def update(self, s, target):
        _, loss = self.sess.run([self.train_op, self.loss], {self.state: s, self.target: target})
        return loss

    def evaluate(self, s):
        return self.sess.run(self.eval, {self.state: s})[0, 0]  # return a float


OUTPUT_GRAPH = False
EPISODE_TIME_THRESHOLD = 300
DISPLAY_REWARD_THRESHOLD = -550  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
GAMMA = 0.9

env = gym.make('Pendulum-v0')
# env.seed(1)  # reproducible

actor = Actor(n_features=env.observation_space.shape[0], action_range=[env.action_space.low[0], env.action_space.high[0]], lr=0.001)
critic = Critic(n_features=env.observation_space.shape[0], lr=0.002)

with tf.Session() as sess:
    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    actor.sess, critic.sess = sess, sess    # define the tf session
    tf.global_variables_initializer().run()

    for i_episode in range(3000):
        observation = env.reset()
        t = 0
        ep_rs = []
        while True:
            # if RENDER:
            env.render()
            action, mu, sigma = actor.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            reward /= 10
            TD_target = reward + GAMMA * critic.evaluate(observation_)    # r + gamma * V_next
            TD_eval = critic.evaluate(observation)    # V_now
            TD_error = TD_target - TD_eval

            actor.update(s=observation, a=action, adv=TD_error)
            critic.update(s=observation, target=TD_target)

            observation = observation_
            t += 1
            # print(reward)
            ep_rs.append(reward)
            if t > EPISODE_TIME_THRESHOLD:
                ep_rs_sum = sum(ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break

