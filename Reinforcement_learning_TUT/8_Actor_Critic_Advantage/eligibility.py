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
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001, lambda_=0):

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

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(lr)

            shape = [(4, 20), (20,), (20, 2), (2,)]
            self.eligibility = [tf.Variable(np.ones(s), dtype=np.float32) for s in shape]

            self.g_v = grads_and_vars = optimizer.compute_gradients(self.loss)
            self.new_grads_and_vars = []
            for i in range(len(self.eligibility)):
                self.eligibility[i] = lambda_ * self.eligibility[i] + grads_and_vars[i][0]
                self.new_grads_and_vars.append((self.eligibility[i], grads_and_vars[i][1]))
            self.train_op = optimizer.apply_gradients(self.new_grads_and_vars)

    def set_zero_eligibility(self):
        self.eligibility = [e*0 for e in self.eligibility]

    def update(self, s, a, adv):
        feed_dict = {self.state: s, self.act_index: a, self.advantage: adv}
        _, loss, e = self.sess.run([self.train_op, self.loss, self.eligibility], feed_dict)
        return loss, e

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


OUTPUT_GRAPH = False
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
EPISODE_TIME_THRESHOLD = 1000
RENDER = False  # rendering wastes time
GAMMA = 0.9

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible

actor = Actor(n_features=env.observation_space.shape[0], n_actions=env.action_space.n, lr=0.001)
critic = Critic(n_features=env.observation_space.shape[0], lr=0.01)     # we need a good teacher, so the teacher should learn faster than the actor

with tf.Session() as sess:
    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    actor.sess, critic.sess = sess, sess    # define the tf session
    tf.global_variables_initializer().run()

    for i_episode in range(3000):
        observation = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER: env.render()

            action = actor.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            # the smaller theta and closer to center, the better
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            track_r.append(reward)

            TD_target = reward + GAMMA * critic.evaluate(observation_)    # r + gamma * V_next
            TD_eval = critic.evaluate(observation)    # V_now
            TD_error = TD_target - TD_eval

            actor.update(s=observation, a=action, adv=TD_error)
            critic.update(s=observation, target=TD_target)

            observation = observation_
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

