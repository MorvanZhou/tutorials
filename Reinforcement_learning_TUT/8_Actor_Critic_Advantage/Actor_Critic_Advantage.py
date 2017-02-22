"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example (based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb)

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.7.3
"""

import tensorflow as tf
import gym

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.01):
        with tf.variable_scope('Actor'):
            self.tf_state = tf.placeholder(tf.int32, [1, n_features], "state")
            self.tf_acts = tf.placeholder(tf.int32, name="action")
            self.tf_advantage = tf.placeholder(tf.float32, name="advantage")

            self.all_act_prob = tf.layers.dense(
                inputs=self.tf_state,
                units=n_actions,    # number of output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(),    # weights
            )
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act_prob, labels=self.tf_acts)
            loss = tf.reduce_mean(log_prob * self.tf_advantage)  # advantage guided loss


class Critic(object):
    def __init__(self, lr=0.01):
        pass


def run(actor, critic, env):
    for i_episode in range(3000):
        pass

env = gym.make('CartPole-v0')
env.seed(2)  # reproducible, general Policy gradient has high variance
actor = Actor(n_features=len(env.observation_space.high), lr=0.01)
critic = Critic(lr=0.01)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    run(actor, critic, env)
