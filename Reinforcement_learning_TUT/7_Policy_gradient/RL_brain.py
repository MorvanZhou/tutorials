"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

View more on 莫烦Python: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            batch_size=32,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size

        self.transitions = None

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, observation):
        a0_prob = self.sess.run(self.prediction, {self.x_inputs: observation[np.newaxis, :]})
        action = 0 if np.random.uniform() < a0_prob else 1  # p=1, a=0; and p=0, a=1
        return action

    def store_transition(self, s, a, r):
        transition = np.hstack((s, a, r))   # observation, action_label, reward
        if not hasattr(self, 'transition_buffer'):
            self.transition_buffer = [transition]   # is a list
        else:
            self.transition_buffer.append(transition)   # is a list

    def episode_reward_decay(self):
        episode_transitions = np.vstack(self.transition_buffer)
        episode_r = episode_transitions[:, -1]

        # decay the future rewards
        discounted_episode_r = np.zeros_like(episode_r)
        running_add = 0
        for t in reversed(range(0, episode_r.size)):
            running_add = running_add * self.gamma + episode_r[t]
            discounted_episode_r[t] = running_add

        # normalize and reduce variance
        discounted_episode_r -= np.mean(discounted_episode_r)
        discounted_episode_r /= np.std(discounted_episode_r)

        # re assign discounted_episode_r to reward column
        episode_transitions[:, -1] = discounted_episode_r

        # put episode transitions into self.transitions
        self.transitions = episode_transitions if self.transitions is None else np.vstack((self.transitions, episode_transitions))

        # empty the buffer for next episode
        self.transition_buffer = []

    def learn(self):
        batch_episodes_s = self.transitions[:, :self.n_features]
        batch_episodes_a = self.transitions[:, -2:-1]
        batch_episodes_dr = self.transitions[:, -1:]
        # action 0 has target 1 (100% to take action 0)
        # action 1 has target 0 (0% to take action 0)
        batch_episodes_fake_target = - (batch_episodes_a - 1)

        _, loss = self.sess.run([self._train_op, self.loss], {
            self.x_inputs: batch_episodes_s,
            self.fake_targets: batch_episodes_fake_target,
            self.advantages: batch_episodes_dr
        })

        self.transitions = None
        return loss

    def _build_net(self):
        # build evaluate_net
        self.x_inputs = tf.placeholder(tf.float32, [None, self.n_features], name='x_inputs')    # observation
        self.fake_targets = tf.placeholder(tf.float32, [None, 1], name='fake_targets')  # fake targets
        self.advantages = tf.placeholder(tf.float32, [None, 1], name="advantages")  # advantages

        l1 = self._add_layer('hidden0', self.x_inputs, self.n_features, 10, tf.nn.relu)     # hidden layer 1
        self.prediction = self._add_layer('output', l1, 10, 1, tf.nn.sigmoid)  # predicting for action 0
        with tf.name_scope('loss'):
            loglik = self.fake_targets*tf.log(self.prediction) + (1 - self.fake_targets)*tf.log(1-self.prediction)  #
            self.loss = -tf.reduce_mean(loglik * self.advantages)
        with tf.name_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def _add_layer(self, layer_name, inputs, in_size, out_size, activation_function=None):
        with tf.variable_scope(layer_name):
            # create weights and biases
            Weights = tf.get_variable(
                name='weights',
                shape=[in_size, out_size],
                initializer=tf.truncated_normal_initializer(mean=0., stddev=0.3)
            )
            biases = tf.get_variable(
                name='biases',
                shape=[out_size],
                initializer=tf.constant_initializer(0.1),
            )

            Wx_plus_b = tf.matmul(inputs, Weights) + biases

            # activation function
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            return outputs




