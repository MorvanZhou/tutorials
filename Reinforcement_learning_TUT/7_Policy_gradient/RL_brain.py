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
            reward_decay=0.9,
            e_greedy=0.9,
            hidden_layers=[10, 10],
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.hidden_layers = hidden_layers
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_steps = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features*2+2)))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.train.SummaryWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def choose_action(self, observation):
        if type(observation) is list:
            observation = np.array(observation)
        if observation.ndim < 2:
            # to have batch dimension when feed into tf placeholder
            observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_steps % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        batch_memory = self.memory.sample(self.batch_size) if self.memory_counter > self.memory_size else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory.iloc[:, -self.n_features:],
                self.s: batch_memory.iloc[:, :self.n_features]
            })

        # change target w.r.t q_eval's action
        target = q_eval.copy()
        target[np.arange(self.batch_size, dtype=np.int32), batch_memory.iloc[:, self.n_features].astype(int)] = \
            batch_memory.iloc[:, self.n_features+1] + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: batch_memory.iloc[:, :self.n_features], self.q_target: target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_steps += 1

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()

    def _replace_target_params(self):
        replace_ops = []
        for layer, params in enumerate(self._eval_net_params):
            replace_op = [tf.assign(self._target_net_params[layer][W_b], params[W_b]) for W_b in range(2)]
            replace_ops.append(replace_op)
        self.sess.run(replace_ops)

    def _build_net(self):
        # create eval and target net weights and biases separately
        self._eval_net_params = []
        self._target_net_params = []

        # build evaluate_net
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            self.q_eval = self._build_layers(self.s, self.n_actions, trainable=True)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_sum(tf.square(self.q_target - self.q_eval))
            with tf.name_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # build target_net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            self.q_next = self._build_layers(self.s_, self.n_actions, trainable=False)

    def _build_layers(self, inputs, action_size, trainable):
        layers_output = [inputs]
        for i, n_unit in enumerate(self.hidden_layers):
            with tf.variable_scope('layer%i' % i):
                output = self._add_layer(
                    layers_output[i],
                    in_size=layers_output[i].get_shape()[1].value,
                    out_size=n_unit,
                    activation_function=tf.nn.relu,
                    trainable=trainable,
                )
                layers_output.append(output)
        with tf.variable_scope('output_layer'):
            output = self._add_layer(
                layers_output[-1],
                in_size=layers_output[-1].get_shape()[1].value,
                out_size=action_size,
                activation_function=None,
                trainable=trainable
            )
        return output

    def _add_layer(self, inputs, in_size, out_size, activation_function=None, trainable=True):

        # create weights and biases
        Weights = tf.get_variable(
            name='weights',
            shape=[in_size, out_size],
            trainable=trainable,
            initializer=tf.truncated_normal_initializer(mean=0., stddev=0.5)
        )
        biases = tf.get_variable(
            name='biases',
            shape=[out_size],
            initializer=tf.constant_initializer(0.1),
            trainable=trainable
        )

        # record parameters
        if trainable is True:
            self._eval_net_params.append([Weights, biases])
        else:
            self._target_net_params.append([Weights, biases])

        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # activation function
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


if __name__ == '__main__':
    dqn = DeepQNetwork(4,4)