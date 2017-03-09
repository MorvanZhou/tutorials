"""
Function approximation with backward-view TD(lambda),
on-policy SARSA with eligibility trace.

View more on 莫烦Python: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class FunctionEligibility:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            lambda_=0.9,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.lambda_ = lambda_

        # total learning step
        self.learn_step_counter = 0

        self.e = []
        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.n_actions], initializer=w_initializer,)
                b1 = tf.get_variable('b1', [1, self.n_actions], initializer=b_initializer, )
                out = tf.matmul(s, w1) + b1

                self.e.append(tf.get_variable('e_w1', [self.n_features, self.n_actions], initializer=tf.constant_initializer(0.), trainable=False))
                self.e.append(tf.get_variable('e_b1', [1, self.n_actions], initializer=tf.constant_initializer(0.), trainable=False))
            return out

        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [1, self.n_features], name='s')  # input State
        self.a = tf.placeholder(tf.int32, name='a')
        self.td_error = tf.placeholder(tf.float32, name='td')

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
             n_l1, w_initializer, b_initializer = 20, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
             self.q_eval = build_layers(self.s, n_l1, w_initializer, b_initializer)

        self.q_eval_wrt_a = self.q_eval[0, self.a]

        optimizer = tf.train.RMSPropOptimizer(self.lr)
        variables = tf.trainable_variables()
        q_grads = tf.gradients(self.q_eval_wrt_a, variables)
        self.e = [tf.assign(e, -self.td_error*(self.gamma * self.lambda_ * e + g)) for e, g in zip(self.e, q_grads)]
        self._train_op = optimizer.apply_gradients(zip(self.e, variables))

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self, s, a, r, s_, a_, is_zero_e):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        q_next = self.sess.run(self.q_eval, {self.s: s_})
        q_next_wrt_a_ = q_next[0, a_]
        q_eval_wrt_a = self.sess.run(self.q_eval_wrt_a, {self.s: s, self.a: a})
        td_error = r + self.gamma * q_next_wrt_a_ - q_eval_wrt_a

        feed_dict = {self.td_error: td_error, self.s: s, self.a: a}
        if is_zero_e:
            for e in self.e:
                feed_dict[e] = np.zeros(e.get_shape())
        _ = self.sess.run([ self._train_op], feed_dict)
        # print(test)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

if __name__ == '__main__':
    from maze_env import Maze

    def run_maze():
        step = 0
        for episode in range(300):
            s = env.reset()
            a = RL.choose_action(s)
            is_zero_e = True

            while True:
                env.render()

                s_, r, done = env.step(a)
                a_ = RL.choose_action(s_)

                RL.learn(s, a, r, s_, a_, is_zero_e)
                is_zero_e = False

                s, a = s_, a_

                if done:
                    break
                step += 1

        # end of game
        print('game over')
        env.destroy()



    # maze game
    env = Maze()
    RL = FunctionEligibility(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      # e_greedy_increment=0.001,
                      # output_graph=True
                      )
    env.after(50, run_maze)
    env.mainloop()
