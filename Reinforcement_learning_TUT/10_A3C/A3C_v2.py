"""
Asynchronous Advantage Actor Critic (A3C), Reinforcement Learning.

The Pendulum example. Version 2: convergence not promised

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 300
MAX_GLOBAL_EP = 800
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
LR = 0.001    # learning rate for actor

env = gym.make(GAME)
env.seed(1)  # reproducible

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
    def __init__(self, scope, n_s, n_a,
                 a_bound=None, sess=None,
                 opt=None, global_params=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_s], 'S')
                self._build_net(n_a)
                self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:   # local net, calculate losses
            self.sess = sess
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_s], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, n_a], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'R')

                mu, sigma, self.v = self._build_net(n_a)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                self.mu, self.sigma = tf.squeeze(mu * a_bound[1]), tf.squeeze(sigma+1e-2)
                self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
                with tf.name_scope('a_loss'):
                    log_prob = self.normal_dist.log_prob(self.a_his)
                    self.exp_v = tf.reduce_mean(log_prob * td)
                    self.exp_v += 0.01*self.normal_dist.entropy()  # encourage exploration
                    self.a_loss = -self.exp_v
                with tf.name_scope('total_loss'):
                    self.loss = self.a_loss + 0.5*self.c_loss   # if combine, hard to converge
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(self.normal_dist.sample([1]), a_bound[0], a_bound[1])

                with tf.name_scope('local_grad'):
                    self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                    self.grads = tf.gradients(self.loss, self.params)  # get local gradients

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, global_params)]
                with tf.name_scope('push'):
                    self.update_op = opt.apply_gradients(zip(self.grads, global_params))

    def _build_net(self, n_a):
        w_init = tf.random_normal_initializer(0., .1)
        l = tf.layers.dense(self.s, 100, tf.nn.relu, kernel_initializer=w_init, name='l1')
        l = tf.layers.dense(l, 100, tf.nn.relu, kernel_initializer=w_init, name='l2')
        mu = tf.layers.dense(l, n_a, tf.nn.tanh, kernel_initializer=w_init, name='mu')
        sigma = tf.layers.dense(l, n_a, tf.nn.sigmoid, kernel_initializer=w_init, name='sigma')  # use sigmoid, don't need too large variance
        v = tf.layers.dense(l, 1, kernel_initializer=w_init, name='v')  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})


class Worker(object):
    def __init__(self, env, name, n_s, n_a, a_bound, sess, opt, g_params):
        self.env = env
        self.sess = sess
        self.name = name
        self.AC = ACNet(name, n_s, n_a, a_bound, sess, opt, g_params)

    def work(self, update_iter, max_ep_step, gamma, coord):
        total_step = 1
        buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []

        while not coord.should_stop() and GLOBAL_EP.eval(self.sess) < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(max_ep_step):
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                r /= 10
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_s_.append(s_)

                if total_step % update_iter == 0 or done:   # update global and assign to local net
                    buffer_s, buffer_a, buffer_r, buffer_s_ = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_r), np.vstack(buffer_s_)

                    v_next = self.sess.run(self.AC.v, {self.AC.s: buffer_s_})
                    if done: v_next[-1, 0] = 0
                    v_target = buffer_r + gamma * v_next

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if ep_t == max_ep_step-1:
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP.eval(self.sess),
                        "| Ep_r: %.2f" % ep_r,

                          )
                    sess.run(COUNT_GLOBAL_EP)
                    break

if __name__ == "__main__":
    sess = tf.Session()

    with tf.device("/cpu:0"):
        GLOBAL_EP = tf.Variable(0, dtype=tf.int32, name='global_ep', trainable=False)
        COUNT_GLOBAL_EP = tf.assign(GLOBAL_EP, tf.add(GLOBAL_EP, tf.constant(1), name='step_ep'))
        OPT = tf.train.RMSPropOptimizer(LR)
        globalAC = ACNet(GLOBAL_NET_SCOPE, N_S, N_A)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(
                Worker(
                    gym.make(GAME).unwrapped, i_name, N_S, N_A, A_BOUND, sess,
                    OPT, globalAC.params
                ))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work(UPDATE_GLOBAL_ITER, MAX_EP_STEP, GAMMA, coord)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

