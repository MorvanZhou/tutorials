"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example. Convergence promised, but difficult environment, A3C with continuous action hard to converge.

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
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 300
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
UPDATE_TARGET_ITER = 300
SHARED_COUNTER = 0
GLOBAL_EP = 0
GAMMA = 0.9
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
    def __init__(self, scope, n_s, n_a, a_bound):

        if scope == GLOBAL_NET_SCOPE:   # get global network, has target and eval net
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_s], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, n_a], 'A')
                self.s_ = tf.placeholder(tf.float32, [None, n_s], 'S_')
                self.q_ = self._build_net(n_s, n_a, a_bound, is_global=True)
                self.ae_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor/eval')
                self.at_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor/target')
                self.ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic/eval')
                self.ct_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic/target')
                
                with tf.name_scope('update_target'):
                    self.update_a_target_op = [g_t.assign(g_e) for g_t, g_e in zip(self.at_params, self.ae_params)]
                    self.update_c_target_op = [g_t.assign(g_e) for g_t, g_e in zip(self.ct_params, self.ce_params)]
                
        else:   # local net, only has eval net, no target
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_s], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, n_a], 'A')
                self.q_target = tf.placeholder(tf.float32, [None, 1], 'Qtarget')

                self.a_pred, self.q = self._build_net(n_s, n_a, a_bound, is_global=False)

                td = tf.subtract(self.q_target, self.q, name='TD_error')

                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.variable_scope('dQda'):
                    dQda = tf.gradients(self.q, self.a_his)[0]

                with tf.name_scope('local_grad'):
                    self.ae_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor/eval')
                    self.ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic/eval')
                    self.ae_grads = tf.gradients(ys=self.a_pred, xs=self.ae_params, grad_ys=dQda, name='ae_grads')
                    self.ce_grads = tf.gradients(self.c_loss, self.ce_params, name='ce_grads')

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_ae_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.ae_params, GLOBAL_AC.ae_params)]
                    self.pull_ce_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.ce_params, GLOBAL_AC.ce_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.ae_grads, GLOBAL_AC.ae_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.ce_grads, GLOBAL_AC.ce_params))

    def _build_net(self, n_s, n_a, a_bound, is_global):
        w_init = tf.random_normal_initializer(0., .1)

        def build_a(name, s):
            with tf.variable_scope(name):
                l_a = tf.layers.dense(s, 100, tf.nn.relu, kernel_initializer=w_init, name='la')
                a = tf.layers.dense(l_a, n_a, tf.nn.tanh, kernel_initializer=w_init, name='a')
                a = tf.multiply(a, a_bound[1], name='a_scaled')
            return a

        def build_c(name, s, a):
            n_l1 = 100
            with tf.variable_scope(name):
                w1_s = tf.get_variable('w1_s', [n_s, n_l1], initializer=w_init, )  # combine a and s as inputs
                w1_a = tf.get_variable('w1_a', [n_a, n_l1], initializer=w_init, )
                b1 = tf.get_variable('b1', [1, n_l1], initializer=w_init, )
                l_c = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)  # the a is from real
                q = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='Q')  # Q
            return q

        if is_global:
            with tf.variable_scope('actor'):
                build_a('eval', self.s)
                a_ = build_a('target', self.s_)
            with tf.variable_scope('critic'):
                build_c('eval', self.s, self.a_his)
                q_ = build_c('target', self.s_, a_)
            return q_
        else:
            with tf.variable_scope('actor'):
                a = build_a('eval', self.s)
            with tf.variable_scope('critic'):
                q = build_c('eval', self.s, self.a_his)
            return a, q

    def update_global_eval(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global_eval(self):  # run by a local
        SESS.run([self.pull_ae_params_op, self.pull_ce_params_op])

    def update_target(self):    # run by global
        SESS.run([self.update_a_target_op, self.update_c_target_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]  # single state
        return SESS.run(self.a_pred, feed_dict={self.s: s})[0]  # single action


class Worker(object):
    def __init__(self, env, name, n_s, n_a, a_bound):
        self.env = env
        self.name = name
        self.AC = ACNet(name, n_s, n_a, a_bound)

    def work(self, update_iter, max_ep_step, gamma, coord):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        var = 5  # control exploration
        buffer_s_ = []
        while not coord.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(max_ep_step):
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)
                a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration

                s_, r, done, info = self.env.step(a)
                r /= 10     # normalize reward
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                buffer_s_.append(s_)
                if total_step % update_iter == 0 or done:   # update global and assign to local net
                    var = max([.999*var, 0.1])  # decay the action randomness
                    # if done:
                    #     q_s_ = 0   # terminal
                    # else:
                    #     q_s_ = SESS.run(GLOBAL_AC.q_, {GLOBAL_AC.s_: s_[np.newaxis, :]})[0, 0]
                    # buffer_q_target = []
                    # for r in buffer_r[::-1]:    # reverse buffer r
                    #     q_s_ = r + gamma * q_s_
                    #     buffer_q_target.append(q_s_)
                    # buffer_q_target.reverse()

                    q_s_ = SESS.run(GLOBAL_AC.q_, {GLOBAL_AC.s_: np.vstack(buffer_s_)})
                    buffer_q_target = np.vstack(buffer_r) + GAMMA * q_s_
                    buffer_s_ = []

                    buffer_s, buffer_a, buffer_q_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_q_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.q_target: buffer_q_target,
                    }
                    self.AC.update_global_eval(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global_eval()

                global SHARED_COUNTER, GLOBAL_EP
                TARGET_UPDATE_LOCK.acquire()    # lock the update process
                if SHARED_COUNTER % UPDATE_TARGET_ITER == 0:
                    GLOBAL_AC.update_target()
                SHARED_COUNTER += 1
                TARGET_UPDATE_LOCK.release()

                s = s_
                total_step += 1

                if ep_t == max_ep_step-1:
                    GLOBAL_EP += 1
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %.2f" % ep_r,
                        '| Var: %.3f' % var,
                          )
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE, N_S, N_A, A_BOUND)  # we only need its params
        workers = []
        TARGET_UPDATE_LOCK = threading.Lock()
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(gym.make(GAME).unwrapped, i_name, N_S, N_A, A_BOUND,))

    coord = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work(UPDATE_GLOBAL_ITER, MAX_EP_STEP, GAMMA, coord)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

