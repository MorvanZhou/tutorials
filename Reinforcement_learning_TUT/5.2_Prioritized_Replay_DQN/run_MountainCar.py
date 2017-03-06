"""
Deep Q network,

The mountain car example
"""


import gym
from RL_brain import DoubleDQNPrioritizedReplay, DeepQNetwork
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = gym.make('MountainCar-v0')
env.seed(1)
MEMORY_SIZE = 10000

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.005, e_greedy=0.9,
               reward_decay=0.9,
                  replace_target_iter=500, memory_size=MEMORY_SIZE,
                  e_greedy_increment=0.0001, sess=sess)

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DoubleDQNPrioritizedReplay(n_actions=3, n_features=2, learning_rate=0.005, e_greedy=0.9,
               reward_decay=0.9,
                  replace_target_iter=500, memory_size=MEMORY_SIZE,
                  e_greedy_increment=0.0001, double_q=False, sess=sess)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    for i_episode in range(10):
        observation = env.reset()
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done: reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode: ', i_episode,
                     ' epsilon: ', round(RL.epsilon, 2))
                break

            observation = observation_
            total_steps += 1
    return RL.qn

print('train natural DQN')
qn_natural = train(RL_natural)
print('train DQN prioritized')
qn_prio = train(RL_prio)

plt.plot(np.array(qn_natural), c='b', label='natural DQN')
plt.plot(np.array(qn_prio), c='r', label='DQN with prioritized replay')
plt.legend()
plt.ylabel('max q next')
plt.xlabel('training steps')
plt.grid()
plt.show()


