"""
Deep Q network,

The cart pole example
"""


import gym
from RL_brain import DeepQNetwork
import time

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=len(env.observation_space.high),
                  learning_rate=0.01, e_greedy=0.99,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,
                  hidden_layers=[20, 20],)

total_steps = 0


for i_episode in range(100):

    observation = env.reset()

    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = observation_

        # the smaller theta and closer to center the better

        r1 = (env.x_threshold - abs(x))/env.x_threshold - 2
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()
            print('episode: ', i_episode,
                  'cost: ', round(RL.cost, 4),
                  ' epsilon: ', round(RL.epsilon, 2))

        if done:
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
