"""
Deep Q network,

The mountain car example
"""


import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.0005, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0002,
                  hidden_layers=[20, 20])

total_steps = 0


for i_episode in range(10):

    observation = env.reset()

    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # the higher the better
        reward = abs(position - (-0.5))

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
