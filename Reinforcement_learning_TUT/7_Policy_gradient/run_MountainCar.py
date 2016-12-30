"""
Deep Q network,

The mountain car example
"""


import gym
from RL_brain import PolicyGradient

env = gym.make('MountainCar-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=len(env.observation_space.high),
    learning_rate=0.01,
    reward_decay=0.99,
    output_graph=False,
)

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
            print('episode: ', i_episode, "step: ", RL.learn_steps,
                  'cost: ', round(RL.cost, 4), ' epsilon: ', round(RL.epsilon, 2))

        if done:
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
