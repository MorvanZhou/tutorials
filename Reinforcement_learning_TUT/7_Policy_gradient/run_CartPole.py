"""
Deep Q network,

The cart pole example
"""


import gym
from RL_brain import PolicyGradient

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=len(env.observation_space.high),
    learning_rate=0.01,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(10000):

    observation = env.reset()

    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # x, x_dot, theta, theta_dot = observation_
        #
        # # the smaller theta and closer to center the better
        #
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 2
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians
        # reward = r1 + r2

        RL.store_transition(observation, action, reward)

        if done:
            RL.episode_reward_decay()

            if i_episode % RL.batch_size:
                loss = RL.learn()
                print(loss)

            break

        observation = observation_
