"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
# env.seed(2)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=len(env.observation_space.high),
    learning_rate=0.02,
    reward_decay=0.7,
    # output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # x, x_dot, theta, theta_dot = observation_
        # # the smaller theta and closer to center the better
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # reward = r1 + r2

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
