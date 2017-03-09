"""
Function approximation with eligibility trace,
Not working!!!! Not converging !!!!

The cart pole example
"""


import gym
from Eligibility_trace_function_approximation import FunctionEligibility

env = gym.make('CartPole-v0')


RL = FunctionEligibility(env.action_space.n, 4,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      e_greedy_increment=0.001,
                      # output_graph=True
                      )

for i_episode in range(500):

    s = env.reset()
    a = RL.choose_action(s)
    is_zero_e = True
    while True:
        env.render()

        s_, r, done, info = env.step(a)
        a_ = RL.choose_action(s_)

        # x, x_dot, theta, theta_dot = s_
        #
        # # the smaller theta and closer to center the better
        #
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # r = r1 + r2
        if done: r = -2

        RL.learn(s, a, r, s_, a_, is_zero_e)
        is_zero_e = False
        if done:
            print(i_episode)
            break

        s, a = s_, a_

