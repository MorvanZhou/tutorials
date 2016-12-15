import gym
from RL_brain import QTable, SarsaTable
import time

def q_learning(env):
    RL = QTable(actions=list(range(env.action_space.n)))
    for i_episode in range(10000):
        observation = env.reset()
        t = 0
        while True:
            # time.sleep(0.05)
            env.render()
            action = RL.choose_action(str(observation))
            observation, reward, done, info = env.step(action)
            if done:
                break
            else:
                t += 1

def sarsa(env):
    RL = SarsaTable(actions=list(range(env.action_space.n)))
    for i_episode in range(10000):
        observation = env.reset()
        action = RL.choose_action(str(observation))
        t = 0
        while True:
            # time.sleep(0.05)
            env.render()
            action = RL.choose_action(str(observation))
            observation, reward, done, info = env.step(action)
            if done:
                break
            else:
                t += 1

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')

    # # action: [-1, 0, 1] m/s
    # print(env.action_space.n)
    #
    # # observation: [position, velocity]
    # print('observation_high: ', env.observation_space.high)
    # print('observation_low: ', env.observation_space.low)

    q_learning(env)