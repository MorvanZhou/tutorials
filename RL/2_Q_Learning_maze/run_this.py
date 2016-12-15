"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on 莫烦Python: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QTable


def update():
    for episode in range(100):
        # initial observation
        observation1 = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation1))

            # RL take action and get next observation and reward
            observation2, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation1), action, reward, str(observation2))

            # swap observation
            observation1 = observation2

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()