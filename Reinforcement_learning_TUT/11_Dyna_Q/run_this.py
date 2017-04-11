"""
Simplest model-based RL, Dyna-Q.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on 莫烦Python: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable, EnvModel


def update():
    for episode in range(40):
        s = env.reset()
        while True:
            env.render()
            a = RL.choose_action(str(s))
            s_, r, done = env.step(a)
            RL.learn(str(s), a, r, str(s_))

            # use a model to output (r, s_) by inputting (s, a)
            # the model in dyna Q version is just like a memory replay buffer
            env_model.store_transition(str(s), a, r, s_)
            for n in range(10):     # learn 10 more times using the env_model
                ms, ma = env_model.sample_s_a()  # ms in here is a str
                mr, ms_ = env_model.get_r_s_(ms, ma)
                RL.learn(ms, ma, mr, str(ms_))

            s = s_
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env_model = EnvModel(actions=list(range(env.n_actions)))

    env.after(0, update)
    env.mainloop()