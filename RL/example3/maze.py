"""
Reinforcement learning maze example.

Red rectangle represents out explorer.
Black rectangles are wall where gives a -1 reward.
Yellow bin circle is the paradise where explorer can get a +1 reward.
All other states have 0 reward.

This script is the environment part of this example. The RL is in RL_brain.py.

View more on 莫烦Python: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from RL_DQN import QLearning


UNIT = 40   # pixels
MAZE_H = 10
MAZE_W = 10

window = tk.Tk()
window.title('maze')
window.geometry('{0}x{1}'.format(MAZE_H*UNIT, MAZE_H*UNIT))

canvas = tk.Canvas(window, bg='white',
                   height=MAZE_H*UNIT,
                   width=MAZE_W*UNIT)

# create grids
for c in range(0, MAZE_W*UNIT, UNIT):
    x0, y0, x1, y1 = c, 0, c, MAZE_H*UNIT
    canvas.create_line(x0, y0, x1, y1)
for r in range(0, MAZE_H*UNIT, UNIT):
    x0, y0, x1, y1 = 0, r, MAZE_H*UNIT, r
    canvas.create_line(x0, y0, x1, y1)

# create origin
origin = np.array([20, 20])

# wall
walls = []
for i in [0,1,2,3,5,6,7,8,9]:
    wall_center = origin + np.array([UNIT*i, UNIT*5])
    wall = canvas.create_rectangle(
        wall_center[0]-15, wall_center[1]-15,
        wall_center[0]+15, wall_center[1]+15,
        fill='black')
    walls.append(wall)


# create oval
oval_center = origin + np.array([UNIT*9, UNIT*9])
oval = canvas.create_oval(
    oval_center[0] - 15, oval_center[1] - 15,
    oval_center[0] + 15, oval_center[1] + 15,
    fill='yellow')

# create red rect
rect = canvas.create_rectangle(
    origin[0]-15, origin[1]-15,
    origin[0]+15, origin[1]+15,
    fill='red')

# pack all
canvas.pack()


def reset(rect):
    canvas.delete(rect)
    origin = np.array([20, 20])
    rect = canvas.create_rectangle(
        origin[0] - 15, origin[1] - 15,
        origin[0] + 15, origin[1] + 15,
        fill='red')
    return rect


def get_reward_and_next_state(s, a):
    global rect
    base_action = np.array([0, 0, 0, 0])
    if a == 'u':
        if s[1] > UNIT:
            base_action[1] -= UNIT
            base_action[3] -= UNIT
    elif a == 'd':
        if s[1] < (MAZE_H-1)*UNIT:
            base_action[1] += UNIT
            base_action[3] += UNIT
    elif a == 'r':
        if s[0] < (MAZE_W-1)*UNIT:
            base_action[0] += UNIT
            base_action[2] += UNIT
    elif a == 'l':
        if s[0] > UNIT:
            base_action[0] -= UNIT
            base_action[2] -= UNIT

    # hit the wall
    if (np.array(s) + base_action).tolist() in [canvas.coords(wall) for wall in walls]:
        base_action *= 0

    canvas.move(rect, base_action[0], base_action[1])  # move agent

    s_ = canvas.coords(rect)    # next state

    # reward function
    if s_ == canvas.coords(oval):
        reward = 1
        s_ = 'terminal'
    else:
        reward = 0
    return s_, reward


# update loop
def update():
    global rect
    S = canvas.coords(rect)
    is_terminated = False
    while not is_terminated:
        A = QLearn.choose_action(str(S))
        R, S_ = get_reward_and_next_state(S, A)  # take action & get next state and reward
        QLearn.update_table(str(S), A, R, str(S_))
        S = S_
        window.update()
        time.sleep(0.01)
        if S_ == 'terminal':
            time.sleep(0.1)
            rect = reset(rect)
            is_terminated = True

    window.after(100, update)


QLearn = QLearning(actions=['u', 'd', 'l', 'r'])

window.after(100, update)
window.mainloop()