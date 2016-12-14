"""
Reinforcement learning maze example.

Red rectangle represents out explorer.
Black rectangles are hell where gives a -1 reward.
Yellow bin circle is the paradise where explorer can get a +1 reward.
All other states have 0 reward.

This script is the environment part of this example. The RL is in RL_brain.py.

View more on 莫烦Python: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from RL_brain import QLearning


UNIT = 40   # pixels
MAZE_H = 4
MAZE_W = 4

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

# hell
hell1_center = origin + np.array([UNIT*2, UNIT])
hell1 = canvas.create_rectangle(
    hell1_center[0]-15, hell1_center[1]-15,
    hell1_center[0]+15, hell1_center[1]+15,
    fill='black')
# hell
hell2_center = origin + np.array([UNIT, UNIT*2])
hell2 = canvas.create_rectangle(
    hell2_center[0]-15, hell2_center[1]-15,
    hell2_center[0]+15, hell2_center[1]+15,
    fill='black')

# create oval
oval_center = origin + UNIT*2
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
    base_action = np.array([0,0])
    if a == 'u':
        if s[1] > UNIT:
            base_action[1] -= UNIT
    elif a == 'd':
        if s[1] < (MAZE_H-1)*UNIT:
            base_action[1] += UNIT
    elif a == 'r':
        if s[0] < (MAZE_W-1)*UNIT:
            base_action[0] += UNIT
    elif a == 'l':
        if s[0] > UNIT:
            base_action[0] -= UNIT

    canvas.move(rect, base_action[0], base_action[1])  # move agent

    s_ = canvas.coords(rect)    # next state

    # reward function
    if s_ == canvas.coords(oval):
        reward = 1
        s_ = 'terminal'
    elif s_ in [canvas.coords(hell1), canvas.coords(hell2)]:
        reward = -1
        s_ = 'terminal'
    else:
        reward = 0
    return reward, s_


# update loop
def update():
    global rect
    S = canvas.coords(rect)
    is_terminated = False
    while not is_terminated:
        A = QLearn.choose_action(str(S))
        R, S_ = get_reward_and_next_state(S, A)  # take action & get reward and next state
        QLearn.update_table(str(S), A, R, str(S_))
        S = S_  # state will become next state
        window.update()
        time.sleep(0.1)
        if S_ == 'terminal':
            time.sleep(0.5)
            rect = reset(rect)
            is_terminated = True

    window.after(100, update)


QLearn = QLearning(actions=['u', 'd', 'l', 'r'])

window.after(100, update)
window.mainloop()