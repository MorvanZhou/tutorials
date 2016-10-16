import tkinter as tk
import numpy as np
import pandas as pd


class TableLUQ(object):
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self._actions = actions
        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma

        # state-actions table. columns include [state, a1, a2, a3...]
        self._q_table = pd.DataFrame(columns=['state']+self._actions)
        self._q_table.set_index('state', inplace=True)

    def learn(self, s0, a0, r, s1):
        q_s0_a0 = self._q_table.loc[s0, a0]
        if s1 in self._q_table.index:
            q_s1_a = self._q_table.loc[s1, :]
            q_s1_a_max = q_s1_a.max()
        else:
            q_s1_a_max = 0
        self._q_table.loc[s0, a0] = q_s0_a0 + self._alpha * (
            r + self._gamma * q_s1_a_max - q_s0_a0
        )

    def choose_action(self, state):
        if np.random.random() < self._epsilon:
            action = np.random.choice(self._actions)
        else:
            if state not in self._q_table.index:
                new_state_actions = pd.Series([0]*len(self._actions), index=self._actions, name=state)
                self._q_table = self._q_table.append(new_state_actions)
                action = np.random.choice(self._actions)
            else:
                state_actions_pair = self._q_table.loc[state, :]
                shuffled_state_actions_pair = state_actions_pair.reindex(
                    np.random.permutation(
                        state_actions_pair.index))
                action = shuffled_state_actions_pair.argmax()
        return action

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def alpha(self):
        return self._alpha

    @property
    def gamma(self):
        return self._gamma


def get_state(h_loc, p_loc):
    state = str(
        (
            p_loc[0] - h_loc[0],
            p_loc[1] - h_loc[1]
         )
    )
    return state


def get_reward(h_loc, p_loc):
    distance = ((p_loc[0] - h_loc[0])**2 + (p_loc[1] - h_loc[1])**2)**(1/2)
    reward = 1/distance
    return reward


def move(h_loc, action):
    h_x, h_y = h_loc[0], h_loc[1]
    if action == 'u':
        h_y -= 1
        h_y = max([h_y, 0])
    elif action == 'd':
        h_y += 1
        h_y = min([h_y, 4])
    elif action == 'l':
        h_x -= 1
        h_x = max([h_x, 0])
    else:
        h_x += 1
        h_x = min([h_x, 4])
    x_amount = (h_x - h_loc[0])*100
    y_amount = (h_y - h_loc[1])*100
    canvas.move(hunter_icon, x_amount, y_amount)
    h_loc = [h_x, h_y]

window = tk.Tk()
window.geometry('500x500')
canvas = tk.Canvas(window, height=500, width=500)
canvas.pack()

hunter_loc = [0, 0]     # can move
prey_loc = [4, 4]       # fixed

hunter_icon = canvas.create_rectangle(
    (
        hunter_loc[0]*100,
        hunter_loc[1]*100,
        hunter_loc[0]*100+100,
        hunter_loc[1]*100+100),
    fill='black'
)
prey_icon = canvas.create_oval(
    (
        prey_loc[0]*100,
        prey_loc[1]*100,
        prey_loc[0]*100+100,
        prey_loc[1]*100+100),
    fill='red'
)

move_up = [+1, 0]
move_down = [-1, 0]
move_left = [0, -1]
move_right = [0, +1]

hunter_actions = {
    'u': move_up,
    'd': move_down,
    'l': move_left,
    'r': move_right,
}


window.mainloop()