# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import environment
import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import sys
import os
import learning_methods
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import platform
from matplotlib import pyplot as plt

class HunterWorldEnvironment(environment.Environment):

    def get_feedback(self, state, action):
        """
        The feedback when object that an action at one state.
        Parameters
        ----------
        state: The current state of the object
        action: The action been taken by object

        Returns
        -------
        [ reward, next_state]
        """
        self.get_next_state(state, action)
        self.get_reward(self.next_state)
        self.update()     # environment update
        return [self.reward, self.next_state]

    def get_reward(self, next_state):
        if next_state == (0, 0):
            self.terminate = True
            reward = 10 # np.average(list(main_window.grids.values()))*100
        else:
            self.terminate = False
            reward = -1 #-np.average(list(main_window.grids.values()))/500
        self.reward = reward

    def get_features(self):
        x_prey, y_prey = main_window.prey_init_coord
        x_hunter, y_hunter = self.get_current_coord()
        f1, f2 = (x_prey - x_hunter), (y_prey - y_hunter)
        features = np.array([f1, f2]).T
        return features

    def get_next_state(self, state, action):
        x_prey, y_prey = main_window.prey_init_coord
        self.get_current_coord()
        x_hunter, y_hunter = self.get_next_coord(self.current_coord, action)
        self.next_state = (x_prey - x_hunter, y_prey - y_hunter)

    def get_current_coord(self):
        x, y = main_window.canvas.bbox(main_window.hunter_body)[:2]
        x, y = x+1, y+1
        self.current_coord = (x, y)
        return self.current_coord

    def get_next_coord(self, current_coord, action):
        # set hunter's next second coord to a temp value
        unit = main_window.unit
        left_bound = 0
        right_bound = (main_window.grids['x'] - 1) * unit
        up_bound = 0
        down_bound = (main_window.grids['y'] - 1) * unit
        # hunter at left most grid
        if current_coord[0] == left_bound and action == 'left':
            self.next_coord = current_coord
        # hunter at right most grid
        elif current_coord[0] == right_bound and action == 'right':
            self.next_coord = current_coord
        # hunter at top most grid
        elif current_coord[1] == up_bound and action == 'up':
            self.next_coord = current_coord
        # hunter at bottom most grid
        elif current_coord[1] == down_bound and action == 'down':
            self.next_coord = current_coord
        # other situation
        else:
            if action == 'up':
                x = current_coord[0]
                y = current_coord[1] - unit
            elif action == 'down':
                x = current_coord[0]
                y = current_coord[1] + unit
            elif action == 'left':
                x = current_coord[0] - unit
                y = current_coord[1]
            else:
                x = current_coord[0] + unit
                y = current_coord[1]
            self.next_coord = (x, y)
        global coord_set
        if self.next_coord not in coord_set:
            raise ('next_coord ', self.next_coord, ' is not in coord_set')
        return self.next_coord

    def update(self):
        """
        move object to next location
        """
        xAmount = self.next_coord[0] - self.current_coord[0]
        yAmount = self.next_coord[1] - self.current_coord[1]
        main_window.canvas.move(main_window.hunter_body, xAmount, yAmount)
        main_window.canvas.update()
        self.current_coord = self.next_coord
        time.sleep(sleep_time)

    def reset(self):

        if fix_prey == True:
            main_window.canvas.delete(main_window.hunter_body)
            main_window.create_hunter()
        else:
            main_window.canvas.delete(main_window.prey_body)
            main_window.create_prey()
        main_window.canvas.update()
        self.current_coord = main_window.hunter_init_coord

    def showQ(self, q):
        main_window.label_q_v.set('Q value: %s' % round(q, 2))


class Window:
    def __init__(self, n_by_m, size):
        self.geo_size = size
        self.grids = n_by_m

    def adjust_epsilon(self, epsilon):
        hunter.epsilon = float(epsilon)

    def adjust_alpha(self, alpha):
        hunter.alpha = float(alpha)

    def adjust_gamma(self, gamma):
        hunter.gamma = float(gamma)

    def button_next_chunk_clicked(self, event=None):
        global hunter, hunter_body, prey, all_time_steps_ratio, episode
        chunk_size = self.entry_chunk_value.get()
        chunk = 0
        while chunk < chunk_size:
            hunter_body.initial_state = (self.prey_init_coord[0] - self.hunter_init_coord[0],
                                         self.prey_init_coord[1] - self.hunter_init_coord[1])
            hunter.run_for_each_episode(initial_state=hunter_body.initial_state, env=hunter_body)
            hunter_body.reset()

            episode += 1
            self.label_episode_v.set('Episode: %s' % episode)
            chunk += 1

    def button_learning_graph_clicked(self):
        global all_time_steps_ratio, episode

        y = np.array(all_time_steps_ratio)
        x = np.arange(episode - len(all_time_steps_ratio) + 1, episode + 1)
        plt.scatter(x, y,)
        plt.xlabel('Episode')
        plt.ylabel('Time to catch prey')
        plt.show()

    def button_save_learning_result_clicked(self):
        global episode
        if os.path.exists('q_table_trained_{}_episode.pickle'.format(episode)) \
                and os.path.exists('q_index_trained_{}_episode.pickle'.format(episode)):
            answer = messagebox.askyesno(message="Hunter's experience has already saved,\nSave again?")
            if answer == True:
                hunter.q_table.to_pickle('q_table_trained_{}_episode.pickle'.format(episode))
                hunter.q_index.to_pickle('q_index_trained_{}_episode.pickle'.format(episode))
                messagebox.showinfo(message="Hunter's experience has been saved")
        else:
            hunter.q_table.to_pickle('q_table_trained_{}_episode.pickle'.format(episode))
            hunter.q_index.to_pickle('q_index_trained_{}_episode.pickle'.format(episode))
            messagebox.showinfo(message="Hunter's experience has been saved")

    def button_load_learning_result_clicked(self):
        global episode
        initialdir = os.path.dirname(os.path.realpath(__file__))
        while True:
            if platform.system() == 'Darwin':
                q_index_path = askopenfilename(initialdir=initialdir, message='Choose q_index',
                                       filetypes=[('pickle files', '*.pickle'),
                                                  ("All files", "*.*") ])
            else:
                q_index_path = askopenfilename(initialdir=initialdir, title='Choose q_index',
                                       filetypes=[('pickle files', '*.pickle'),
                                                  ("All files", "*.*") ])
            if 'q_index' not in q_index_path:
                messagebox.showwarning(message='Wrong file.')
            else:
                break
        while True:
            if platform.system() == 'Darwin':
                q_table_path = askopenfilename(initialdir=initialdir, message='Choose q_table',
                                       filetypes=[('pickle files', '*.pickle'),
                                                  ("All files", "*.*") ])
            else:
                q_table_path = askopenfilename(initialdir=initialdir, title='Choose q_table',
                                       filetypes=[('pickle files', '*.pickle'),
                                                  ("All files", "*.*") ])
            if 'q_table' not in q_table_path:
                messagebox.showwarning(message='Wrong file.')
            else:
                break

        try:
            hunter.q_table = pd.read_pickle(q_table_path)
            hunter.q_index = pd.read_pickle(q_index_path)
            episode = [int(d) for d in q_table_path.split('_') if d.isdigit()][0]
            self.label_episode_v.set('Episode: %s' % episode)
        except Exception as e:
            messagebox.showerror(message=e)

    def create_hunter(self):
        self.hunter_init_coord = (0, 0)   # location (n, m)
        self.hunter_diameter = self.unit
        hunter_body_coord = (self.hunter_init_coord[0] * self.unit, self.hunter_init_coord[1] * self.unit,
                             self.hunter_init_coord[0] + self.hunter_diameter, self.hunter_init_coord[1] + self.hunter_diameter)
        self.hunter_body = self.canvas.create_rectangle(hunter_body_coord, fill='black')

    def create_prey(self):
        while True:
            if fix_prey == True:
                self.prey_init_coord = (7*self.unit, 5*self.unit)  # location (n, m)
            else:
                n, m = self.grids['x'], self.grids['y']
                n = random.choice(range(n))
                m = random.choice(range(m))
                self.prey_init_coord = (n*self.unit, m*self.unit)  # location (n, m)
            if self.prey_init_coord != self.hunter_init_coord:
                break
        self.prey_diameter = self.unit
        prey_body_coord = (self.prey_init_coord[0], self.prey_init_coord[1],
                           self.prey_init_coord[0] + self.prey_diameter, self.prey_init_coord[1] + self.prey_diameter)
        self.prey_body = self.canvas.create_oval(prey_body_coord, fill='red')

    def make_window(self):
        global coord_set
        self.window = Tk()
        self.window.geometry(self.geo_size)

        # keyboard event
        self.window.bind('<Return>', self.button_next_chunk_clicked, add=True)

        # canvas
        self.canvas = Canvas(self.window, height=600, width=600, relief='sunken', bg='white')
        n, m = self.grids['x'], self.grids['y']
        self.unit = 600 / max(n, m)
        unit = self.unit
        coord_set = []
        for i in range(n):
            for j in range(m):
                x1, y1 = i*unit, j*unit
                x2, y2 = x1+unit, y1+unit
                coord_set.append((x1, y1))
                coord = (x1, y1, x2, y2)
                self.canvas.create_rectangle(coord)

        self.create_hunter()

        self.create_prey()

        self.canvas.pack()

        # button
        self.button_next_chunk = Button(self.window, command=self.button_next_chunk_clicked, text='Next Chunk',
                                        ).place(x=450, y=640)
        Button(self.window, command=self.button_learning_graph_clicked, text='Learning Graph')\
            .place(x=450, y=670)
        Button(self.window, command=self.button_load_learning_result_clicked, text='load Learning Result')\
            .place(x=450, y=700)
        Button(self.window, command=self.button_save_learning_result_clicked, text='Save Learning Result')\
            .place(x=450, y=730)

        # scales
        epsilon_v = DoubleVar()
        self.scale_epsilon = Scale(self.window, command=self.adjust_epsilon, digits=3,
                                   from_=0, to=1, label='Epsilon:', resolution=0.01, showvalue=True,
                                   sliderlength=30, variable=epsilon_v, orient=HORIZONTAL).pack(side=LEFT)
        epsilon_v.set(0.1)
        alpha_v = DoubleVar()
        self.scale_alpha = Scale(self.window, command=self.adjust_alpha, digits=3,
                                   from_=0, to=1, label='Alpha:', resolution=0.01, showvalue=True,
                                   sliderlength=30, variable=alpha_v, orient=HORIZONTAL).pack(side=LEFT)
        alpha_v.set(0.1)
        gamma_v = DoubleVar()
        self.scale_gamma = Scale(self.window, command=self.adjust_gamma, digits=3,
                                   from_=0, to=1, label='Gamma:', resolution=0.01, showvalue=True,
                                   sliderlength=30, variable=gamma_v, orient=HORIZONTAL).pack(side=LEFT)
        gamma_v.set(0.9)

        # entry
        self.entry_chunk_value = IntVar()
        self.label_chunk = Label(self.window, text='Chunk size:').pack(side=LEFT)
        self.entry_chunk = Entry(self.window, textvariable=self.entry_chunk_value, width=4).pack(side=LEFT)
        self.entry_chunk_value.set(1)

        # episode label
        global episode
        self.label_episode_v = StringVar()
        Label(self.window, fg='red', textvariable=self.label_episode_v, font=("Helvetica", 20)).place(x=200, y=620)
        self.label_episode_v.set('Episode: 0')

        # Q value label
        self.label_q_v = StringVar()
        Label(self.window, fg='red', textvariable=self.label_q_v, font=("Helvetica", 10)).place(x=10, y=620)
        self.label_q_v.set('Q value: 0')

    def run_window(self):
        self.window.mainloop()


def make_grid(n_by_m):
    # Build window
    if len(n_by_m) != 2:
        raise ValueError('Wrong grid geometry')
    main_window = Window(n_by_m, size='600x800')
    main_window.make_window()
    return main_window


def run(main_window):
    global hunter, hunter_body, prey, episode, all_time_steps_ratio
    episode = 0
    all_time_steps_ratio = []
    # second coord system: (n, m)
    hunter_body = HunterWorldEnvironment()
    hunter = learning_methods.Qlearn_tabel(all_actions=['up', 'down', 'left', 'right'])
    main_window.run_window()




#####################
# main program
#####################

if __name__ == '__main__':
    args = sys.argv[1:]
    print('system agrs', args)
    n_by_m_grid = {'x': 6, 'y': 6}
    fix_prey = False
    focus_on_training = False
    sleep_time = 0
    main_window = make_grid(n_by_m_grid)
    run(main_window)
