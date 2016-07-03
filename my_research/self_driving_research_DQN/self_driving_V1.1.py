# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import pyglet
import numpy as np
import pandas as pd
import learning_methods
import matplotlib.pyplot as plt
import platform


class Window(pyglet.window.Window):
    def __init__(self, width=400, height=400):
        super().__init__(width, height, resizable=True, caption='{0}-{1}'.format(car_learn.learning_method_name,
                                                                                 car_learn.alpha_method))
        #self.fps_display = pyglet.clock.ClockDisplay()
        self.set_location(x=500, y=30)
        self.icon = pyglet.image.load('car.png')
        self.set_icon(self.icon)
        self.label_batch = pyglet.graphics.Batch()
        self.squ_grad_momentum_label = pyglet.text.Label(x=500, y=690,
                                                text='squ_momentum = %s'%car_learn.squ_grad_momentum, font_size=10,
                                               batch=self.label_batch)
        self.momentum_label = pyglet.text.Label(x=500, y=670, text='momentum = %s'%car_learn.momentum, font_size=10,
                                               batch=self.label_batch)
        self.epsilon_label = pyglet.text.Label(x=500, y=650, text='epsilon = %s'%car_learn.epsilon, font_size=10,
                                               batch=self.label_batch)
        self.alpha_label = pyglet.text.Label(x=500, y=630, text='alpha = %s'%car_learn.alpha, font_size=10,
                                             batch=self.label_batch)
        self.gamma_label = pyglet.text.Label(x=500, y=610, text='gamma = %s'%car_learn.gamma, font_size=10,
                                             batch=self.label_batch)
        self.learning_label = pyglet.text.Label(x=500, y=580, text='Exploration', font_size=20,
                                                batch=self.label_batch)
        self.pause = False
        self.time = 0
        self.terminate_count = 0
        self.age = 0
        self.isLearning = True
        self.reward_series = pd.Series()
        self.accumulated_reward = 0
        self.episode = 0
        self.focus_on_training = False

    def initiate(self):
        self.car_batch = pyglet.graphics.Batch()
        self.cars = []
        self.headway_labels = []
        self.reward_labels = []
        self.v_labels = []
        self.a_labels = []
        self.region_centre = np.array([self.width/2, self.height/2])
        self.centre_radius = (self.width - 100)/2
        self.car_number = 5
        self.isTerminate = False
        self.normal_img = pyglet.image.load('car.png')
        self.crash_img = pyglet.image.load('car_crash.png')

        v_update = np.random.choice(np.arange(11, 120))
        p = 0
        for i in range(self.car_number):
            self.cars.append(CarEnvironment(img=self.normal_img, batch=self.car_batch, usage='stream',
                                 ID=i, p=p, v=v_update, a=0, l=4))
            v_update += np.random.randint(-10, 10)
            p -= 4 + v_update/3.6*(0.5*np.random.random()+0.75)

            # labels
            self.headway_labels.append(pyglet.text.Label(x=40, y=i*15, text='headway%s = --' % i, font_size=10))
            self.reward_labels.append(pyglet.text.Label(x=170, y=i*15, text='reward%s = --' % i, font_size=10))
            self.v_labels.append(pyglet.text.Label(x=300, y=i*15, text='v{0} = {1}'.format(i, int(self.cars[i].v*36/10)), font_size=10))
            self.a_labels.append(pyglet.text.Label(x=370, y=i*15, text='a{0} = {1}'.format(i, self.cars[i].a), font_size=10))

    def unit_to_screen(self, distance):
        return distance/self.pixel_unit

    def on_draw(self):
        self.clear()
        if self.focus_on_training == False:
            #self.fps_display.draw()
            self.label_batch.draw()
            self.car_batch.draw()
            for i in range(self.car_number):
                self.headway_labels[i].draw()
                self.reward_labels[i].draw()
                self.v_labels[i].draw()
                self.a_labels[i].draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ENTER:
            if self.pause == True:
                pyglet.clock.schedule(self.update)
                self.pause = False
            else:
                pyglet.clock.unschedule(self.update)
                self.pause = True
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.ESCAPE:
            self.close()
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.C:
            # plot cost history
            car_learn.cost_his.plot()
            plt.ylabel('Cost')
            plt.xlabel('Learning step')
            plt.gca().set_xlim(left=0)
            plt.grid(True)
            if platform.system() == 'Windows':
                plt.show()
            else:
                plt.show(False)
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.A:
            # plot cost history
            car_learn.max_action_value_his.plot()
            plt.ylabel('Maximum action value')
            plt.xlabel('Learning step')
            plt.gca().set_xlim(left=0)
            plt.grid(True)
            if platform.system() == 'Windows':
                plt.show()
            else:
                plt.show(False)
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.V:
            # plot value history
            pass
            """
            plt.scatter(self.reward_series.index, self.reward_series.values, c='black', s=20)
            plt.ylabel('Q Value')
            plt.xlabel('Episode')
            plt.gca().set_xlim(left=0)
            plt.grid(True)
            if platform.system() == 'Windows':
                plt.show()
            else:
                plt.show(False)"""
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.R:
            # plot accumulated reward history
            self.reward_series.plot()
            plt.ylabel('Reward')
            plt.xlabel('Episode')
            plt.gca().set_xlim(left=0)
            plt.grid(True)
            if platform.system() == 'Windows':
                plt.show()
            else:
                plt.show(False)
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.F:
            if self.focus_on_training == False:
                print('Focus on training...')
                self.focus_on_training = True
            else:
                self.focus_on_training = False

        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.D:
            if car_learn.enable_alpha_decrease == True:
                car_learn.enable_alpha_decrease = False
            else:
                car_learn.enable_alpha_decrease = True
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.P:
            global epsilon_backup, time_step
            if self.isLearning == True:
                self.isLearning = False
                time_step = 1/30
                epsilon_backup = car_learn.epsilon
                car_learn.epsilon = 0
                self.initiate()
                self.epsilon_label.text = 'epsilon = ' + str(round(car_learn.epsilon, 2))
                self.learning_label.text = 'Exploitation'
            else:
                car_learn.epsilon = epsilon_backup
                time_step = 0.1
                self.epsilon_label.text = 'epsilon = ' + str(round(car_learn.epsilon, 2))
                self.learning_label.text = 'Exploration'
                self.isLearning = True
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.S:
            config = car_learn.get_config()   # get fixed_Theta & Theta
            config.to_pickle('Config_%s_V1.1.pickle' % car_learn.learning_method_name)
            print('\n#### save all configurations ####\nMemory size= %s' % car_learn.memory.shape[0])
        elif modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.L:
            config = pd.read_pickle('Config_%s_V1.1.pickle' % car_learn.learning_method_name)
            car_learn.set_config(config)
            self.epsilon_label.text = 'epsilon = ' + str(round(car_learn.epsilon,2))
            self.alpha_label.text = 'alpha = ' + str(car_learn.alpha)
            print('\n#### load all all configurations ####\nMemory size= %s' % car_learn.memory.shape[0])
        elif symbol == pyglet.window.key.UP:
            self.cars[0].a += 1
            self.a_labels[0].text = 'a%s = '%0 + str(self.cars[0].a)
        elif symbol == pyglet.window.key.DOWN:
            self.cars[0].a -= 1
            self.a_labels[0].text = 'a%s = '%0 + str(self.cars[0].a)
        elif symbol == pyglet.window.key.Q:
            if car_learn.epsilon < 1:
                car_learn.epsilon += 0.05
                self.epsilon_label.text = 'epsilon = ' + str(round(car_learn.epsilon,2))
        elif symbol == pyglet.window.key.A:
            if car_learn.epsilon > 0:
                car_learn.epsilon -= 0.05
                self.epsilon_label.text = 'epsilon = ' + str(round(car_learn.epsilon,2))
        elif symbol == pyglet.window.key.W:
            if car_learn.alpha*1.2 <= 0.1:
                car_learn.alpha = car_learn.alpha*1.2
                self.alpha_label.text = 'alpha = ' + str(car_learn.alpha)
            else:
                car_learn.alpha = 0.1
                self.alpha_label.text = 'alpha = ' + str(car_learn.alpha)
        elif symbol == pyglet.window.key.S:
            car_learn.alpha = car_learn.alpha/1.2
            self.alpha_label.text = 'alpha = ' + str(car_learn.alpha)
        elif symbol == pyglet.window.key.E:
            if car_learn.gamma < 1:
                car_learn.gamma += 0.01
                self.gamma_label.text = 'gamma = ' + str(round(car_learn.gamma,2))
        elif symbol == pyglet.window.key.D:
            if car_learn.gamma > 0:
                car_learn.gamma -= 0.01
                self.gamma_label.text = 'gamma = ' + str(round(car_learn.gamma,2))

    def update_screen(self, i):
        self.cars[i].p += self.cars[i].v * time_step + 1/2*self.cars[i].a*time_step**2
        self.cars[i].v += self.cars[i].a * time_step
        if self.focus_on_training == False:
            self.cars[i].central_radian = self.unit_to_screen(self.cars[i].p)/self.centre_radius
            deltaX = self.centre_radius * np.cos(self.cars[i].central_radian)
            deltaY = self.centre_radius * np.sin(self.cars[i].central_radian)
            self.cars[i].x = deltaX + self.region_centre[0]
            self.cars[i].y = deltaY + self.region_centre[1]
            self.cars[i].rotation = -np.degrees([self.cars[i].central_radian + np.pi/2])
            self.v_labels[i].text = 'v%s = '%i + str(int(self.cars[i].v*3600/1000))
            self.a_labels[i].text = 'a%s ='%i + str(round(self.cars[i].a,2))
            self.cars[i].h = (self.cars[i-1].p-self.cars[i-1].l-self.cars[i].p)/self.cars[i].v

    def update_label(self, i):
        if self.cars[i].reward <= 0 and i>=1:
            color = (255,0,0,255)
            self.cars[i].image = self.crash_img
            self.cars[i].image.anchor_x = self.cars[i].image.width / 2
            self.cars[i].image.anchor_y = self.cars[i].image.height / 2
        elif i>=1:
            color = (255,255,255,255)
            self.cars[i].image = self.normal_img
        if i >= 1:
            self.headway_labels[i].color = color
            self.headway_labels[i].text = 'headway%s = '%i + str(round(self.cars[i].h, 3))
            self.reward_labels[i].text = 'reward%s = '%i + str(round(self.cars[i].reward, 3))
        self.v_labels[i].text = 'v%s = '%i + str(int(self.cars[i].v*3600/1000))
        self.a_labels[i].text = 'a%s ='%i + str(round(self.cars[i].a,2))

    def update(self, dt):
        # Act in environment
        all_cars_action = []
        all_cars_state = []
        for i in range(self.car_number):
            if i == 0:    # For me to control
                self.update_screen(i)
            else:       # There for RL cars
                state = self.get_state(self.cars[i], self.cars[i-1])
                action_label = car_learn.take_and_return_action(self, state)

                all_cars_action.append(action_label)
                all_cars_state.append(state)

        for i in range(1, self.car_number):
            # learn Q
            action_label = all_cars_action[i-1]
            state = all_cars_state[i-1]
            # already at next state now
            next_state = self.get_state(self.cars[i], self.cars[i-1], check_terminate=True)
            reward = self.get_reward(next_state)
            if self.focus_on_training == False:
                self.update_label(i)
            if self.isLearning == True:
                self.accumulated_reward += reward
                car_learn.store_transition(self, state, action_label, reward, next_state, self.isTerminate)
                self.time += 1
                self.terminate_count += 1
                if self.time % car_learn.batch_size == 0 and self.age >= car_learn.replay_start_size:
                    car_learn.update_Theta()
                    if car_learn.alpha_method in ['Adam', 'RMSProp', 'RMSProp_DQN']:
                        self.alpha_label.text = 'alpha = %s' % str(car_learn.average_learning_rate)
                    else:
                        self.alpha_label.text = 'alpha = %s' % str(car_learn.alpha)
                    self.epsilon_label.text = 'epsilon = %s ' % str(car_learn.epsilon)
                elif self.age < car_learn.replay_start_size:
                    if self.age == 0:
                        print('Randomly walking...')
                    self.age += 1

            # Terminate check
            terminate_time = 1000
            if self.isTerminate == True or self.terminate_count >= terminate_time:
                self.reward_series.set_value(self.episode, self.accumulated_reward)
                self.episode += 1
                self.terminate_count = 0
                self.accumulated_reward = 0
                self.initiate()
                self.isTerminate = False
                break

    def update_environment(self, state, action):
        """
        !!! IMPORTANT !!!
        for learning method
        ----------
        Make environment to update itself.
        """
        ID = int(state['ID'])
        # Change car's acceleration
        self.cars[ID].a = action
        self.update_screen(ID)

    def get_features(self, state, actions_array):
        """
        !!! IMPORTANT !!!
        for learning method.
        get features for particular action.
        It is to say what are the new features after taking actions.
        actions shape = array([[1,2,3,4]])
        feature shape = (n, 1)
        """
        p, v, ID = state['p'], state['v'], int(state['ID'])
        p_l1, v_l1, a_l1, l_l1 = state['p_l1'], state['v_l1'], state['a_l1'], state['l_l1']
        a = actions_array    # horizontal array([[1,2,3,4]])
        p += v * time_step + 1/2 * a * time_step**2         # an array or single value
        # Assume front will travel with last step's configurations
        p_l1 += v_l1 * time_step + 1/2 * a_l1 * time_step**2   # single value
        v += a * time_step                                  # an array or single value
        v_l1 += a_l1 * time_step                              # single value

        deltaX1 = p_l1-l_l1 - p   # deltaX is an array
        headway1 = deltaX1/v      # headway is also an array
        headway1[headway1 == np.inf] = 10
       # headway1 = 10 if np.isinf(headway1) else headway1
        v_l1 = v_l1*np.ones_like(a)

        # Careful about feature normalization [-1, 1]
        f1 = headway1-1.5
        f3 = (v_l1-15)/15

        features = np.concatenate((f1, f1**2, f1**3, f3), axis=0)
        return features

    def get_state(self, self_car, leader1, check_terminate=False):
        state = pd.Series({'p': self_car.p, 'v': self_car.v, 'a': self_car.a, 'l': self_car.l, 'ID': self_car.ID,
                 'p_l1': leader1.p, 'v_l1': leader1.v, 'a_l1': leader1.a, 'l_l1': leader1.l, 'ID_l1': leader1.ID})

        if check_terminate == True:
            p_f, l_f, p, v = leader1.p, leader1.l, self_car.p, self_car.v
            distance = (p_f-l_f) - p
            headway = distance / v
            desired_headway = 1
            if headway <= 0:
                self.isTerminate = True
            elif headway <= desired_headway*3.5:
                self.isTerminate = False
            else:
                self.isTerminate = True
        return state

    def get_reward(self, next_state):
        p, v, ID, a = next_state['p'], next_state['v'], int(next_state['ID']), next_state['a']
        p_f, v_f, l_f = next_state['p_l1'], next_state['v_l1'], next_state['l_l1']

        distance = (p_f-l_f) - p
        h = distance / v
        h = 10 if np.isinf(h) else h        # avoid reward to inf
        #desired_headway = 1

        if h < 1.3 and h >= 1:
            reward = 4*(1.3-h)
        elif h > 0.7 and h < 1:
            reward = 4*(h-0.7)
        elif h >= 1.3:
            reward = -2*(h-1.3)
        else:
            # h<=0.7
            reward = -1*(0.7-h)

        self.cars[ID].reward = reward
        return reward

    def isCollide(self, car, front_car):
        if car.p >= front_car.p-front_car.l:
            return True
        else:
            return False


class CarEnvironment(pyglet.sprite.Sprite):
    def __init__(self, img, batch, usage, ID, p, a=0, v=60, l=4, ):
        pyglet.sprite.Sprite.__init__(self, img=img, batch=batch, usage=usage)
        self.a = a
        self.v = v/3.6  # convert to m/s
        self.p = p
        self.l = l              # length
        self.ID = ID
        self.scale = 0.05
        self.image.anchor_x = self.image.width / 2
        self.image.anchor_y = self.image.height / 2
        self.length = self.image.width
        window.pixel_unit = self.l / self.width
        self.central_radian = window.unit_to_screen(self.p)/window.centre_radius
        dx = window.centre_radius * np.cos(self.central_radian)
        dy = window.centre_radius * np.sin(self.central_radian)
        self.position = window.region_centre + np.array([dx, dy])
        self.rotation = -np.degrees([self.central_radian + np.pi/2])
        self.isCollide = False
        self.reward = 0

def plot_reward_fun():
    import matplotlib.pyplot as plt
    def reward(h):
        if h<1.3 and h>=1:
            reward = 1 + 2*(1.3-h)
        elif h>0.7 and h<1:
            reward = 1 + 2*(h-0.7)
        elif h>=1.3:
            reward = -1 - (h-1.3)
        else:
            reward = -1 - (0.7-h)

        return reward
    X1 = np.linspace(0, 3, 100)
    Y1= []
    for x in X1:
        y = reward(x)
        Y1.append(y)
    plt.plot(X1, Y1, 'k-')
    plt.ylabel('Reward')
    plt.xlabel('Headway (s)')
    plt.show()

if __name__ == '__main__':
    np.random.seed(112)
    actions = np.arange(-3, 2.1, 0.2)
    #car_learn = learning_methods.DQN_SGD(all_actions=actions, epsilon=1, alpha=0.001, gamma=0.99, search_time=5000, memory_capacity=30000, batch_size=50, swap_theta_at=500, n_jobs=4)
    car_learn = learning_methods.DQNNaive(all_actions=actions, n_jobs=1, epsilon=1, epsilon_decay_rate=0.992,
                                         alpha=0.01, min_alpha=0.001, squ_grad_momentum=0.9, min_squ_grad=0.01, search_time=10000,
                                         gamma=0.99, momentum=0.9, alpha_method='RMSProp', regularization=2,
                                         n_hidden_layers=1, n_hidden_units=None, activation_function='ReLU',
                                         memory_capacity=50000, batch_size=40, rec_his_rate=.2,
                                         target_theta_update_frequency=1000, replay_start_size=2000)
    #plot_reward_fun()
    window = Window(700, 700)
    window.initiate()
    pyglet.clock.schedule_interval(window.update, 1/100)
    time_step = 0.1

    pyglet.app.run()
