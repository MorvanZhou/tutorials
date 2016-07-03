# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import pandas as pd
import numpy as np
from multiprocessing import Process, Queue, cpu_count


class DQNNaive:
    """
    Deep Q-Networks Recalled Neural Networks.
    ------------------------------
    1, Take action at according to epsilon-greedy policy
    2, Store transition (st, at, rt, st+1) in replay memory D
    3, Sample random mini-batch of transitions (s, a, r, s') from D
    4, Compute Q-learning targets with regard to old, fixed parameters theta
    5, Input X = (action, features),
    6, output y_predict= Q(s, a, theta),
        y = R + gamma * max_a[Q(s', a, theta)]

    Use activation function to different layers:
    Non_last layer: Rectified Linear Unit (ReLU) OR SoftPlus
    Last layer: linear activation function
    ------------------------------
    Methods used in environment:
    env.update_environment(state, action_value)
    env.get_features(state, action_value)

    """
    def __init__(self, all_actions=None, epsilon=0.9, epsilon_decay_rate=0.99,
                 alpha=0.01, gamma=0.99, search_time=3000, min_alpha=0.001,
                 momentum=0.95, squ_grad_momentum=0.999, min_squ_grad=0.001, alpha_method='RMSProp', regularization=None,
                 n_hidden_layers=1, n_hidden_units=None, activation_function='ReLU', memory_capacity=30000,
                 batch_size=50, rec_his_rate=0.2, target_theta_update_frequency=1000, n_jobs=-1,
                 replay_start_size=2000):
        """

        Parameters
        ----------
        all_actions:
            all action values, is an array, shape of (n_actions, ).
        epsilon:
            epsilon greedy policy.
        epsilon_decay_rate:
            decay rate accompany with time steps.
        alpha:
            initial learning rate.
        gamma:
            discount factor in Q-learning update.
        search_time:
            For Annealing learning rate.
        min_alpha:
            The minimum learning rate value.
        momentum:
            Gradient momentum used by Adam.
        squ_grad_momentum:
            parameter for RMSProp, squared gradient (denominator) momentum.
        min_squ_grad:
            parameter for RMSProp, constant added to the squared gradient in the denominator.
        alpha_method:
            What alpha decay method has been chosen.
        regularization:
            regularization term of Neural Networks.
        n_hidden_layers:
            Number of hidden layers.
        n_hidden_units:
            Number of hidden neurons.
        activation_function:
            The activation function used in Neural Networks.
        memory_capacity:
            Number of transitions storing in memory.
        batch_size:
            mini-batch size to update gradient for each stochastic gradient descent (SGD).
        rec_his_rate:
            The ratio of most recent transitions in sampled mini-batch.
        target_theta_update_frequency:
            The frequency (measured in the number of parameter updates) with which the target network is updated.
        n_jobs:
            Number of CPU used to calculate gradient update.
        replay_start_size:
            Start replay and learning at this size.

        Returns
        -------

        """
        try:
            actions_value = all_actions
            actions_label = [str(value) for value in actions_value]
            self.actions = pd.Series(data=actions_value, index=actions_label)
        except TypeError:
            self.actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.alpha_method = alpha_method    # 'RMSProp', or 'Annealing'
        self.squ_grad_momentum = '--' if self.alpha_method in ['Annealing', 'Momentum'] else squ_grad_momentum
        if min_squ_grad < alpha**2-0.001 and self.alpha_method != 'Annealing':
            raise ValueError('min_squ_grad need greater than alpha^2')
        self.min_squ_grad = min_squ_grad
        self.learning_time = 0
        self.gamma = gamma
        self.init_alpha = alpha
        self.min_alpha = min_alpha
        self.search_time = search_time
        self.lambda_reg = regularization
        self.enable_alpha_decrease = True
        self.learning_method_name = 'DQN_RNNs'
        self.activation_function = activation_function
        if self.activation_function not in ['ReLU', 'SoftPlus']:
            raise NameError("Activation function must in: 'ReLU', 'SoftPlus'")
        self.n_layers = n_hidden_layers + 2
        self.n_hidden_units = n_hidden_units
        # Let W be vectors with one component for each possible feature
        self.fixed_Thetas = None   # np.array([1,2,3,4,]).T
        self.Thetas = None
        self.swap_Theta_counting = 0
        self.target_theta_update_frequency = target_theta_update_frequency
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.memory = pd.DataFrame()
        self.memory_index = 0
        self.cost_his = pd.Series()
        self.max_action_value_his = pd.Series()
        self.momentum = momentum if self.alpha_method in ['Adam', 'RMSProp_DQN', 'Momentum'] else '--'
        self.rec_his_rate = rec_his_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.replay_start_size = replay_start_size

    def take_and_return_action(self, env, state):
        """
        Execute in environment, for each object
        return chosen action_label
        """
        all_q, all_features = self.get_all_q_and_all_features(env, state)
        action_label = self.choose_action(all_q)
        self.update_environment(state, action_label, env)
        return action_label

    def store_transition(self, env, state, action_label, reward, next_state, terminate):
        """
        Waite all object updated, get S' and R.
        Then store it.
        self.memory is pd.Series
        """
        # next_state['isTerminate'] = True or False
        # next_state['data'] = data
        S_A_features = self.get_single_action_features(env, state, action_label)
        next_S_As_features = self.get_all_actions_features(env, next_state)
        T = pd.Series({'S_A_features': S_A_features, 'A': action_label,
                       'R': reward, "S'_As_features": next_S_As_features, 'isTerminate': terminate})
        if self.memory.empty or self.memory.shape[0] < self.memory_capacity:
            self.memory = self.memory.append(T, ignore_index=True)
        else:
            # restrain the memory size
            self.memory.iloc[self.memory_index] = T
            if self.memory_index < self.memory_capacity-1:
                self.memory_index += 1
            else:
                self.memory_index = 0

    def process_do(self, sub_batch_T, Thetas, cost_queue=None, queue=None, max_action_value_queue=None):
        """
        sub_batch_T contains:
            S_A_features, A, R, S'_As_features, isTerminate
        """
        # all_S_A_features: shape(sample num, features num)
        all_S_A_features = np.array([A for A in sub_batch_T['S_A_features'].values]).squeeze(2).T
        all_Q = self.get_Q(all_S_A_features, Thetas)
        # all_y_predict: shape(sample num, 1)
        all_y_predict = all_Q

        # all_next_S_As_features: shape(sample num, (features num, all_actions) )
        all_next_S_As_features = sub_batch_T["S'_As_features"].values
        all_next_Q_max = self.get_next_Q_max(all_next_S_As_features)
        # all_isTerminate: shape(sample num, 1)
        all_isTerminate = sub_batch_T['isTerminate'][:, np.newaxis]
        # next_Q_max = 0 if it's terminate state
        np.place(all_next_Q_max, all_isTerminate, 0)
        all_reward = sub_batch_T['R'][:, np.newaxis]
        # all_y: shape(sample num, 1)
        all_y = all_reward + self.gamma * all_next_Q_max

        Gradients = self.get_gradients_back_propagate(all_y, all_y_predict, Thetas)
        thetas_sum = 0
        for thetas in self.Thetas:
            thetas_sum += np.square(thetas[1:, :]).sum(0).sum()
        cost = 1 / (2 * len(sub_batch_T)) * \
            (np.square(all_y-all_y_predict).sum(0).sum() + self.lambda_reg * thetas_sum)
        max_action_value = np.max(all_next_Q_max)
        print('Max action value: ', max_action_value)
        if queue == None and cost_queue == None:
            return [Gradients, cost, max_action_value]
        else:
            queue.put(Gradients)
            cost_queue.put(cost)
            max_action_value_queue.put(max_action_value)

    def get_deltaTheta(self, Gradients):
        """
        To calculate deltaTheta part for gradient descent
        Parameters
        ----------
        Gradients:
            Gradients from back propagation
        method_name:
            The method used to speed up gradient descent

        Returns
        -------
        deltaTheta
        """
        # alpha
        if self.alpha_method == 'Annealing':
            # Annealing alpha
            if self.alpha > self.min_alpha and self.enable_alpha_decrease == True:
                self.alpha = self.init_alpha/(1+self.learning_time/self.search_time)
            deltaTheta = self.alpha*Gradients
        elif self.alpha_method == 'Momentum':
            try:
                self.last_deltaTheta
            except AttributeError:
                print('Initialise last deltaTheta')
                self.last_deltaTheta = np.zeros_like(Gradients)
            if self.alpha > self.min_alpha and self.enable_alpha_decrease == True:
                self.alpha = self.init_alpha/(1+self.learning_time/self.search_time)
            deltaTheta = self.momentum * self.last_deltaTheta + self.alpha * Gradients
            self.last_deltaTheta = deltaTheta
        elif self.alpha_method == 'RMSProp':
            # RMSProp
            try:
                self.g
            except AttributeError:
                print('Initialise cached gradients')
                self.g = np.zeros_like(Gradients)
            self.g = self.squ_grad_momentum * self.g + (1-self.squ_grad_momentum) * Gradients**2
            learning_rate = self.alpha / (np.sqrt(self.g + self.min_squ_grad))
            deltaTheta = learning_rate*Gradients
            self.average_learning_rate = np.mean(learning_rate)
        elif self.alpha_method == 'RMSProp_DQN':
            # RMSProp used in DQN
            try:
                self.g, self.h
            except AttributeError:
                print('Initialise cached gradients')
                self.g = np.zeros_like(Gradients)
                self.h = np.zeros_like(Gradients)
            self.g = self.squ_grad_momentum * self.g + (1-self.squ_grad_momentum) * Gradients**2
            self.h = self.momentum * self.h + (1-self.momentum) * Gradients
            value = self.g - self.h**2
            np.place(value, value<0, 0)
            learning_rate = self.alpha/(np.sqrt(value + self.min_squ_grad))
            deltaTheta = learning_rate * Gradients
            self.average_learning_rate = np.mean(learning_rate)
        elif self.alpha_method == 'Adam':
            # Adam
            try:
                self.v, self.m
            except AttributeError:
                print('Initialise m, v')
                self.v = np.zeros_like(Gradients)
                self.m = np.zeros_like(Gradients)
            self.m = self.momentum * self.m + (1-self.momentum) * Gradients
            self.v = self.squ_grad_momentum * self.v + (1-self.squ_grad_momentum) * Gradients**2
            learning_rate = self.alpha / (np.sqrt(self.v + self.min_squ_grad))
            deltaTheta = learning_rate * self.m
            self.average_learning_rate = np.mean(learning_rate)
        else:
            raise NameError('No alpha method name %s' % self.alpha_method)

        return deltaTheta

    def update_Theta(self):
        print('\nLearning step: {0} || Memory size: {1}'.format(
                self.learning_time, self.memory.shape[0]))
        self.learning_time += self.batch_size
        batch_T = self.sample_mini_batch(self.memory)

        if self.n_jobs == -1:
            core_num = cpu_count()
        elif self.n_jobs <= 0 or self.n_jobs>cpu_count():
            raise AttributeError('n_job wrong.')
        else:
            core_num = self.n_jobs
        Thetas = self.Thetas

        if core_num > 1:
            # core_num > 1
            batch_T_split = np.array_split(batch_T, core_num)
            queue = Queue()
            cost_queue = Queue()
            max_action_value_queue = Queue()
            processes = []
            for core in range(core_num):
                P = Process(target=self.process_do, args=(batch_T_split[core], Thetas, cost_queue, queue, max_action_value_queue))
                processes.append(P)
                P.start()
            for i in range(core_num):
                processes[i].join()

            Gradients = queue.get()
            cost = cost_queue.get()
            max_action_value = [max_action_value_queue.get()]
            for i in range(core_num-1):
                Gradients = np.vstack((Gradients, queue.get()))
                cost += cost_queue.get()
                max_action_value.append(max_action_value_queue.get())
            cost = cost/core_num
            max_action_value = max(max_action_value)
            Gradients = Gradients.sum(axis=0)
        else:
            # core_num = 1
            Gradients, cost, max_action_value = self.process_do(batch_T, Thetas)

        # record cost history
        if self.learning_time % self.batch_size == 0:
            self.cost_his.set_value(self.learning_time, cost)
            self.max_action_value_his.set_value(self.learning_time, max_action_value)

        # epsilon
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon * self.epsilon_decay_rate
        else:
            self.epsilon = 0.1

        deltaTheta = self.get_deltaTheta(Gradients)

        # Gradient update
        Thetas_layers = self.n_layers-1
        Thetas_shapes = [self.Thetas[i].shape for i in range(Thetas_layers)]
        for i, shape in enumerate(Thetas_shapes):
            Thetas_backup = Thetas[i] + deltaTheta[:shape[0]*shape[1]].reshape(shape)
            deltaTheta = deltaTheta[shape[0]*shape[1]:].copy()
            if np.abs(Thetas_backup).max() > 20:
                print('\n\n\n!!!!! Warning, Thetas overshooting, turn alpha down!!!\n\n\n')
            else:
                self.Thetas[i] = Thetas_backup

        # Change fixed Theta
        if self.swap_Theta_counting >= self.target_theta_update_frequency:
            self.swap_Theta_counting += self.batch_size - self.target_theta_update_frequency
            self.fixed_Thetas = self.Thetas.copy()
            print("\n\n\n## Swap Thetas ##\n\n")
        else:
            self.swap_Theta_counting += self.batch_size

    def forward_propagate(self, features, Thetas, for_bp=False):
        """
        Input:
        if for_bp = False:
            features: shape(features num, actions num)
        else:
            features: shape(features num, samples num)
            for_bp: calculate for back propagation
        --------------------------
        return:
            A: shape(1, action num)
        """
        A = features.copy()
        if for_bp:
            self.As_4_bp = []
            self.Zs = [None]
        for i in range(1, self.n_layers):
            A = np.vstack((np.ones((1, A.shape[1])), A))         # [1, a1, a2, a3].T
            if for_bp:
                self.As_4_bp.append(A)           # layer1 to n-1
            # layer i + 1
            Z = Thetas[i-1].dot(A)
            A = self.calculate_AF(Z, i)          # [a1, a2, a3].T
            if for_bp:
                self.Zs.append(Z)
        return A

    def calculate_AF(self, A, layer):
        X = A.copy()
        if layer < self.n_layers-1:
            # nonlinear activation function
            if self.activation_function == 'ReLU':
                y = X.clip(0)
            elif self.activation_function == 'SoftPlus':
                y = np.log(1+np.exp(X))
            else:
                raise NameError(self.activation_function, ' is not in the name list')
        else:
            # linear activation function
            y = X
        return y

    def get_gradients_back_propagate(self, all_y, all_A, Thetas):
        last_layer = self.n_layers-1
        gradients = np.array([])
        for i in range(last_layer, 0, -1):
            # ignore +1 in every A's beginning
            if i == last_layer:
                all_error = (all_y - all_A).T       # all_error: shape(n_actions, n_samples)
            else:
                all_error = (Thetas[i].T.dot(all_delta))[1:]
            all_delta = all_error * self.calculate_AFD(self.Zs[i].copy(), layer=i)
            if self.lambda_reg is not None:
                # regularization term:
                regularization = self.lambda_reg*Thetas[i-1]
                regularization[:, 0] = 0
                # errors: shape(n_hidden_units, last_n_hidden_units)
                gradients_for_current_layer = np.dot(all_delta, self.As_4_bp[i-1].T) + regularization
            else:
                # errors: shape(n_hidden_units, last_n_hidden_units)
                gradients_for_current_layer = np.dot(all_delta, self.As_4_bp[i-1].T)

            gradients = np.append(gradients_for_current_layer, gradients)
        Gradients = gradients/self.batch_size
        return Gradients

    def calculate_AFD(self, A, layer):
        X = A.copy()
        if layer < self.n_layers-1:
            if self.activation_function == 'ReLU':
                np.place(X, np.array(X > 0), 1)
                d = X.clip(0)
            elif self.activation_function == 'SoftPlus':
                d = 1/(1 + np.exp(-X))
            else:
                raise NameError(self.activation_function, ' is not in the name list')
        else:
            d = np.ones_like(X)
        return d

    def sample_mini_batch(self, memory):
        """
        Batch consist recent and old transitions.
        The default is 10% recent and 90% old transitions.

        Parameters
        ----------
        memory: all transitions that in the memory.

        Returns
        -------
        sampled batch
        """
        if memory.shape[0] < self.batch_size:
            batch_size = memory.shape[0]
        else:
            batch_size = self.batch_size
        rec_his_size = int(batch_size*self.rec_his_rate)
        old_his_size = batch_size - rec_his_size
        rec_index = np.random.choice(memory.index[-batch_size:], rec_his_size, replace=False)
        old_index = np.random.choice(memory.index[:-batch_size], old_his_size, replace=False)
        index = np.concatenate((rec_index, old_index))
        np.random.shuffle(index)
        batch = memory.ix[index, :]       # this is an array
        return batch

    def update_environment(self, state, action_label, env):
        action = self.actions[action_label]
        env.update_environment(state, action)

    def get_single_action_features(self, env, state, action_label):
        """
        return:
            features: np.array, shape(feature num, 1)
        """
        action = self.actions[action_label]
        features = env.get_features(state, actions_array=np.array([[action]]))    # np.array([[1,2,3,4]]).T   shape=(n,1)
        return features

    def get_all_actions_features(self, env, next_state):
        """
        return:
            all_features: pd.DataFrame, shape(feature num, action num)
        """
        # np.array([[1,2,3,4]]).T   shape=(n,1)
        F_As = env.get_features(next_state, actions_array=self.actions.values[np.newaxis, :])
        all_features = pd.DataFrame(F_As, columns=self.actions.index)
        return all_features

    def get_Q(self, features, Thetas):
        """

        Parameters
        ----------
        features
            shape(feature num, sample num)

        Returns
        -------
        all_Q:
            shape(sample num, 1)
        """
        all_Q = self.forward_propagate(features, Thetas, for_bp=True).T
        return all_Q

    def get_next_Q_max(self, all_next_S_As_features):
        """

        Parameters
        ----------
        all_next_S_As_features:
            shape(sample num, (features num, all_actions) )

        Returns
        -------
        all_next_Q_max:
            shape(sample num, 1)

        """
        all_next_Q_max = np.empty((0,1))
        for next_S_As_features in all_next_S_As_features:
            # next_S_As_features: shape(features num, all_actions)
            Q_max = self.forward_propagate(next_S_As_features, self.fixed_Thetas, for_bp=False).max()
            all_next_Q_max = np.append(all_next_Q_max, Q_max)
        all_next_Q_max = all_next_Q_max[:, np.newaxis]
        return all_next_Q_max

    def get_all_q_and_all_features(self, env, state):
        """
        Given state and env
        for a in all_action(S):     # current state S
            X_A[a] = [x0, x1, x2...].T
            q_cap[a] = F_A[a].T * W
        """
        # F = np matrix = axis0: features value, axis1: actions
        F = env.get_features(state, actions_array=self.actions.values[np.newaxis, :])
        # all_features = pd.DataFrame, axis0: feature value, axis1: actions
        all_features = pd.DataFrame(F, columns=self.actions.index)
        try:
            # all_q: pd.Series q for all actions
            all_q = pd.Series(self.forward_propagate(all_features, self.Thetas).ravel(), index=self.actions.index)
        except TypeError:
            # create fixed Theta and Theta, random initial theta, shape(n_hidden_units, features)
            if self.n_hidden_units == None:
                self.n_hidden_units = 2*(F.shape[0]+1)
            self.fixed_Thetas = []
            # nonlinear activation function
            self.fixed_Thetas.append(np.random.random((self.n_hidden_units, F.shape[0]+1)) * (2*0.1) - 0.1)
            if self.n_layers > 3:
                # nonlinear activation function
                for i in range(self.n_layers-3):
                    self.fixed_Thetas.append(
                            np.random.random((self.n_hidden_units, self.n_hidden_units+1)) * (2*0.1) - 0.1)
            # linear activation function
            self.fixed_Thetas.append(np.random.random((1, self.n_hidden_units+1)) * (2*0.1) - 0.1)
            self.Thetas = self.fixed_Thetas.copy()
            all_q = pd.Series(self.forward_propagate(all_features, self.Thetas).ravel(), index=self.actions.index)
        return [all_q, all_features]

    def choose_action(self, all_q):
        """
        Choose action A = argmax_a(q_cap) with probability 1-epsilon, else a random action.
        """
        if np.random.random() <= 1 - self.epsilon:
            # choose optimal action
            all_q = all_q.reindex(np.random.permutation(all_q.index))
            action_label = all_q.argmax(skipna=False)
        else:
            # random choose action
            action_label = np.random.choice(self.actions.index)

        return action_label

    def get_optimal_action(self, all_features):
        """
        This is for applying this trained model
        Parameters
        ----------
        all_features: all features, not include 1 (bias term)

        Returns
        -------
        The optimal action
        """
        all_Features = pd.DataFrame(all_features, columns=self.actions.index)
        # all_q: pd.Series q for all actions
        all_q = pd.Series(self.forward_propagate(all_Features, self.Thetas).ravel(), index=self.actions.index)
        all_q = all_q.reindex(np.random.permutation(all_q.index))
        action_label = all_q.argmax(skipna=False)
        action = self.actions[action_label]
        return action

    def get_config(self):
        all_configs = pd.Series({'Thetas': self.Thetas, 'fixed_Thetas': self.fixed_Thetas, 'learning_time': self.learning_time,
                                 'alpha': self.alpha, 'memory': self.memory, 'epsilon': self.epsilon, 'actions': self.actions,
                                 })
        return all_configs

    def set_config(self, config):
        self.Thetas, self.fixed_Thetas, self.learning_time, self.alpha, self.memory, self.epsilon, self.actions, \
         = config['Thetas'], config['fixed_Thetas'], config['learning_time'], config['alpha'], \
                     config['memory'], config['epsilon'], config['actions']