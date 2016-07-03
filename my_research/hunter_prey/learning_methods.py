# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import pandas as pd
import numpy as np


class Qlearn_tabel:
    def __init__(self, all_actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        """
        Parameters
        ----------
        actions: Available actions
        epsilon: exploration rate
        alpha: step size
        gamma: reduction rate

        """
        # a list of actions eg. ['W', 'E', 'N', 'S']
        self.actions = all_actions
        # q_table: DataFrame: (index: stateID, column: actions)
        self.q_table = pd.DataFrame(columns=['stateID']+self.actions)
        self.q_table.set_index('stateID', inplace=True)
        # state_index: stateID index for q_table: (index: (x, y), Series: stateID)
        self.state_index = pd.Series(name='stateID')
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.learning_method_name = 'Q_learning'

    def run_for_each_episode(self, initial_state, env):
        """
        Initialize S

        Repeat (for each step of episode):
            Choose A from S using policy derived from Q (e.g., epsilon-greedy)
            Take action A, observe R, S'
            Q(S, A) = Q(S, A) + alpha[R + gamma * max_a[Q(S', a)] - Q(S, A)]
            S = S'

        until S is terminal
        """

        self.state = initial_state
        terminated = False
        while not terminated:       # for each step of episode
            self.action = self.choose_action(self.state)
            self.reward, self.next_state = self.take_action(self.action, env)

            self.learnQ(state1=self.state, action1=self.action, reward=self.reward, state2=self.next_state)

            self.state = self.next_state

            if env.terminate == True:
                terminated = True

    def take_action(self, action, env):
        """
        Connect to environment
        """
        # env.get_feedback belows to environment module
        reward, next_state = env.get_feedback(self.state, action)
        return [reward, next_state]

    def checkQ(self, state, action, pass_value):
        """
        Check existence of any q value
        Parameters
        ----------
        state
        action
        pass_value = 0 for next state, = None for current state
        """
        passed_q_value = pass_value
        try:
            # State exist
            stateID = self.state_index[state]
            # q value exist
            q_value = self.q_table.loc[stateID][action]
            if pd.isnull(q_value):
                # q value not exist
                stateID = self.state_index[state]
                self.q_table.loc[(stateID, action)] = passed_q_value
        except Exception as e:
            # State Not exist
            # Add state to state_index
            self.state_index = self.state_index.append(pd.Series({state: self.state_index.count()}))
            # Add action value to q_table
            row = {action: [passed_q_value]}
            new_row = pd.DataFrame(row)
            self.q_table = self.q_table.append(new_row, ignore_index=True)

    def getQ(self, state, action):
        """
        Returns
        -------
        get q value from q_table by substituting (state, action)
        """
        stateID = self.state_index[state]
        q_value = self.q_table.loc[stateID][action]
        return q_value

    def signQ(self, state, action, value):
        """
        Sign new Q value to q_table.
        Parameters
        ----------
        state
        action
        value replaced Q value
        """
        stateID = self.state_index[state]
        self.q_table.loc[(stateID, action)] = value

    def learnQ(self, state1, action1, reward, state2):
        # Check existence
        self.checkQ(state1, action1, pass_value=None)
        [self.checkQ(state2, a, pass_value=0) for a in self.actions]
        # Calculation phase
        old_v = self.getQ(state1, action1)
        max_q_new = max([self.getQ(state2, a) for a in self.actions])
        new_v = reward + self.gamma * max_q_new
        if pd.isnull(old_v):
            self.replaced_old_q = reward
        else:
            q_error = new_v - old_v
            self.replaced_old_q = old_v + self.alpha * q_error
        self.signQ(state1, action1, value=self.replaced_old_q)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            # check if state exist.
            [self.checkQ(state, a, pass_value=0) for a in self.actions]
            q_for_all_actions = [self.getQ(state, a) for a in self.actions]
            max_Q = max(q_for_all_actions)
            count = q_for_all_actions.count(max_Q)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q_for_all_actions[i] == max_Q]
                i = np.random.choice(best)
            else:
                i = q_for_all_actions.index(max_Q)

            action = self.actions[i]
        return action