from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import sparse
import logging
import random
import re
from collections import OrderedDict

from state import State
import util

class Game(metaclass = ABCMeta):
    
    @abstractmethod
    def get_available_actions(self, state):
        """Returns the available actions for the input state.

        Keyword arguments:
        state -- the input state
        """
        pass
    
    @abstractmethod
    def get_current_state(self):
        """Returns the initial state.
        """
        pass
    
    @abstractmethod
    def get_reward(self, node):
        """Returns the reward for the input node.

        Keyword arguments:
        state -- the input node
        """
        pass
    
    @abstractmethod
    def is_done(self, state):
        """Returns if the game is done for the input state. 

        Keyword arguments:
        state -- the input state
        """
        pass
    
    @abstractmethod
    def simulate_action(self, state, action):
        """Applies the input action on the input state and returns the resulting state.

        Keyword arguments:
        state -- the input state
        action -- the input action
        """
        pass
    
    def evaluate_actions_at_state(self, actions, state):
        """Returns the value for each action for current state and if the game is done for the next state.
        
        Keyword arguments:
        actions -- actions for the current state
        action -- the current state
        """
        inp = np.array([(self.simulate_action(state, i)).state.reshape(self.sample.shape) for i in actions])
        is_done = self.is_done(inp)
        values = self.get_prediction_change(actions, state)
        return values, is_done
    
    @abstractmethod
    def get_prediction_change(self, actions, state):
        """Returns the change for every action the change in the prediction if a action is applied on the input state.

        Keyword arguments:
        state -- the input state
        actions -- the input actions
        """
        pass
    
    @abstractmethod
    def _generate_available_actions(self):
        """Creates and returns all available actions.
        """
        pass
        
    def get_next_action(self, state):
        """Returns the next action for a state in the rollout.
        
        Keyword arguments:
        state -- the input state
        """
        available_actions = self.get_available_actions(state)
        if len(available_actions) == 0:
            action = None
        else:
            action = random.choice(available_actions)
        return action

class Minus_Game(Game):
    
    def __init__(self, sample, predict, target_label, hide_value=0, kernel_size=1, max_depth=None, ratio=0.0, threshold=0.0):
  
        self.sample = sample 
        self.sample_dim = util.dim(sample)
        self.target_label = target_label
        self.hide_value = hide_value
        self.predict = predict
        
        self.kernel_size = kernel_size
        
        self.start_label = np.argmax(self.predict(np.expand_dims(self.sample, axis=0)))
        self.initial_state = self.get_current_state()
        self.all_actions, self.available_actions = self._generate_available_actions()
        
        self.max_depth = max_depth
        if max_depth is None:
            self.max_depth = len(self.available_actions)
        self.offset = 1
        self.ratio = ratio
        self.threshold = threshold
            
    def get_prediction_change(self, actions, state):
        new_states = [state.minus(State(i.state), hide_value=self.hide_value) for i in actions]
        inp1 = state.apply(State(self.sample)).state
        inp2 = [i.apply(State(self.sample), hide_value=self.hide_value).state.reshape(self.sample.shape) for i in new_states]
        inp2 = np.stack(inp2)
  
        out1 = self.predict(np.expand_dims(inp1, 0))[0, self.target_label]
        out1 = np.repeat(out1[np.newaxis], len(inp2), axis=0)
        out2 = self.predict(inp2)[:, self.target_label]
        
        if self.target_label == self.start_label :
            return out1 - out2
        elif self.target_label != self.start_label:
            return out2 - out1

    def get_reward(self, node):
        pred_t = 0.0
        if self.ratio > 0.0:
            state = node.state.apply(State(self.sample)).state
            #if self.sample_dim == 0:
            #    state = np.expand_dims(state, 0)
            pred_t = self.predict(np.expand_dims(state, axis=0))[0, self.start_label]
            if self.target_label == self.start_label:
                pred_t = 1 - pred_t
        reward = (1 - self.ratio) * max(1 - (node.depth - self.offset) / self.max_depth, 0) + self.ratio * pred_t
        assert reward >= 0
        return reward  
    
    def get_available_actions(self, state_leaf):
        actions = []
        for i in self.available_actions:
            if state_leaf.state.sum() - (state_leaf.state * i.state).sum() > self.kernel_size * self.kernel_size/2 and i in self.available_actions:
                actions.append(i)
        return actions
    
    def get_current_state(self):
        return State(np.ones(self.sample.shape))
   
    def is_done(self, state):
        if state.shape != self.sample.shape:
            states = [State(i).apply(State(self.sample), hide_value=self.hide_value).state for i in state]
            states = np.stack(states, axis=0)
            out = self.predict(states)
            pred_t = out[:, self.target_label]
            argmax = np.argmax(out, axis=1)
            if self.target_label == self.start_label:
                return np.logical_and(argmax != self.target_label, np.greater_equal(pred_t, self.threshold))
            else:
                return np.logical_and(argmax == self.target_label, np.less_equal(pred_t, self.threshold))
        else:
            inp = State(state).apply(State(self.sample), hide_value=self.hide_value).state
            out = self.predict(np.expand_dims(inp, axis=0))
            pred_t = out[0, self.target_label]
            argmax = np.argmax(out)
            if self.target_label == self.start_label and argmax != self.target_label and pred_t >= self.threshold:
                return True
            elif self.target_label != self.start_label and pred_t <= self.threshold:
                return True
            else:
                return False         
            
    def simulate_action(self, state, action):
        return state.minus(action)
    
    def _generate_available_actions(self):
        all_actions = []
        actions = []
        if self.sample_dim > 0:
            for i in np.arange(0, self.sample.shape[1] - self.kernel_size + 1, self.kernel_size):
                for j in range(0, self.sample.shape[0] - self.kernel_size + 1, self.kernel_size):
                    mask = np.ones(self.sample.shape)
                    mask[i:i+self.kernel_size, j:j+self.kernel_size] = 0
                    action = State(mask)
                    all_actions.append(action)
                    if not np.equal(self.sample, action.apply(State(self.sample), hide_value=self.hide_value).state).all():
                        actions.append(action)
        else:
            for j in range(0, self.sample.shape[0] - self.kernel_size + 1, self.kernel_size):
                mask = np.ones((self.sample.shape[0], ))
                mask[j:j+self.kernel_size] = 0
                action = State(mask)
                all_actions.append(action)
                if not np.equal(self.sample, action.apply(State(self.sample), hide_value=self.hide_value).state).all():
                        actions.append(action)
        return all_actions, actions