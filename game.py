from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import sparse
import logging
import random

from state import State

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
    
    def evaluate_actions_at_state(self, actions, state, player):
        """Returns the value for each action for current state for the current player and if the game is done for the next state.
        
        Keyword arguments:
        actions -- actions for the current state
        action -- the current state
        player -- the current player
        """
        inp = np.array([self.simulate_action(state, i.state).state * self.sample for i in actions])
        is_done = self.is_done(inp, player)
        values = self.get_prediction_change(actions, state, player)
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
    def normalize(self, state):
        """Normalizes the state in order to create a mask based on the game.

        Keyword arguments:
        state -- the input state
        """
        pass
    
    @abstractmethod
    def _generate_available_actions(self):
        """Creates and returns all available actions.
        """
        pass
        
    def get_next_action(self, state, player):
        """Returns the next action for a state in the rollout.
        
        Keyword arguments:
        state -- the input state
        player -- the current player
        """
        available_actions = self.get_available_actions(state)
        if len(available_actions) == 0:
            action = None
        else:
            action = random.choice(available_actions)

class Minus_Game(Game):
    
    def __init__(self, sample, predict, target_label, kernel_shape=(1,1), max_depth=None, ratio=0.0, threshold=0.0, network_0=None, network_1=None, logger = None):
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
            
        self.sample = sample 
        self.start_label = np.argmax(predict(sample))
        self.target_label = target_label 
        self.predict = predict
        
        self.kernel_shape = kernel_shape
        
        self.initial_state = self.get_current_state()
        self.all_actions, self.available_actions = self._generate_available_actions()
        
        #self.network_0 = network_0
        #if self.network_0 is not None:
        #self.network_1 = network_1
        #if self.network_1 is not None:
        #    self.network_1.eval()
        
        self.max_depth = max_depth
        if max_depth is None:
            self.max_depth = kernel_shape[0] * kernel_shape[1]
        self.offset = 1
        self.ratio = ratio
        self.threshold = threshold
            
    def get_prediction_change(self, actions, state, player):
        new_states = [state.minus(State(i.state)) for i in actions]
        inp1 = state.state * self.sample
        inp2 = np.concatenate([i.state * self.sample for i in new_states])
        
        out1 = self.predict(inp1)[0, self.target_label]
        out1 = np.repeat(out1[np.newaxis], len(inp2), axis=0)
        out2 = self.predict(inp2)[:, self.target_label]
        if player == 0 and self.target_label == self.start_label or player == 1 and self.target_label != self.start_label:
            return out1 - out2
        elif player == 0 and self.target_label != self.start_label or player == 1 and self.target_label == self.start_label:
            return out2 - out1

    def get_reward(self, node):
        pred_t = 0.0
        if self.ratio > 0.0:
            state = node.state.state * self.sample
            pred_t = self.predict(state)[0, self.start_label]
            if self.target_label == self.start_label:
                pred_t = 1 - pred_t
        reward = (1 - self.ratio) * max(1 - (node.depth - node.root.depth - self.offset) / self.max_depth, 0) + self.ratio * pred_t
        assert node.root.depth < node.depth
        return reward  
    
    def get_available_actions(self, state_leaf):
        actions = []
        for i in self.available_actions:
            if state_leaf.state.sum() - (state_leaf.state * i.state).sum() > self.kernel_shape[0]*self.kernel_shape[1]/2 and i in self.available_actions:
                actions.append(i)
        return actions
    
    def get_current_state(self):
        return State(np.ones(self.sample.shape))
   
    def is_done(self, state, player):
        if state.shape != self.sample.shape:
            inp = np.concatenate([i * self.sample for i in state])
            out = self.predict(inp)
            pred_t = out[:, self.target_label]
            argmax = np.argmax(out, axis=1)
            if player == 0 and self.target_label == self.start_label:
                return np.logical_and(argmax != self.target_label, np.less_equal(pred_t, 1 - self.threshold))
            elif player == 0 and self.target_label != self.start_label:
                return np.logical_and(argmax == self.target_label, np.greater_equal(pred_t, self.threshold))
            elif player == 1 and self.target_label == self.start_label:
                return np.logical_and(argmax == self.target_label, np.greater_equal(pred_t, self.threshold))
            else:
                return np.logical_and(argmax != self.target_label, np.less_equal(pred_t, 1 - self.threshold))
        else:
            inp = state * self.sample
            out = self.predict(inp)
            pred_t = out[0, self.target_label]
            argmax = np.argmax(out)
            if player == 0 and self.target_label == self.start_label and argmax != self.target_label and pred_t <= 1 - self.threshold:
                return True
            elif player == 0 and self.target_label != self.start_label and argmax == self.target_label and pred_t >= self.threshold:
                return True
            elif player == 1 and self.target_label == self.start_label and argmax == self.target_label and pred_t >= self.threshold:
                return True
            elif player == 1 and self.target_label != self.start_label and argmax == self.target_label and pred_t <= 1 - self.threshold:
                return True
            else:
                return False         
            
    def simulate_action(self, state, action):
        return state.minus(State(action))
    
    def normalize(self, state):
        return np.logical_xor(state, np.ones(self.sample.shape), dtype=float).astype(float)
    
    def _generate_available_actions(self):
        all_actions = []
        actions = []
        for i in range(0, self.sample.shape[1] - self.kernel_shape[1] + 1, self.kernel_shape[0]):
            for j in range(0, self.sample.shape[0] - self.kernel_shape[0] + 1, self.kernel_shape[1]):
                mask = np.ones(self.sample.shape)
                mask[j:j+self.kernel_shape[1], i:i+self.kernel_shape[0]] = 0
                action = State(mask)
                all_actions.append(action)
                if not np.equal(self.sample, self.sample * mask).all():
                    actions.append(action)
        return all_actions, actions
            