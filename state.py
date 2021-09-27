import numpy as np
import util

class State():
    def __init__(self, state):
        self.state = state
        
    def __repr__(self):
        return str(self.state)
        
    def __eq__(self, other):
        return np.equal(self.state, other.state).all()
    
    def __hash__(self):
        return hash(str(self.state))
    
    def __len__(self):
        return len(self.state)
    
    def add(self, other):
        return State(np.logical_or(self.state.astype(bool), other.state.astype(bool)).astype(float))
    
    def minus(self, other):
        return State(np.logical_and(self.state.astype(bool), other.state.astype(bool)).astype(float))
    
    def get_state(self):
        return self.state
    
    def __iter__(self):
        return StateIterator(self)
    
class Text_State(State):
    
    def __init__(self, state):
        super(Text_State, self).__init__(state)
        
    def __eq__(self, other):
        return np.equal(np.array(self.state), np.array(other.state)).all() 

    def __hash__(self):
        return hash(str(self.state))   
    
    def add(self, other):
        return Text_State(self.state + other.state)
    
    def minus(self, other):
        return Text_State(self.state - other.state)
    
class StateIterator:
    
    def __init__(self, states):
        self._states = states
        self._index = 0
    
    def __next__(self):
        self._index += 1
        if self._index >= len(self._states):
            self._index = -1
            raise StopIteration
        else:
            return self._states[self._index]
            