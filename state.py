import numpy as np

class State():
    def __init__(self, state):
        self.state = state
        
    def __repr__(self):
        return str(self.state)
        
    def __eq__(self, other):
        return np.equal(self.state, other.state).all()
    
    def __hash__(self):
        return hash(self.__repr__())
    
    def __len__(self):
        return len(self.state)
    
    def add(self, other):
        return State(np.logical_or(self.state.astype(bool), other.state.astype(bool)).astype(float))
    
    def minus(self, other):
        return State(np.logical_and(self.state.astype(bool), other.state.astype(bool)).astype(float))
    
    def get_state(self):
        return self.state