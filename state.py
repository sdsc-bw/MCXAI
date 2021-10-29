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
    
    def minus(self, other, hide_value=0):
        return State(np.logical_and(self.state.astype(bool), other.state.astype(bool)).astype(float) + hide_value * self.invert() +  hide_value * other.invert())
    
    def apply(self, other, hide_value=0):
        return State(self.state * other.state + hide_value * self.invert())
    
    def invert(self):
        return np.logical_xor(self.state, np.ones(self.state.shape), dtype=float).astype(float)
    
    def get_state(self):
        return self.state
    