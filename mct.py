import numpy as np
from numpy import log as ln
import pandas as pd
import random
import copy
import pickle

from state import State

class MC_node():
    def __init__(self, state, game_is_done=False, parent_edge=None, depth=None):
        self.state = state
        self.N = 0
        self.edges = []
        self.sorted = False # sinify whether self.edges is sorted or not, only roll out in MCT can set it to false
        self.depth = depth
        self.parent_edge = parent_edge
        self.game_is_done = game_is_done
        
    def add_edge(self, e):
        self.edges.append(e)
        
    def add_edges(self, es):
        self.edges.extend(es)
        
    def is_leaf(self):
        if len(self.edges) == 0:
            return True
        else:
            return False
        
    def get_actions_of_edges(self):
        return [i.action for i in self.edges]
    
    def get_first_child(self):
        return self.edges[0].get_out_node()
    
    def get_children(self):
        return [i.get_out_node() for i in self.edges]
    
    def get_edge_with_action(self, action):
        for i in self.edges:
            if np.equal(i.action.state, action.state):
                return i
        return 0
    
    def get_infor_of_edges(self):
        dat = pd.DataFrame(columns = self.get_actions_of_edges())
        dat.loc['N', :] = self.get_N_of_edges()
        dat.loc['W', :] = self.get_W_of_edges()
        dat.loc['c', :] = self.get_c_of_edges()
        dat.loc['Win rate', :] = self.get_winrate_of_edges()
        dat.loc['Part2', :] = self.get_part()
        dat.loc['Value', :] = self.get_value_of_edges()
        dat.loc['Original Value'] = self.get_orig_value_of_edges()
        dat.loc['Game is done'] = self.get_done_of_edges()
        return dat
            
    def get_num_edges(self):
        return len(self.edges)
    
    def get_N_of_edges(self):
        return [i.N for i in self.edges]
    
    def get_c_of_edges(self):
        return [i.c for i in self.edges]
    
    def get_part(self):
        return [i.get_part() for i in self.edges]
    
    def get_value_of_edges(self):
        return [i.get_value() for i in self.edges]
    
    def get_orig_value_of_edges(self):
        return [i.value for i in self.edges]
    
    def get_winrate_of_edges(self):
        return [i.get_winrate() for i in self.edges]
    
    def get_distribution(self):
        return [i.get_distribution() for i in self.edges]
    
    def get_W_of_edges(self):
        return [i.W for i in self.edges]
    
    def get_done_of_edges(self):
        return [i.out_node.game_is_done for i in self.edges]
    
    def get_N(self):
        return self.N
    
    def get_state(self):
        return self.state
    
    def reset_sorted(self):
        ### Aborted
        self.sorted = False
    
    def sort_edges_by_value(self):
        return sorted(self.edges, key = lambda x: x.get_value(), reverse = True)
    
    def sort_edges_by_winrate(self):
        return sorted(self.edges, key = lambda x: x.get_winrate(), reverse = True)
    
    def sort_edges_by_N(self):
        return sorted(self.edges, key = lambda x: (x.N, x.value), reverse = True)
    
    def get_sum_winrates(self):
        return sum(self.get_winrate_of_edges())
    
    def __eq__(self, other):
        return self.state == other.state
    
    def __hash__(self):
        return hash(self.state)

class MC_edge():
    
    def __init__(self, action, in_node, out_node, c, value):
        self.action = action
        self.in_node = in_node
        self.out_node = out_node
        self.out_node.parent_edge = self
        self.value = value
        self.N = 0
        self.W = 0
        self.c = c
        
    def get_in_node(self):
        return self.in_node
    
    def get_out_node(self):
        return self.out_node
    
    def get_action(self):
        return self.action
    
    def get_state(self):
        return (self.Q, self.U, self.W, self.N, self.P)
    
    def get_value(self):
        if self.N == 0:
            return 4 + self.value
        else:
            return self.get_winrate() + self.c*np.sqrt(ln(self.in_node.N)/(self.N))
        
    def get_part(self):
        if self.N == 0:
            return self.value
        else:
            return np.sqrt(ln(self.in_node.N)/self.N)
        
    def get_winrate(self):
        if self.N == 0:
            return self.value
        else:
            return self.W/self.N
        
    def get_distribution(self):
        if self.N == 0:
            return 0.0
        else:
            return self.W/self.N
        
        
class MCFE_tree:

    def __init__(self, root_state):
        # init attribute
        self.root = MC_node(root_state, depth=0)
        self.tree = set()
        self.actions = set()
        self.add_node_to_tree(self.root)
            
    def add_node_to_tree(self, node):
        if node not in self.tree:
            self.tree.add(node)
            return 1
        else:
            return 0
        
    def add_actions(self, action_list):
        for a in action_list:
            self.actions.add(State(a))
            
    def back_fill(self, reward, node):
        """
        Performs the backpropagation.
        
        Keyword arguments:
        reward -- reward for the current node/state
        node -- node corresponding to the current state
        """
        n = node
        n.N += 1
        while n != self.root:
            #print("Hello")
            #print(n)
            edge = n.parent_edge
            edge.N += 1
            edge.W += reward
            n = n.parent_edge.in_node
            n.N += 1
                
    def expansion(self, leaf, edges, values):
        """
        Adds edges to tree and returns most valuable edge.
                
        Keyword arguments:
        leaf -- leaf that is supposed to be expanded
        edges -- new edges
        values -- related values of the edges/actions
        
        """
        leaf.add_edges(edges)
        out = edges[0]
        best_score= values[0]
        for edge, value in zip(edges, values):
            if best_score < value:
                best_score = value
                out = edge  
        self.add_node_to_tree(out.get_out_node())
        return out

            
    def selection(self, root=None):
        """
        Selection with UCT1
        
        Keyword arguments:
        leaf -- node from where the selection starts
        """
        path = []
        current_node = root
        if root.is_leaf():
            return root, path # the paths here are empty
        else:
            while not current_node.is_leaf():
                edges = current_node.sort_edges_by_value()
                edge = edges[0]
                path.append(edge)
                current_node = edge.get_out_node()
        return current_node, path
    
    def selection_with_N(self):
        """
        Selection with highest N.
        """
        path = []
        current_node = self.root
        if self.root.is_leaf():
            return self.root, path # the paths here are empty
        else:
            while not current_node.is_leaf():
                edges = current_node.sort_edges_by_N()
                edge = edges[0]
                path.append(edge)
                current_node = edge.get_out_node()
        return current_node, path