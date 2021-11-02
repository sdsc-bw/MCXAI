from abc import ABCMeta, abstractmethod

from mct import MC_node, MC_edge, MCFE_tree
from state import State
import util

import copy
import numpy as np
import pandas as pd
import random

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

max_dist = 50

class Agent(metaclass = ABCMeta):
     
    def run(self, eps=0.000001, max_episodes=10000):
        """Runs the episodes of a MCTS.

        Keyword arguments:
        eps -- constant to define when distribution is stable
        episodes -- number of episodes per step
        """     
        depth = 0
        curr_root = self.root
        while depth < len(self.game.available_actions):
            if depth == 0:
                distrb = []
            else:
                distrb = curr_root.get_distribution()
            for i in range(max_episodes):
                self.episode(curr_root)
                if i % 1000 == 0:
                    logger.info('X'*70)
                    logger.info('Episode:\t%d'%(i))
                    logger.info('X'*70)
                if i % max_dist == 0 and i != 0:
                    curr_distrb = curr_root.get_distribution()
                    if len(distrb) > 0 and abs(util.kl_divergence(np.array(distrb), np.array(curr_distrb))) < eps:
                        logger.info('Distribution stable after \t%d episodes'%(i))
                        break
                    distrb = curr_distrb
            edges = curr_root.sort_edges_by_N()
            curr_root = edges[0].out_node
            depth += 1
            if curr_root.game_is_done:
                break
            logger.info('Maximum number of episodes reached')
            
            
    @abstractmethod
    def episode(self, root):
        """Performs a episode of the MCTS. Starts the selection at the given root node.

        Keyword arguments:
        root -- node from where to start the selection
        """
        pass
    
    @abstractmethod
    def roll_out(self, leaf):
        """Performs a rollout of the MCTS for a given leaf. Returns the end node.

        Keyword arguments:
        leaf -- node from where to start the rollout
        """
        pass                
                    
    def get_results(self):
        """Returns the path and the leaf node of the path with the highest winrate.
        """
        return self.mct.selection_with_N()
    
    def _create_edges_for_leaf_and_evaluate(self, leaf):
        """Returns the possible edges and their values for the input leaf.

        Keyword arguments:
        leaf -- the input leaf 
        """
        state_leaf = leaf.state
        # get index of available_actions
        available_actions = self.game.get_available_actions(state_leaf)
        if len(available_actions) == 0:
            return [], []
        states = [self.game.simulate_action(state_leaf, action) for action in available_actions]
        values, is_dones = self.game.evaluate_actions_at_state(available_actions, state_leaf)
        edges = [MC_edge(action, leaf, MC_node(state, depth=leaf.depth + 1, game_is_done=is_done), self.c, value) for action, state, value, is_done in zip(available_actions, states, values, is_dones)]
        return edges, values
    
class Minus_Agent(Agent):
    
    def __init__(self, game, c):
        self.game = game
        self.c = c
        self.root_state = self.game.get_current_state()
        self.mct = MCFE_tree(self.root_state)
        self.root = self.mct.root
        self.mct.add_actions(self.game.available_actions)
        self.num_episode = 0
        self.max_depth = self.game.max_depth
    

    def episode(self, root):
        self.num_episode += 1
        node, _  = self.mct.selection(root)
        self.mct.add_node_to_tree(node)
        if not node.game_is_done:
            # didn't reach the maximum level, expand
            # evaluate node state
            edges, values = self._create_edges_for_leaf_and_evaluate(node)
            if len(edges) != 0:
                # expansion with ts
                expanded_edge = self.mct.expansion(node, edges, values)
                node = expanded_edge.get_out_node()
            r_node = self.roll_out(node)
        else:
            r_node = node
            
        reward = self.game.get_reward(r_node)
        self.mct.back_fill(reward, node)
    
    def roll_out(self, leaf): 
        new_state = leaf.state
        new_node = leaf
        counter = 1
        
        states = [leaf.state]
        nodes = [leaf]
        while counter < self.max_depth:
            action = self.game.get_next_action(new_state)
            if action is None:
                break
            new_state = self.game.simulate_action(new_state, action)
            new_node = MC_node(new_state, depth=leaf.depth + counter)
            states.append(new_state)
            nodes.append(new_node)
            counter += 1
            new_node.game_is_done = self.game.is_done(new_node.state.state)
            if new_node.game_is_done:
                break
        return new_node
    
    def get_masked_sample(self):
        node, path = self.mct.selection_with_N()
        return node.state.apply(State(self.game.sample)).state