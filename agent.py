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

max_len = 50

class Agent(metaclass = ABCMeta):
     
    def run(self, episodes=None, min_episodes=0, n_edges=0, step_when_leaf_not_done=True):
        """Runs the episodes of a MCTS. Can analyze a number of the best edges of the root more closly.

        Keyword arguments:
        episodes -- number of episodes
        n_edges -- number of edges that are looked at more closly
        """
        if n_edges == 0:
            i = 0
            distrb = []
            while True:
                self.episode(self.root)
                if i % 100 == 0:
                    self.logger.info('X'*70)
                    self.logger.info('Round:\t%d'%(i))
                    self.logger.info('X'*70)
                if i % max_len == 0 and i != 0:
                    curr_distrb = self.root.get_winrate_of_edges()
                    if len(distrb) > 0 and abs(util.kl_divergence(np.array(distrb), np.array(curr_distrb))) < self.eps and i > min_episodes:
                        self.logger.info('Distribution stable after \t%d episodes'%(i))
                        break
                    distrb = curr_distrb
                i += 1
                if episodes is not None and i >= episodes:
                    break
        else:
            best_edges = []
            for j in range(n_edges):
                if j == 0:
                    curr_root = self.root
                    depth = 0
                    distrb = []
                else:
                    curr_root = best_edges[j].out_node
                    depth = 1
                    distrb = curr_root.get_winrate_of_edges()

                while depth < len(self.game.available_actions):
                    i = 0
                    while True:
                        self.episode(curr_root)
                        if i % 1000 == 0:
                            self.logger.info('X'*70)
                            self.logger.info('Round:\t%d'%(i))
                            self.logger.info('X'*70)
                        if i % max_len == 0 and i != 0:
                            curr_distrb = curr_root.get_distribution()
                            if len(distrb) > 0 and abs(util.kl_divergence(np.array(distrb), np.array(curr_distrb))) < self.eps and i > min_episodes:
                                self.logger.info('Distribution stable after \t%d episodes'%(i))
                                break
                            distrb = curr_distrb
                        i += 1
                        if episodes is not None and i >= episodes:
                            break
                    edges = curr_root.sort_edges_by_N()
                    curr_root = edges[0].out_node
                    if j == 0 and depth == 0:
                        best_edges = edges[1:(n_edges + 1)]
                    #if curr_root.is_leaf() or (curr_root.game_is_done and curr_root.player == 1):
                    #    break
                    if curr_root.is_leaf() or self.game.is_done(curr_root.state.state, 0):
                        break
                    distrb = curr_root.get_distribution()
                    depth += 1

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
    
    def _create_edges_for_leaf_and_evaluate(self, leaf, player, root):
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
        values, is_dones = self.game.evaluate_actions_at_state(available_actions, state_leaf, player)
        edges = [MC_edge(action, leaf, MC_node(state, player=player, root=root, depth=leaf.depth + 1, game_is_done=is_done), self.c, value) for action, state, value, is_done in zip(available_actions, states, values, is_dones)]
        return edges, values

    def get_data(self):
        """Returns the states and their action winrates with more than 100 visits.
        """
        winrates_0 = []
        for action in self.game.all_actions:
            edge = self.root.get_edge_with_action(action)
            if edge == 0:
                winrates_0.append(0.0)
            else:
                winrates_0.append(edge.get_winrate())
        
        _, _, path = self.mct.selection_with_N()
        winrates_1 = []
        for action in self.game.all_actions:
            edge = path[0].in_node.get_edge_with_action(action)
            if edge == 0:
                winrates_1.append(0.0)
            else:
                winrates_1.append(edge.get_winrate())
        return winrates_0, winrates_1
    
    @abstractmethod
    def get_best_path(self):
        """Returns rank and mask of the path with the highest winrate.
        """
        pass
    
    @abstractmethod
    def get_best_actions(self, masked_sample, n=5):
        """Returns the ranks and mask of the input state with the n best actions.

        Keyword arguments:
        masked_sample -- a masked state of the initial sample
        n -- number of actions to output
        """
        pass
    
class Minus_Agent(Agent):
    
    def __init__(self, game, c=0.05):
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        self.game = game
        self.root_state = game.get_current_state()
        self.mct = MCFE_tree(self.root_state, logger = self.logger)
        self.root = self.mct.root
        self.mct.add_actions(self.game.available_actions) 
        self.num_episode = 0
        self.c = c
        self.eps = 0.000001
        self.max_depth = len(self.game.available_actions) * 0.75
        self.distrb = []
    

    def episode(self, root, player=0):
        self.num_episode += 1
        node, path_0, path_1  = self.mct.selection(root=root)
        self.mct.add_node_to_tree(node)
        if node.player == 0 or not node.game_is_done:
            # didn't reach the maximum level, expand
            # evaluate node state
            if node.player == 0 and node.game_is_done:
                root = node
                player = 1
            elif node.player == 1: 
                player = 1
                root = node.root
            else:
                root = self.root
            edges, values = self._create_edges_for_leaf_and_evaluate(node, player, root)
            if len(edges) != 0:
                # expansion with ts
                expanded_edge = self.mct.expansion(node, edges, values)
                if player == 0:
                    path_0.append(expanded_edge)
                else:
                    path_1.append(expanded_edge)
                node = expanded_edge.get_out_node()
        if not node.game_is_done:
            r_node = self.roll_out(node)
        else:
            r_node = node
            
        if len(path_0) == 0 and len(path_1) == 0:
            reward_0 = self.game.get_reward(r_node)
            reward_1 = 0
        elif len(path_0) != 0 and len(path_1) == 0:
            reward_0 = self.game.get_reward(r_node)
            reward_1 = 0
        elif len(path_0) != 0 and len(path_1) != 0:
            reward_0 = self.game.get_reward(path_0[-1].out_node)
            reward_1 = self.game.get_reward(r_node)
        else:
            reward_0 = 0
            reward_1 = self.game.get_reward(r_node)
        self.mct.back_fill(reward_0, reward_1, path_0, path_1)
    
    def roll_out(self, leaf): 
        new_state = leaf.state
        new_node = leaf
        counter = 1
        
        states = [leaf.state]
        nodes = [leaf]
        while counter < self.max_depth:
            action = self.game.get_next_action(new_state, leaf.player)
            if action is None:
                break
            new_state = self.game.simulate_action(new_state, action)
            new_node = MC_node(new_state, depth=leaf.depth + counter, player=leaf.player, root=leaf.root)
            states.append(new_state)
            nodes.append(new_node)
            counter += 1
            new_node.game_is_done = self.game.is_done(new_node.state.state, leaf.player)
            if new_node.game_is_done:
                break
        return new_node
    
    def get_best_path(self):
        _, path_0, path_1 = self.mct.selection_with_N()
        n = len(path_0)
        ranks_0 = np.zeros(self.game.sample.shape)
        if self.game.start_label == self.game.target_label:
            factor = 1
        else:
            factor = -1
        for i in range(1, n + 1):
            action = self.game.invert(path_0[i - 1].action.state)
            ranks_0 += factor * i * action
        mask_0 = path_0[-1].out_node.state.state
            
        ranks_1 = np.zeros(self.game.sample.shape)
        mask_1 = np.ones(self.game.sample.shape)
        if len(path_1) > 0 and path_1[-1].out_node.game_is_done:
            n = len(path_1)
            factor *= -1
            for i in range(1, n + 1):
                action = self.game.invert(path_1[i - 1].action.state)
                ranks_1 += factor * i * action
                
            mask_1 = path_1[-1].out_node.state.state
        return ranks_0, ranks_1, mask_0, mask_1

    def get_best_path_as_list(self):
        _, path_0, path_1 = self.mct.selection_with_N()
        n = len(path_0)
        ranks_0 = []
        if self.game.start_label == self.game.target_label:
            factor = 1
        else:
            factor = -1
        for i in range(1, n + 1):
            ranks_0.append((path_0[i - 1].out_node.state.state[-1].strip(), i))
            
        ranks_1 = []
        if len(path_1) > 0 and path_1[-1].out_node.game_is_done:
            n = len(path_1)
            factor *= -1
            for i in range(1, n + 1):
                ranks_1.append((path_1[i - 1].out_node.state.state[-1].strip(), i))
                
        return ranks_0, ranks_1, path_0, path_1

    def get_best_actions(self, masked_sample, n=5):
        if n == 0:
            return  np.zeros(self.game.sample.shape), np.zeros(self.game.sample.shape)
        # get node with highest N
        best_node = None
        for node in self.mct.tree:
            if node.is_leaf():
                continue
            if torch.allclose(masked_sample, node.state.state * self.game.sample):
                if best_node is None or best_node.N < node.N:
                    best_node = node
        if best_node is None: 
            raise KeyError('State not in trees.')
        edges = best_node.sort_edges_by_winrate()[:n]
        ranks = np.zeros(self.game.sample.shape)
        mask = np.zeros(self.game.sample.shape)
        if self.game.start_label == self.game.target_label and best_node.get_first_child().player == 0 or self.game.start_label != self.game.target_label and best_node.get_first_child().player == 1 :
            factor = 1
        else:
            factor = -1
        for i in range(len(edges)):
            action = self.game.invert(edges[i].action.state)
            ranks += factor * (i + 1) * action
            mask += action
        return ranks, mask 
