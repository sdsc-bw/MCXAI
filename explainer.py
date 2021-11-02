import math
import numpy as np

from game import Minus_Game
from agent import Minus_Agent
import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
class Explainer:
    
    def __init__(self, sample, predict, target_label, hide_value=0, max_episodes=10000, eps=0.000001, c=math.sqrt(2), kernel_size=1, max_depth=None, ratio=0.0, threshold_1=0.0, threshold_2=None):
        """Constructor of the explainer
        
        Keyword arguments:
        sample -- sample to be explained
        predict -- prediction function of the model (return probabilitis for each class)
        target_label -- label which should be explained
        max_episodes -- maximal number of episodes per step
        eps -- constant to define when the current node is stable
        c -- exploration-parameter, defines the dimension of exploration in the MCT
        kernel_size -- size of the kernel for segmentation
        max_depth -- defines the maximal depth during search
        ratio -- defines the impact of the prediction probability in the reward
        threshold_1 -- defines the threshold of the predicted probability of ending the game for the first game
        threshold_2 -- defines the threshold of the predicted probability of ending the game for the second game
        """
        self.sample = sample
        self.predict = predict
        self.target_label = target_label
        self.hide_value = hide_value
        self.max_episodes = max_episodes
        self.eps = eps
        self.c = c
        self.kernel_size = kernel_size
        self.max_depth = max_depth
        self.ratio = ratio
        self.threshold_1 = threshold_1
        if threshold_2:
            self.threshold_2 = threshold_2
        else:
            self.threshold_2 = 1 - self.threshold_1
        
        self.pred_label = np.argmax(self.predict(np.expand_dims(sample, axis=0)))
        if self.target_label == self.pred_label:
            # if prediction is correct, first initialize the first game/agent and initilize second game/agent after first is finished
            self.game_1 = Minus_Game(sample, self.predict, self.target_label, self.hide_value, kernel_size=self.kernel_size, max_depth=self.max_depth, ratio=self.ratio, threshold=self.threshold_1)
            self.agent_1 = Minus_Agent(self.game_1, c=c)
            
            self.game_2 = None
            self.agent_2 = None  
        else:
            # if prediction is false, skip first game/agent and initialize the second game/agent
            self.game_1 = None
            self.agent_1 = None
            
            self.game_2 = Minus_Game(sample, self.predict, self.target_label, self.hide_value, kernel_size=self.kernel_size, max_depth=self.max_depth, ratio=self.ratio, threshold=self.threshold_2)
            self.agent_2 = Minus_Agent(self.game_2, c=c)
        
    def explain(self, both_games=True):
        """Runs the MCTS to create an explanation.
        
        Keyword arguments:
        both_games -- defines if the explainer should run both games
        """
        if both_games and self.target_label == self.pred_label:
            # run agent 1
            self.agent_1.run(eps=self.eps, max_episodes=self.max_episodes)
            masked_sample = self.agent_1.get_masked_sample()
            logger.info('Classification game finished')
            # initialize and run agent 2
            self.game_2 = Minus_Game(masked_sample, self.predict, self.target_label, self.hide_value, kernel_size=self.kernel_size, max_depth=self.max_depth, ratio=self.ratio, threshold=self.threshold_2)
            self.agent_2 = Minus_Agent(self.game_2, c=self.c)
            self.agent_2.run(eps=self.eps, max_episodes=self.max_episodes)
            logger.info('Misclassification game finished')
        elif self.target_label == self.pred_label:
            # only run agent 1
            self.agent_1.run(eps=self.eps, max_episodes=self.max_episodes)
            logger.info('Classification game finished')
        else:
            # only run agent 2
            self.agent_2.run(eps=self.eps, max_episodes=self.max_episodes)
            logger.info('Misclassification game finished')
        
    def get_explanation(self):
        """Returns an array with the rank of each feature. Rank 0 means that the sample is not in the best path.
        """
        ranks_1 = np.zeros(self.sample.shape)
        ranks_2 = np.zeros(self.sample.shape)
        if self.agent_1:
            _, path = self.agent_1.get_results()
            if path[-1].out_node.game_is_done:
                n = len(path)
                for i in range(1, n + 1):
                    feature = path[i - 1].action.invert()
                    ranks_1 += i * feature
        if self.agent_2:
            _, path = self.agent_2.get_results()
            if path[-1].out_node.game_is_done:
                n = len(path)
                for i in range(1, n + 1):
                    feature = path[i - 1].action.invert()
                    ranks_2 += i * feature  
        return ranks_1, ranks_2
    
    def get_explanation_as_list(self, feature_names=None):
        """Returns a list with the features of the best path and their corresponding ranks.
        """
        ranks_1 = []
        ranks_2 = []
        if self.agent_1:
            _, path = self.agent_1.get_results()
            if path[-1].out_node.game_is_done:
                n = len(path)
                for i in range(1, n + 1):
                    feature = path[i - 1].action.invert()
                    if feature_names is None:
                        ranks_1.append((np.argmax(feature), i))
                    else:
                        ranks_1.append((feature_names[np.argmax(feature)], i))
        if self.agent_2:
            _, path = self.agent_2.get_results()
            if path[-1].out_node.game_is_done:
                n = len(path)
                for i in range(1, n + 1):
                    feature = path[i - 1].action.invert()
                    if feature_names is None:
                        ranks_2.append((np.argmax(feature), i))
                    else:
                        ranks_2.append((feature_names[np.argmax(feature)], i))  
        return ranks_1, ranks_2
    