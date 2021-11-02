# MCXAI
To this day, a variety of approaches for providing local interpretability of black-box machine learning models have been introduced. Unfortunately, all of these methods suffer from one or more of the following deficiencies:  They are either difficult to understand themselves, they work on a per-feature basis and ignore the dependencies between features and/or they only focus on those features asserting the decision made by the model. To address these points, this work introduces a reinforcement learning-based approach called Monte Carlo tree search for eX-plainable Artificial Intelligent (McXai) to explain the decisions of any black-box classification model (classifier). Our method leverages Monte Carlo tree search and models the process of generating explanations as two games. In one game, the reward is maximized by finding feature sets that support the decision of the classifier, while in the second game, finding feature sets leading to alternative decisions maximizing the reward.  The result is a human friendly representation as a tree structure, in which each node represents a set of features to be studied with smaller explanations at the top of the tree. Our experiments show, that the features found by our  method are more informative with respect to classifications  thanthose found by classical approaches like [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap). Furthermore, by also identifying misleading features, our approach is able to guide towards improved robustness of the black-box model in many situations.

## Requirements
The code created with the help of the followig packages and tested with the shown versions:
Package       | Version
------------- | -------------
Python        | 3.8.12
Tensorflow    | 2.3.0
Scikit-learn  | 0.24.2

## How To Use
To access the code, you can download this repo into your files. We provided examples in the files [explain_covertype.ipynb](./explain_covertype.ipynb) and [explain_mnist.ipynb](./explain_mnist.ipynb) based on the [forest covertypes dataset](https://scikit-learn.org/0.16/datasets/covtype.html) from scikit-learn and the [mnist dataset](https://www.tensorflow.org/datasets/catalog/mnist) from tensorflow. 
In both we create an instance from the class *Explainer*:
```
explainer = Explainer(sample, predict, target_label, max_episodes=...)
```
The following parameters can be set:
* sample: sample to be explained
* predict: prediction function of the model (return probabilitis for each class)
* target_label: label which should be explained
* max_episodes: maximal number of episodes per step
* eps: constant to define when the current node is stable
* c: exploration-parameter, defines the dimension of exploration in the MCT
* kernel_size: size of the kernel for segmentation
* max_depth: defines the maximal depth during search (smaller depths lead to a faster convergence)
* ratio: defines the impact of the prediction probability in the reward (ratio=0.0 means only the depth of the node/state has an impact)
* threshold_1: defines the threshold of the predicted probability of ending the game for the first game (threshold_1=0.0 means game is finished when target class is no longer the predicted class and threshold_1 = 0.2 means probability of target class needs to be below 0.2) 
* threshold_2: defines the threshold of the predicted probability of ending the game for the second game (threshold_1=1.0 means game is finished when target class is the predicted class and threshold_1 = 0.2 means probability of target class needs to be above 0.8) 
