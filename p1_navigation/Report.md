# Report

## Overview of implementation

* Environment and agent setup, agent training, plotting of agent scores after each episode of training is done in `Navigation.ipynb`
* The Q function representation is encoded in `model.py` using a non-linear function approximator. Currently, it provides a single representation of the Q-function as a fully connected Neural Network. 
    * The network has three layers, each layer is a fully connected layer using a RELU activation function. The input to the first layer is 37-dimensional, to accept the agent-state. 
    * The last layer uses a Softmax activation function; it produces a 4-dimensional vector to enable selection of 1-of-4 actions in that state. 
    * The 37-dimensional input vector is fully-connected to a layer of 64 neurons; these are connected to a layer of 32 neurons, which are then connected to the output layer having 4 neurons.
* The Agent is defined in `dqn_agent.py`. Its definition includes methods `step`, `learn` and `act`; it also has a `soft_update` method described below.
    * The `step` method adds agent experience to the experience-replay memory. Periodically (specified by `UPDATE_EVERY` parameter), the agent checks if the memory has enough (`BATCH_SIZE`) samples and initiates a call to the `learn` method of the agent. More on this in the next section.
    * The `act` method performs action selection in a given state as per an $\epsilon$-greedy policy. It calls the Q-model with the current state, recovers the action values for every possible action in the current state and selects one of them based on the policy. 
    * The `learn` method updates the Q-function representation of the agent - this is the training process. For each given tuple of `(state, action, reward, next_state, done)`  (sampled from the experience replay memory), the agent computes the action-value function using the target Q-network and the local Q-network and performs gradient based minimiztion of this loss while adjusting the weights of the local Q-network. More on the local and target Q-networks in the next section.
    * After performing a learning step, the learning algorithm revises the target Q-function weights to be a weighted sum (controlled by parameter $\tau$) of its current weights and the weights of the local Q-function.
* The Experience Replay Buffer is also defined in `dqn_agent.py`. It is a double-ended queue data structure containing agent-environement interactions represented by tuples `(state, action, reward, next_state, done)`, as they are obtained. 
    * Each tuple encodes the current state of the agent, its chosen action, the reward obtained, the next state the agent finds itself in and a `done` flag suggesting if the episode has ended or not. 
    * The Replay Buffer includes methods to add new exemplars to the data structure and to sample from it.
* The learning algorithm is provided in `learn_alg.py`. Currently, it includes the standard Deep Q-learning algorithm proposed by Mnih et al, "Human-level control through deep reinforcement learning", Nature, Vol 518, Feb 2015. More on the algorithm in the section below.

## Learning algorithm

* Two key algorithms are used - Deep Q-learning and Experience Replay. Both are covered in detail in Mnih et al, "Human-level control through deep reinforcement learning", Nature, Vol 518, Feb 2015 and Experience. A brief idea of each is presented below.
* Deep Q-learning (DQN)
    * The objective of training an agent is to maximize its expected discounted rewards obtained while interacting with the environment. In the current context, the agent needs to learn that collecting yellow banannas leads to good outcomes and collecting blue banannas leads to poor outcomes. 
    * For each state that the agent can find itself in, it can perform a defined set of actions. Therefore, to learn to maximize rewards, the agent needs to learn a function that can recommend actions to be performed in each state. This unknown and to-be-estimated function is the Q-function $Q(s,a)$, where $s$ and $a$ denote the state and action respectively. In the current context, the Q-function is represented by a non-linear function approximator - a Neural Network function $Q(s,a|\theta)$ where $\theta$ are the weights of the neural network.
    * The weights $\theta$ may be learnt by minimizing the squared-loss between the target and predicted (local) action values. Since both depend on the weights $\theta$, one of them has to be fixed, while adjusting the weights of the other. Thus, the DQN algorithm basically uses two identically structured Q-networks to learn the Q-function for the agent.
    * At each step of the optimization algorithm, the target Q-function uses the weights of the local Q-network from a previous time-update (denoted by $Q(s,a|\theta^-$)). The loss function to be minimized is given by $E_{(s,a,r,s')} \left[\left( r + \gamma.\max_{a'} Q(s',a'|\theta^-)  - Q(s,a|\theta) \right)^2\right]$ . Here, $r$ is the reward obtained by performing action $a$ in state $s$ and going to state $s'$ where action $a'$ has the maximum action-value as per the current target function. This can be done using standard gradient-descent based methods and their variants, that are typically used in Neural Network training. As a result the local Q-function will have its weights adjusted towards approximating the expected discounted rewards obtained by the agent interacting with the environment using the target Q-function.
    * Periodically, the target Q-function weights are replaced by a weighted combination of its current weights and the local Q-function weights; in the limit, this could be a simple replacement of the target Q-network weights with the local Q-network weights.
* Experience Replay
    * It is understood that using non-linear function approximators results in unstable or divergent learning. the root cause for this behavior is understood to be the correlation between the experience observations.
    * Originally suggested for efficient sample re-use, Experience Replay involves storing experience observations in a data structure and sampling a subset of them towards use for training the agent. This idea has been found to be effective at breaking the correlations between a sequence of observations and thus enabling effective training.

## Brief description of hyperparameters

* In `learn_alg.py`, the DQN algorithm uses several hyperparameters described below
    * `n_episodes`: maximum number of episodes of training
    * `max_t`: maximum number of time steps per episode; a high-value should suffice
    * `eps_start`: initial value of $\epsilon$ for the $\epsilon$-greedy policy - the agent chooses the action with the maximum Q-value with probability $1-\epsilon$ and randomly any of the other actions using a probability of $\epsilon$.
    * `eps_end`: final value of $\epsilon$. Starting from high $\epsilon$ and reducing to a low value results in the agent favoring exploration (of different actions) initially and gradually favoring exploitation (choosing action with max Q-value) as $\epsilon$ reduces.
    * `eps_decay`: this controls the rate at which $\epsilon$ decays with time.
* In `dqn_agent.py`, the agent and experience replay buffer use the following hyperparameters
    * `BUFFER_SIZE`: size of the double-ended queue data structure containing agent-environment interaction observations.
    * `BATCH_SIZE`: batch size of observations used for training. At any time step, this is the number of observations that are sampled from the Experience Replay buffer, to be used for training the agent.
    * `GAMMA`: discount factor used to depreciate future rewards. A low value represents a short-sighted agent that only cares about immediate rewards.
    * `TAU`: the weight used to periodically update the target Q-network weights with a weighted combination of its weights and those of the local Q-network. Low values of this prefer the existing target Q-network weights over those of the local Q-network.
    * `LR`: learning rate for gradient based optimization.
    * `UPDATE_EVERY`: the frequency at which observations are sampled from the Experience Replay buffer and training is performed. 
 
## Outcome

* As shown in `Navigation.ipynb`, the environment was solved in 511 episodes; a plot of the scores obtained is also provided in it. Learning was continued to 2000 episodes; weights were only saved when a higher average score (over 100 episodes) than the current best average score, was obtained. A good set of Q-function weights are provided in `saved/checkpoint.pth`. Three test runs provided average agent scores of 23, 16 and 17. An animation of a trained agent is provided below. In comparison with the untrained agent observed earlier, this agent seeks out yellow banannas and clearly demonstrates the acquisition of this skill.

![Trained agent seeking yellow banannnas](saved/trained_agent.gif)

## Future work

This code will be extended in the near future along multiple directions
1. Instead of the agent state being represented by a 37-dimensional vector, images of what the agent observes may be used to represent the agent state. In this case, the agent will learn directly from pixels. The expectation is to use a Convolution Neural Network representation of the Q-function in `model.py` to enable this.
2. The current code is based on the concept of Experience Replay; this gives all experience exemplars equal likelihood of selection during training. An Experience Replay scheme that favors the selection of more informative exemplars for training, as suggested in Schaul et al, "Prioritized Experience Replay", arXiv:1511.05952, will be pursued in the next iteration of this code.
3. The algorithm used to train the agent is the deep Q-learning algorithm proposed in Mnih et al, "Human-level control through deep reinforcement learning", Nature, Vol 518, Feb 2015. This basically learns weights to minimize the difference between the action values of the target Q-function and the local Q-function. Particularly in the early stages when the environment is relatively unexplored, the target has a propensity to overestimate the action values. To prevent this, van Hasselt et al propose "Deep Reinforcement Learning with Double Q-learning" in arXiv:1509.06461. Another interesting direction of future work is that of "Duelling Network architectures for Deep Reinforcement Learning", arXiv:1511.06581 by Wang et al. This attempts to learn a state-value function and an action-advantage function in parallel. Their combination provides the action value function for every state-action pair. The state-value function does not vary much between actions and hence can be obtained independently. 
