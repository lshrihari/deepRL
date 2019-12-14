[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Project 3: Collaboration and Competition

## Introduction

This project uses Deep Reinforcement Learning to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. The environment involves two agents, each controlling a racket, to bounce a ball over the net. The goal of each agent is to keep the ball in play.

![Trained Agent][image1]

## Environment Details

* State - 24 dimensional vector per agent, comprising 8 variables corresponding to the position and velocity of the ball and the racket.
* Each agent receives its own local observation.
* Action - 2 dimensional vector, per agent, encoding two continuous actions - forward/backward movement and jumping of the racket.
* The task is episodic.
* Reward - The agent receives a reward of +0.1 each time it hits the ball over the net. The agent receives a reward of -0.01 if it lets the ball hit the ground or hits the ball out of bounds.
* The environment is said to have been solved when the aggregate of the maximum reward across the two agents is >= 0.5 for 100 consecutive episodes.

## Setup
* Clone this repository
* Download and unzip the environment in this folder, as per the platform in use.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


## Running the code
* A successful training outcome is shown in "Tennis.ipynb"
* Model weights and the scores obtained for the successful training, are stored in the "saved" folder.
* Step through each line of the file "Tennis.ipynb" to train the agents from scratch.