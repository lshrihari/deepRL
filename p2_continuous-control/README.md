[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control

## Introduction

This project uses Deep Reinforcement Learning to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The environment involves a double jointed arm tracking a ball (target) as closely as possible. Specifically, the environment used in this project involves 20 such arms and the objective of this project is to learn a controller for these arms or agents.

![Trained Agent][image1]

## Environment Details

* State - 33 dimensional vector comprising position, rotation, velocity and angular velocities of each arm.
* Action - 4 dimensional vector encoding torques for each joint; each value ranges between -1 and 1.
* Reward - The agent receives a reward of +0.1 at each time step it is at the target location. The goal is thus to maximally maintain the arm at the target location.
* The environment is said to have been solved when the aggregate reward across te 20 agents is >= 30 for 100 consecutive episodes.

## Setup
* Clone this repository
* Download and unzip the environment in this folder, as per the platform in use.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Running the code
* A successful training outcome is shown in "Continuous_Control_Reacher20.ipynb"
* Model weights and the scores obtained for the successful training, are stored in the "saved" folder.
* Step through each line of the file "Continuous_Control_Reacher20.ipynb" to train the agents from scratch.