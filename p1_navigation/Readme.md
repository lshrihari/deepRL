# Deep RL projects

## Environment
* An agent navigates a square world environment. 
* The environment has blue and yellow banannas dispersed in it.
* The objective of the agent is to collect (cross-over) as many yellow banannas as possible while avoiding blue banannas. 
* Collecting a yellow bananna results in a reward of +1; picking a blue bananna results in a reward of -1
* The state-space of the agent is a 37 dimensional vector comprising its velocity and a ray-based perception of the environment in front of the agent.
* The action-space of the agent is 4 dimensional comprising the actions of moving forward, backward, left and right, respectively represented by 0, 1, 2 and 3.
* The agent is said to have solved the environment if it gets an average score of +13 in 100 consecutive episodes.

## Getting started
* A Python distribution (with `python3`) is assumed present in the system.
* Install packages `pytorch` (CPU version is sufficient; install the GPU version if one is available and suitable CUDA/CUDNN dependencies are already installed) and all its dependencies using `conda` or `pip3` . Package `numpy` should have automatically been installed; if not, install that as well.
* Install Unity ML-agents as per the instructions provided at [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) . Python packages `tensorflow` and `jupyter` will also be installed.
* Install the "Bananna world" environment for this project for [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip), [MAC OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip), [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip) or [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) in accordance with the OS being used to run this project.
* Run the command `jupyter notebook` in the command-line. Open the file `Test_Environment.ipynb` and step through each step of the code. If execution is successful and an untrained agent like the figure below is observed, the setup is successful and training can commence.

![Untrained agent performing random actions](saved/untrained_agent.gif)

## Running the code
* From the Jupyter interface, open file `Navigation.ipynb` and step through each line of the code.
* In the last step, the agent loads pre-trained weights which are saved in `saved/checkpoint.pth` and agent-testing uses these for testing.  If training is done afresh, change `torch.load('saved/checkpoint.pth')` to `torch.load('checkpoint.pth')` in the last code block to load the trained weights.