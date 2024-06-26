# Deep-Q-learning-for-mobile-edge-computing

More details can be found at https://ieeexplore.ieee.org/document/9253665

To run the code, please install tensorflow 1.4.0. File train.py is the main code. File fog_env.py contains the code for mobile edge computing environment. File RL_brain.py contains the code for deep reinforcement learning.

If you use this code for research, please cite the following paper:
Ming Tang and Vincent W.S. Wong, “Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing Systems,” IEEE Transactions on Mobile Computing, 2020 (Early Access).

# Simulation Code for Results Regeneration
I have added some code for simulation and regenerating the results reported in the above mentioned paper.
The main code was forked from the following repository:

https://github.com/mingt2019/Deep-Q-learning-for-mobile-edge-computing.

I hope this code helps someone who also needs simulation codes for regenerating results in the aforementioned paper.
If you see any problem in my code, I'd be happy to be notified about it.

**Note: The first part of this README file belongs to the main project which the link was mentioned earlier.**

## Extra abstraction for queueing algorithms
I have added some abstraction for queueing algorithms for the sake of extensibilty.
The structure of this abstraction can be found in the two following files:
> - QueueHandler.py
> - GeneralQueue.py
