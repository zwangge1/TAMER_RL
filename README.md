# TAMER

This project implements a hybrid learning framework combining TAMER (Training an Agent Manually via Evaluative Reinforcement) and Q-learning. The goal is to explore how human feedback and environmental rewards can be dynamically integrated to train agents more effectively and robustly.

In addition, we have redesigned the interface to provide more diverse and intuitive options that align with subconscious human behavior patterns. 


# Envoirment setting
You need python 3.7+ with numpy, sklearn, pygame and gym.

# Run This File

1. After setting up the environment, execute the `run.py` file.

2. First, you will go through 30 training iterations using the default `CartPole-v1` environment. During the process, two windows will appear:  
   - The first window displays the `CartPole-v1` environment.  
   - The second window shows the decisions made by the Agent.  
     If you believe the Agent applied the correct force in the correct direction, **left-click** on the second window. It will turn green, indicating that you support the Agent's decision. Conversely, if you **right-click**, it means you disagree with the decision, and the Agent will attempt to adjust its behavior in the next iteration.

3. The first 30 training iterations follow the TAMER method. After this, the system will conduct 15 self-evaluation runs. Next, it will automatically start training using the TAMER+RL method for another 30 iterations, followed by another 15 self-evaluation runs. Finally, you will receive a result curve that allows you to evaluate and compare the training process and evaluation results of the two methods.

4. **Note:** In the `CartPole-v1` environment, the `Left` action is correct, while the `None` action is equivalent to `Right`. This is because I didn’t modify the action label settings for the `MountainCar-v0` environment (feel free to help me fix this if needed).

![resP2](https://github.com/user-attachments/assets/918f8525-c815-4d15-8504-f25f98e2cc4c)
