import asyncio
import gymnasium as gym
from tamer.agent import Tamer, MOUNTAINCAR_ACTION_MAP
import matplotlib.pyplot as plt
import numpy as np
import cv2


async def run_experiment(env_name, mode, num_episodes):
    """
    Run the experiment and return training incentives and evaluation incentives.
    """
    env = gym.make(env_name, render_mode='rgb_array')
    agent = Tamer(env, num_episodes, tame=(mode != "Q-learning"), ts_len=0.3, mode=mode)


    # 初始化 Pygame 窗口 (仅适用于 TAMER 和 TAMER+RL)
    if mode != "Q-learning":
        from tamer.interface import Interface
        disp = Interface(action_map=MOUNTAINCAR_ACTION_MAP)
    else:
        disp = None

    print(f"Starting training for {mode}...")
    train_rewards = await agent.train(mode=mode, disp=disp)  # record reward of training

    print(f"Evaluating {mode}...")
    eval_rewards = agent.evaluate(n_episodes=15)  # eval 15 times

    # close Gym and OpenCV windows
    if disp:
        import pygame
        pygame.quit()
    cv2.destroyAllWindows()

    # Calculate the average value every 3 times
    train_rewards_averaged = [np.mean(train_rewards[i:i + 3]) for i in range(0, len(train_rewards), 3)]
    eval_rewards_averaged = [np.mean(eval_rewards[i:i + 3]) for i in range(0, len(eval_rewards), 3)]

    return train_rewards_averaged, eval_rewards_averaged, agent.positive_feedback_count, agent.negative_feedback_count


async def main():
    num_episodes = 30
    
    # env_name = 'MountainCar-v0'
    env_name = 'CartPole-v1'

    # TAMER mode
    tamer_train_rewards, tamer_eval_rewards, tamer_pos, tamer_neg = await run_experiment(env_name, "TAMER", num_episodes)

    # TAMER+RL mode
    tamer_rl_train_rewards, tamer_rl_eval_rewards, tamer_rl_pos, tamer_rl_neg = await run_experiment(env_name, "TAMER+RL", num_episodes)

    # Plotting the reward curve
    plt.figure(figsize=(12, 6))

    x_train_points = range(1, len(tamer_train_rewards) + 1)
    x_eval_points = range(1, len(tamer_eval_rewards) + 1)

    # train
    plt.plot(x_train_points, tamer_train_rewards, label="TAMER Train", color="blue", linestyle="--", marker='o')
    plt.plot(x_train_points, tamer_rl_train_rewards, label="TAMER+RL Train", color="orange", linestyle="--", marker='x')

    # evaluation
    plt.plot(x_eval_points, tamer_eval_rewards, label="TAMER Eval", color="blue", marker='o')
    plt.plot(x_eval_points, tamer_rl_eval_rewards, label="TAMER+RL Eval", color="orange", marker='x')

    # Number of labeled feedbacks
    plt.annotate(f"TAMER: +{tamer_pos}, -{tamer_neg}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize=10, color="blue")
    plt.annotate(f"TAMER+RL: +{tamer_rl_pos}, -{tamer_rl_neg}", xy=(0.7, 0.85), xycoords="axes fraction", fontsize=10, color="orange")

    plt.xlabel("3-Episode Intervals")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.title("Training and Evaluation Rewards for TAMER and TAMER+RL")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    asyncio.run(main())
