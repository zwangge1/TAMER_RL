import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')
HUMAN_FEEDBACK_WEIGHT = 0.2

import matplotlib.pyplot as plt
import cv2

class SGDFunctionApproximator:
    def __init__(self, env):
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))
        self.models = [
            SGDRegressor(learning_rate='constant').partial_fit(
                [self.featurize_state(env.reset()[0])], [0]
            )
            for _ in range(env.action_space.n)
        ]

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if action is None:
            return [m.predict([features])[0] for m in self.models]
        return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Tamer:
    def __init__(self, env, num_episodes, discount_factor=1, epsilon=0,
                 min_eps=0, tame=True, ts_len=0.3, output_dir=LOGS_DIR,mode="TAMER"):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0
        self.mode = mode

        self.H = SGDFunctionApproximator(env)  # init H
        if mode == "TAMER+RL" or not tame:
            self.Q = SGDFunctionApproximator(env)  # init Q

        self.reward_log_columns = ['Episode', 'Ep start ts', 'Feedback ts',
                                   'Human Reward', 'Environment Reward']
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        if np.random.random() < 1 - self.epsilon:
            if self.mode == "TAMER":
                preds = self.H.predict(state)
            elif self.mode == "TAMER+RL":
                preds = self.Q.predict(state)
            else:
                preds = self.Q.predict(state)
            return np.argmax(preds)
        return np.random.randint(0, self.env.action_space.n)



    def _train_episode(self, episode_index, disp, mode):
        print(f"Episode: {episode_index + 1}")
        state, _ = self.env.reset()
        tot_reward = 0
        step_count = 0

        for ts in count():
            step_count += 1

            # 渲染 Gym 界面
            frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow('OpenAI Gym - Training', frame_bgr)
            if cv2.waitKey(25) & 0xFF == 27:  # 按 ESC 退出
                break

            action = self.act(state)
            if mode != "Q-learning":
                disp.show_action(action)

            next_state, reward, done, *_ = self.env.step(action)

            # get Human feedback
            human_reward = 0
            if mode != "Q-learning":
                now = time.time()
                while time.time() < now + self.ts_len:
                    time.sleep(0.01)
                    human_reward = disp.get_scalar_feedback()
                    if human_reward != 0:
                        if human_reward > 0:
                            self.positive_feedback_count += 1
                        elif human_reward < 0:
                            self.negative_feedback_count += 1
                        break

            # Update the reward depend on mode
            if mode == "TAMER" and human_reward != 0:
                td_target = human_reward
                self.H.update(state, action, td_target)

            elif mode == "TAMER+RL":
                # count reward numbers，used for dynamic adjustment
                total_feedback = self.positive_feedback_count + self.negative_feedback_count
                human_reward_weight = 0.9 if total_feedback < 10 else 0.6
                env_reward_weight = 1 - human_reward_weight

                if human_reward != 0:
                    self.H.update(state, action, human_reward)
                
                # count Q value
                q_value_next = np.max(self.Q.predict(next_state))
                td_target = reward + self.discount_factor * q_value_next
                
                # combine H and Q
                combined_reward = human_reward_weight * human_reward + env_reward_weight * td_target
                self.Q.update(state, action, combined_reward)

            tot_reward += reward
            if done or step_count > 1000:
                break
            state = next_state

        print(f"Episode {episode_index + 1} completed. Total Reward: {tot_reward}")
        return tot_reward  # return total reward


    async def train(self, mode, disp=None):
        train_rewards = []  # record reward of each episode 
        for i in range(self.num_episodes):
            ep_reward = self._train_episode(i, disp, mode)
            train_rewards.append(ep_reward)  # Record Cumulative Rewards
        return train_rewards  # Back to the list of training awards


    def evaluate(self, n_episodes=30, max_steps=200):
        """
        Evaluate agent performance and add an early stop mechanism.
        Args.
            n_episodes (int): number of episodes to evaluate.
            max_steps (int): Maximum number of steps per episode.
        
        Returns: list: accumulated rewards for each episode.
            list: Cumulative rewards for each episode.
        """
        print('Evaluating agent')
        self.epsilon = 0  # # Ensure that there is no exploratory behavior during the assessment phase
        rewards = []
        
        for episode in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            step_count = 0 

            while not done:
                # 渲染 Gym 界面
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('OpenAI Gym - Evaluation', frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    done = True

                action = self.act(state) 
                next_state, reward, done, *_ = self.env.step(action)
                tot_reward += reward
                state = next_state
                step_count += 1

                # Early termination and punitive rewards if steps exceed thresholds
                if step_count >= max_steps:
                    print(f"Episode {episode + 1}: Early stopping after {step_count} steps.")
                    tot_reward = -500
                    break

            rewards.append(tot_reward)

        cv2.destroyAllWindows()
        avg_reward = np.mean(rewards)
        print(f'Average total episode reward over {n_episodes} episodes: {avg_reward:.2f}')
        return rewards

