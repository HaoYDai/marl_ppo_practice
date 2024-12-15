import os.path
import time

import gym
import numpy as np
import torch

from ppo_agent import PPOAgent

scenorio = "Pendulum-v1"
env = gym.make(scenorio, render_mode="human")

# Directory for saving models
os.makedirs("models", exist_ok=True)
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + "/models/"
timestamp = time.strftime("%Y%m%d%H%M%S")

NUM_EPISODE = 5000
NUM_STEP = 500
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BATCH_SIZE = 128
UPDATE_INTERVAL = 32

agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
best_reward = -2000

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    done = False
    episode_reward = 0

    for step_i in range(NUM_STEP):
        env.render()
        action, value = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        done = True if (step_i +1) == NUM_STEP else False
        agent.replay_buffer.add_memo(state, action, reward, value, done)
        state = next_state

        if (step_i + 1) % UPDATE_INTERVAL == 0 or (step_i + 1) == NUM_STEP:
            agent.update()

    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()
        torch.save(agent.actor.state_dict(), model + f"ppo_actor_{timestamp}.pth")
        print(f"Best reward: {best_reward}")

    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i}, Reward: {round(episode_reward, 2)}")

    if episode_i % 100 == 0:
        eval_rewards = []
        for _ in range(5):  # 评估5次
            state, _ = env.reset()
            episode_reward = 0
            for _ in range(NUM_STEP):
                action, _ = agent.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            eval_rewards.append(episode_reward)
        avg_reward = np.mean(eval_rewards)
        print(f"Evaluation at episode {episode_i}: Average Reward = {avg_reward}")

env.close()
