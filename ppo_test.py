import gym
import torch
from ppo_agent import PPOAgent

# 创建环境
env = gym.make("Pendulum-v1", render_mode="human")  # 添加 render_mode="human" 来显示动画
state_dim = env.observation_space.shape[0] 
action_dim = env.action_space.shape[0]

# 创建 agent 并加载训练好的模型
agent = PPOAgent(state_dim, action_dim, batch_size=128)
agent.actor.load_state_dict(torch.load("models/ppo_actor_20241215221002.pth"))  # 替换实际的时间戳

# 运行测试回合
episodes = 5  # 测试回合数
for ep in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        env.render()  # 显示动画
        action, _ = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        state = next_state
        
        if done:
            print(f"Episode {ep}: Total Reward = {episode_reward}")
            break

env.close()
