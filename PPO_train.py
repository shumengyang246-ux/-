"""
PPO (Proximal Policy Optimization) 算法实现
用于训练智能体玩 LunarLander-v2 游戏

Actor网络：输入状态，输出动作概率分布
Critic网络：输入状态，输出状态价值
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import os

# ---------------- 超参数配置 ----------------
class PPOConfig:
    def __init__(self):
        # 环境参数
        self.ENV_NAME = "LunarLander-v3"
        self.STATE_DIM = 8
        self.ACTION_DIM = 4
        
        # PPO算法参数 
        self.LEARNING_RATE = 3e-4  # 提高学习率
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIP_EPSILON = 0.2
        self.ENTROPY_COEF = 0.01  # 降低熵系数
        self.MAX_GRAD_NORM = 0.5
        
        # 训练参数 
        self.NUM_EPISODES = 600      #一次训练的回合数
        self.STEPS_PER_UPDATE = 2048  # 每次更新收集的步数
        self.MINI_BATCH_SIZE = 64     # 小批次大小
        self.PPO_EPOCHS = 5         # PPO更新轮数
        self.MAX_STEPS_PER_EPISODE = 1000
        
        # 评估和保存
        self.EVALUATE_EVERY = 100
        self.SAVE_EVERY = 500
        self.MODEL_DIR = "ppo_models"
        
        # 随机种子和设备
        self.SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 神经网络结构 ----------------
class ActorNetwork(nn.Module):
    """Actor网络：输入状态，输出动作概率分布"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()
    
    def _init_weights(self):
        """使用正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回动作概率分布"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """根据状态选择动作，返回动作和对应的log概率"""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定状态下的动作，返回log概率和熵"""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_prob, entropy

class CriticNetwork(nn.Module):
    """Critic网络：输入状态，输出状态价值"""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self._init_weights()
    
    def _init_weights(self):
        """使用正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回状态价值"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value

# ---------------- 经验回放缓冲区 ----------------
class RolloutBuffer:
    """改进的经验缓冲区"""
    def __init__(self, buffer_size: int, state_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        # 使用列表存储，更灵活
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.ptr = 0
    
    def store(self, state, action, reward, done, log_prob, value):
        """存储一步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """计算回报和优势函数 """
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        # 计算GAE
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        for t in reversed(range(len(rewards))): # 关键：数组逆序
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            # GAE
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae # 广义优势估计的递推公式
            advantages[t] = last_gae
        # 回报 = 优势 + 价值
        returns = advantages + values
        return returns, advantages
    
    def get(self):
        """获取所有数据"""
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int64),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32)
        }
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.ptr = 0

# ---------------- PPO智能体 ----------------
class PPOAgent:
    """PPO智能体 """
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # 创建网络
        self.actor = ActorNetwork(config.STATE_DIM, config.ACTION_DIM).to(config.DEVICE)
        self.critic = CriticNetwork(config.STATE_DIM).to(config.DEVICE)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE)
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """根据当前状态选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.DEVICE)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor).item()
        
        return action, log_prob.item(), value
    
    def save_model(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.config.DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"模型已加载: {filepath}")

# ---------------- 训练函数 ----------------
def update_policy(agent: PPOAgent, buffer: RolloutBuffer, config: PPOConfig, last_state: np.ndarray):
    """更新策略网络"""
    # 计算最后状态的价值
    with torch.no_grad():
        last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(config.DEVICE)
        last_value = agent.critic(last_state_tensor).item()
    
    # 计算回报和优势
    returns, advantages = buffer.compute_returns_and_advantages(
        last_value, config.GAMMA, config.GAE_LAMBDA
    )
    
    # 转换为tensor
    data = buffer.get()
    states = torch.FloatTensor(data['states']).to(config.DEVICE)
    actions = torch.LongTensor(data['actions']).to(config.DEVICE)
    old_log_probs = torch.FloatTensor(data['log_probs']).to(config.DEVICE)
    returns_tensor = torch.FloatTensor(returns).to(config.DEVICE)
    advantages_tensor = torch.FloatTensor(advantages).to(config.DEVICE)
    
    # 标准化优势
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    # PPO更新
    dataset_size = len(states)
    
    for epoch in range(config.PPO_EPOCHS):
        # 随机打乱数据
        indices = np.random.permutation(dataset_size)
        
        # 小批次训练
        for start in range(0, dataset_size, config.MINI_BATCH_SIZE):
            end = min(start + config.MINI_BATCH_SIZE, dataset_size)
            batch_indices = indices[start:end]
            
            # 批次数据
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            
            # 计算新的log概率和价值
            new_log_probs, entropy = agent.actor.evaluate_action(batch_states, batch_actions)
            new_values = agent.critic(batch_states).squeeze()
            
            # PPO损失
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config.CLIP_EPSILON, 1 + config.CLIP_EPSILON) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - config.ENTROPY_COEF * entropy.mean()
            
            # 价值损失
            critic_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
            
            # 更新Actor网络
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.MAX_GRAD_NORM)
            agent.actor_optimizer.step()
            
            # 更新Critic网络
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), config.MAX_GRAD_NORM)
            agent.critic_optimizer.step()

def train_ppo(config: PPOConfig):
    """改进的PPO训练函数"""
    # 创建环境和智能体
    env = gym.make(config.ENV_NAME)
    agent = PPOAgent(config)
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    
    # 缓冲区
    buffer = RolloutBuffer(config.STEPS_PER_UPDATE, config.STATE_DIM, config.DEVICE)
    
    print(f"开始训练PPO智能体")
    print(f"环境: {config.ENV_NAME}")
    print(f"设备: {config.DEVICE}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"总回合数: {config.NUM_EPISODES}")
    print("=" * 60)
    
    state, _ = env.reset(seed=config.SEED)
    episode_reward = 0
    episode_length = 0
    episode = 0
    
    max_steps = config.NUM_EPISODES * config.MAX_STEPS_PER_EPISODE
    
    for total_steps in range(max_steps):
        # 选择动作
        action, log_prob, value = agent.get_action(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        buffer.store(state, action, reward, done, log_prob, value)
        
        episode_reward += reward
        episode_length += 1
        state = next_state
        
        # Episode结束
        if done or episode_length >= config.MAX_STEPS_PER_EPISODE:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode += 1
            
            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:6.2f} | "
                      f"Total Steps: {total_steps:7d}")
            
            # 重置环境
            state, _ = env.reset(seed=config.SEED + episode)
            episode_reward = 0
            episode_length = 0
            
            # 检查是否达到目标回合数
            if episode >= config.NUM_EPISODES:
                break
        
        # 更新策略
        if buffer.ptr >= config.STEPS_PER_UPDATE:
            update_policy(agent, buffer, config, state)
            buffer.clear()
        
        # 评估和保存
        if episode > 0 and episode % config.EVALUATE_EVERY == 0 and episode_length == 0:
            avg_reward = np.mean(episode_rewards[-config.EVALUATE_EVERY:])
            print(f"\n{'='*60}")
            print(f"评估 - Episode {episode}: 平均奖励 = {avg_reward:.2f}")
            print(f"{'='*60}\n")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                model_path = os.path.join(config.MODEL_DIR, f"ppo_best_episode_{episode}.pth")
                agent.save_model(model_path)
        
        # 定期保存
        if episode > 0 and episode % config.SAVE_EVERY == 0 and episode_length == 0:
            model_path = os.path.join(config.MODEL_DIR, f"ppo_checkpoint_episode_{episode}.pth")
            agent.save_model(model_path)
    
    env.close()
    
    # 绘制训练曲线
    plot_training_curves(episode_rewards, episode_lengths, config)
    
    return episode_rewards, episode_lengths

def plot_training_curves(rewards: List[float], lengths: List[float], config: PPOConfig):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 奖励曲线
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    
    # 平滑奖励曲线
    window_size = 50
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size, len(rewards) + 1), smoothed_rewards, 
                color='red', linewidth=2, label='Smoothed (50 episodes)')
    
    ax1.axhline(y=200, color='green', linestyle='--', label='Success Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('PPO Training - Episode Rewards')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 长度曲线
    ax2.plot(episodes, lengths, alpha=0.3, color='green', label='Raw')
    
    # 平滑长度曲线
    if len(lengths) >= window_size:
        smoothed_lengths = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size, len(lengths) + 1), smoothed_lengths, 
                color='orange', linewidth=2, label='Smoothed (50 episodes)')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('PPO Training - Episode Lengths')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('ppo_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存到: ppo_training_curves.png")
    plt.show()

# ---------------- 测试函数 ----------------
def test_agent(config: PPOConfig, model_path: str, num_episodes: int = 10):
    """测试训练好的智能体"""
    env = gym.make(config.ENV_NAME, render_mode="human")
    agent = PPOAgent(config)
    agent.load_model(model_path)
    
    total_rewards = []
    
    print(f"\n{'='*60}")
    print(f"测试智能体 - 模型: {model_path}")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=config.SEED + episode)
        episode_reward = 0
        episode_length = 0
        
        while episode_length < config.MAX_STEPS_PER_EPISODE:
            # 选择动作（贪婪策略）
            action, _, _ = agent.get_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:4d}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\n{'='*60}")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"{'='*60}\n")
    
    return total_rewards

# ---------------- 主函数 ----------------
if __name__ == "__main__":
    # 创建配置
    config = PPOConfig()
    
    # 设置随机种子
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    # 训练智能体
    print("="*60)
    print("PPO强化学习训练 - LunarLander-v3")
    print("="*60)
    rewards, lengths = train_ppo(config)
    
    # 测试最佳模型
    print("\n训练完成！")
    model_files = [f for f in os.listdir(config.MODEL_DIR) if f.startswith("ppo_best")]
    if model_files:
        # 按episode数字排序，取最新的
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = os.path.join(config.MODEL_DIR, model_files[-1])
        print(f"\n开始测试最佳模型...")
        test_rewards = test_agent(config, best_model_path, num_episodes=5)
    
    print("\n"+"="*60)
    print("PPO训练完成！")
    print("="*60)