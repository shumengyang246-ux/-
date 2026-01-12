# Actor-Critic算法实现CartPole平衡杆问题
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
 
# 策略网络，输入状态，输出各动作的概率分布
class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)
 
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x
 
# 定义价值网络，输入状态，输出该状态的评分
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)  # 输出的是一个评分（标量值）
 
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
 
# 定义智能体
class Agent:
    def __init__(self):
        self.gamma = 0.98  # 折扣因子
        self.lr_pi = 0.0002  # 策略网络的学习率
        self.lr_v = 0.0005  # 价值网络的学习率
        self.action_size = 2
 
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
 
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)
 
    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :])  # 增加小批量的轴
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

# 算法实现核心
    def update(self, state, action_prob, reward, next_state, done):
        # 增加小批量的轴
        # S_t
        state = torch.tensor(state[np.newaxis, :])
        # S_{t+1}
        next_state = torch.tensor(next_state[np.newaxis, :])
 
        # ①self.v 的损失：均方差
        # target = R_t + γ*V_ω(S_{t+1})
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()
        # V_ω(S_t)
        v = self.v(state)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)
 
        # ②self.pi 的损失
        # δ = R_t + γ*V_ω(S_{t+1}) - V_ω(S_t)
        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()
 
        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = Agent()
    reward_history = []
 
    for episode in range(2000): # 训练2000个回合
        state = env.reset()
        done = False
        total_reward = 0
 
        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
 
            agent.update(state, prob, reward, next_state, done)
 
            state = next_state
            total_reward += reward
 
        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, total_reward))
 
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()