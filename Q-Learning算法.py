# Q-Learning算法代码实现：以python中的gym库为例

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from typing import NamedTuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 参数 ----------------
class Config(NamedTuple):
    TOTAL_EPISODES: int = 5_000  # 训练回合数
    LR: float = 0.1              # 学习率
    GAMMA: float = 0.95          # 折扣因子
    EPSILON_MIN: float = 0.01    # 最小探索率
    EPSILON_MAX: float = 1.0      # 最大探索率
    EPSILON_DECAY: int = 4_000   # 衰减步数
    MAP_SIZE: int = 6        # 地图大小
    SEED: int = 234            # 随机种子
    IS_SLIPPERY: bool = False  
    N_RUNS: int = 5              # 运行次数
    DIM_OF_ACTION: int = 4      # 动作维度
    DIM_OF_OBSERVATION: int = MAP_SIZE * MAP_SIZE  # 状态维度
    RNG = np.random.default_rng(SEED)  # 随机数生成器

# 在这里定义全局 cfg
cfg = Config()

# ---------------- 算法 ----------------
class Algorithm:
    def __init__(self):
        self.state_dim  = cfg.DIM_OF_OBSERVATION
        self.action_dim = cfg.DIM_OF_ACTION
        self.epsilon    = cfg.EPSILON_MAX
        self.lr         = cfg.LR
        self.gamma      = cfg.GAMMA
        self.q_table    = cfg.RNG.uniform(-0.1, 0.1, (self.state_dim, self.action_dim)) # 均匀分布

    def reset_q_table(self):
        self.q_table = cfg.RNG.uniform(-0.1, 0.1, (self.state_dim, self.action_dim)) # 使用均匀分布的随机值初始化Q表

    def predict(self, state: int, exploit: bool = False) -> int:
        if not exploit and cfg.RNG.uniform(0, 1) < self.epsilon:
            return cfg.RNG.integers(self.action_dim)

        max_val = self.q_table[state].max()
        best = np.where(self.q_table[state] == max_val)[0] # 找到最大Q值的动作
        return cfg.RNG.choice(best)

    def learn(self, data):
        s, a, r, s_next, done = data
        # 修改原来稀疏的奖励信号：到达目标给予更大奖励
        if done:
            if s_next == cfg.MAP_SIZE * cfg.MAP_SIZE - 1:  # 到达目标
                r = 10
            else:  # 掉入洞中
                r = -5
        else:
            r = -0.01  # 每步小惩罚，鼓励尽快到达目标
        
        target = r + (1 - done) * self.gamma * self.q_table[s_next].max()
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])


# ---------------- 智能体 ----------------
class Agent:
    def __init__(self):
        self.algo = Algorithm()

    def predict(self, state, exploit=False):
        return self.algo.predict(state, exploit)

    def learn(self, transition):
        self.algo.learn(transition)

    # -------- 可视化 --------
    def plot_policy(self, env):
        # 获取原始环境的desc
        unwrapped_env = env.unwrapped
        desc = unwrapped_env.desc if hasattr(unwrapped_env, 'desc') else None
        
        q_max = self.algo.q_table.max(axis=1).reshape(cfg.MAP_SIZE, cfg.MAP_SIZE)
        best  = self.algo.q_table.argmax(axis=1).reshape(cfg.MAP_SIZE, cfg.MAP_SIZE)
        arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        annot = np.full(best.size, "", dtype="<U1")
        for i, a in enumerate(best.flat):
            annot[i] = arrows[a] if q_max.flat[i] > -1e6 else ""  # 修改阈值
        
        annot = annot.reshape(cfg.MAP_SIZE, cfg.MAP_SIZE)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        # 显示地图
        if desc is not None:
            # 创建地图可视化
            map_vis = np.zeros((cfg.MAP_SIZE, cfg.MAP_SIZE))
            for i in range(cfg.MAP_SIZE):
                for j in range(cfg.MAP_SIZE):
                    if desc[i, j] == b'S':  # 起点
                        map_vis[i, j] = 0.3
                    elif desc[i, j] == b'F':  # 冰面
                        map_vis[i, j] = 0.6
                    elif desc[i, j] == b'H':  # 洞
                        map_vis[i, j] = 0.0
                    elif desc[i, j] == b'G':  # 目标
                        map_vis[i, j] = 1.0
            
            sns.heatmap(map_vis, annot=False, cmap="viridis", 
                       linewidths=.7, linecolor="black", ax=ax[0],
                       xticklabels=[], yticklabels=[])
            ax[0].set_title("Map (S:Start, F:Frozen, H:Hole, G:Goal)")
        else:
            ax[0].text(0.5, 0.5, "Map not available", 
                      ha='center', va='center', transform=ax[0].transAxes)
        ax[0].axis("off")
        
        # 策略箭头
        sns.heatmap(q_max, annot=annot, fmt="s", cmap="RdYlBu",
                    linewidths=.7, linecolor="black", ax=ax[1],
                    xticklabels=[], yticklabels=[])
        ax[1].set_title("Learned policy")
        
        # Q 值
        sns.heatmap(q_max, annot=np.round(q_max, 2), fmt=".2f", cmap="RdYlBu",
                    linewidths=.7, linecolor="black", ax=ax[2],
                    xticklabels=[], yticklabels=[], cbar=False)
        ax[2].set_title("Q values")
        plt.tight_layout()
        plt.show()


# ---------------- 训练 ----------------
def run_episodes(env, agent, n_episodes, run=0, progress=True):
    rewards = np.zeros(n_episodes)
    steps   = np.zeros(n_episodes)
    success_rate = np.zeros(n_episodes)
    
    it = tqdm(range(n_episodes), desc=f"Run {run}") if progress else range(n_episodes)
    
    for ep in it:
        s, _ = env.reset(seed=cfg.SEED + ep + run * n_episodes)
        done = False
        total_r = 0
        step = 0
        
        while not done and step < 100:  # 添加步数限制
            a = agent.predict(s, exploit=False)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            # 学习
            agent.learn((s, a, r, s_next, done))
            total_r += r
            step += 1
            s = s_next
            
        rewards[ep] = total_r
        steps[ep]   = step
        success_rate[ep] = 1 if total_r > 0 else 0  # 成功到达目标
        
        # epsilon 指数衰减
        agent.algo.epsilon = cfg.EPSILON_MIN + (cfg.EPSILON_MAX - cfg.EPSILON_MIN) * \
                            np.exp(-1. * ep / (cfg.EPSILON_DECAY / 4))
    
    return rewards, steps, success_rate


def train_workflow(env, agent):
    all_rewards = np.zeros((cfg.N_RUNS, cfg.TOTAL_EPISODES))
    all_steps   = np.zeros((cfg.N_RUNS, cfg.TOTAL_EPISODES))
    all_success = np.zeros((cfg.N_RUNS, cfg.TOTAL_EPISODES))
    
    for run in range(cfg.N_RUNS):
        agent.algo.reset_q_table()
        agent.algo.epsilon = cfg.EPSILON_MAX
        rews, sts, success = run_episodes(env, agent, cfg.TOTAL_EPISODES, run=run, progress=True)
        all_rewards[run] = rews
        all_steps[run]   = sts
        all_success[run] = success
        
        # 计算最后100个episode的平均表现
        last_100_avg_reward = rews[-100:].mean()
        last_100_success_rate = success[-100:].mean()
        
        print(f"Run {run:2d}:  "
              f"final_avg_reward={last_100_avg_reward:5.2f}  "
              f"success_rate={last_100_success_rate:.2f}  "
              f"avg_steps={sts.mean():5.2f}")
    
    # 可视化最终策略
    agent.plot_policy(env)
    
    # 绘制训练曲线
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # 平滑函数
    def smooth(data, window=100):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 奖励曲线
    mean_rewards = all_rewards.mean(axis=0)
    smoothed_rewards = smooth(mean_rewards)
    ax1.plot(smoothed_rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Progress - Rewards')
    ax1.grid(True)
    
    # 成功率曲线
    mean_success = all_success.mean(axis=0)
    smoothed_success = smooth(mean_success)
    ax2.plot(smoothed_success)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Training Progress - Success Rate')
    ax2.grid(True)
    
    # 步数曲线
    mean_steps = all_steps.mean(axis=0)
    smoothed_steps = smooth(mean_steps)
    ax3.plot(smoothed_steps)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Steps')
    ax3.set_title('Training Progress - Steps')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


# ---------------- main ----------------
if __name__ == "__main__":
    # 使用全局的 cfg
    random_map = generate_random_map(size=cfg.MAP_SIZE, p=0.9, seed=cfg.SEED)

    env = gym.make("FrozenLake-v1",
                   desc=random_map,
                   is_slippery=cfg.IS_SLIPPERY)
                   #render_mode="rgb_array")
    
    print(f"地图大小: {cfg.MAP_SIZE}x{cfg.MAP_SIZE}")
    print(f"地图描述:\n{random_map}")
    
    agent = Agent()
    
    try:
        train_workflow(env, agent)
    finally:
        env.close()