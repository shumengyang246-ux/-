# DQN on CartPole-v1: 快速收敛 + 训练曲线 + 录制GIF演示

import os
import random
from collections import namedtuple, deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import gymnasium as gym

try:
    import imageio  # 用于保存GIF
except Exception:
    imageio = None


# ---------辅助函数---------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(x: List[float], window: int = 50) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    w = min(window, len(x))
    c = np.cumsum(np.insert(np.array(x, dtype=np.float32), 0, 0.0))
    return (c[w:] - c[:-w]) / w


# ---------超参数配置---------
@dataclass
class Config:
    # 更简单的环境：CartPole
    ENV_NAME: str = "CartPole-v1"

    # DQN超参（针对CartPole调到更容易在“适当回合/步数”内收敛）
    LEARNING_RATE: float = 1e-3
    GAMMA: float = 0.99
    BUFFER_CAPACITY: int = 50_000
    MIN_REPLAY_SIZE: int = 500
    BATCH_SIZE: int = 64

    # Exploration：更快衰减，尽快进入“利用为主”看到收敛
    EPS_START: float = 1.0
    EPS_END: float = 0.05
    EPS_DECAY_STEPS: int = 6000  # 比你原来 50k 更快:contentReference[oaicite:3]{index=3}

    # Training
    NUM_EPISODES: int = 800
    MAX_STEPS_PER_EPISODE: int = 500  # CartPole-v1每回合上限通常500
    TRAIN_EVERY_STEPS: int = 1
    TARGET_UPDATE_EVERY_STEPS: int = 500
    GRAD_CLIP_NORM: float = 10.0

    # 提前停止（明显收敛）
    SOLVED_MA_WINDOW: int = 50
    SOLVED_RETURN_MA: float = 475.0  # CartPole-v1 接近满分（500）

    # Demo录制
    DEMO_EPISODES: int = 1
    DEMO_GIF_PATH: str = "cartpole_demo.gif"
    DEMO_MAX_STEPS: int = 500
    DEMO_FPS: int = 30

    # Misc
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR: str = "dqn_models"
    SAVE_EVERY_EPISODES: int = 100


cfg = Config()


# ---------神经网络定义---------
class Network(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---------经验回放缓冲区---------
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ---------智能体/算法---------
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device(cfg.DEVICE)

        self.q_net = Network(state_dim, action_dim).to(self.device)
        self.target_net = Network(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.LEARNING_RATE)
        self.memory = ReplayBuffer(cfg.BUFFER_CAPACITY)
        self.total_steps = 0
        self.action_dim = action_dim

    def epsilon_by_step(self, step: int) -> float:
        frac = min(step / cfg.EPS_DECAY_STEPS, 1.0)
        return cfg.EPS_START + frac * (cfg.EPS_END - cfg.EPS_START)

    def select_action(self, state_np: np.ndarray, greedy: bool = False) -> int:
        if not greedy:
            eps = self.epsilon_by_step(self.total_steps)
            if random.random() < eps:
                return random.randrange(self.action_dim)

        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> float:
        if len(self.memory) < cfg.MIN_REPLAY_SIZE:
            return 0.0
        if len(self.memory) < cfg.BATCH_SIZE:
            return 0.0

        batch = self.memory.sample(cfg.BATCH_SIZE)

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.int64, device=self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(batch.done), dtype=torch.float32, device=self.device)

        q_sa = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1)[0]
            target = rewards + cfg.GAMMA * (1.0 - dones) * q_next

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.GRAD_CLIP_NORM)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )


# ---------训练与演示---------
def make_env(render_mode: Optional[str] = None):
    env = gym.make(cfg.ENV_NAME, render_mode=render_mode)
    return env

def train():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    set_seed(cfg.SEED)

    env = make_env(render_mode=None)
    obs, info = env.reset(seed=cfg.SEED)
    env.action_space.seed(cfg.SEED)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    episode_returns: List[float] = []
    losses: List[float] = []

    for ep in range(1, cfg.NUM_EPISODES + 1):
        obs, info = env.reset(seed=cfg.SEED + ep)
        ep_return = 0.0
        ep_loss_accum = 0.0
        ep_updates = 0

        for t in range(cfg.MAX_STEPS_PER_EPISODE):
            agent.total_steps += 1
            action = agent.select_action(obs, greedy=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(obs, action, next_obs, float(reward), float(done))
            obs = next_obs
            ep_return += float(reward)

            if agent.total_steps % cfg.TRAIN_EVERY_STEPS == 0:
                loss = agent.update()
                if loss > 0:
                    losses.append(loss)
                    ep_loss_accum += loss
                    ep_updates += 1

            if agent.total_steps % cfg.TARGET_UPDATE_EVERY_STEPS == 0:
                agent.sync_target()

            if done:
                break

        episode_returns.append(ep_return)
        avg_ep_loss = ep_loss_accum / max(ep_updates, 1)

        if ep % 10 == 0:
            ma = moving_average(episode_returns, window=cfg.SOLVED_MA_WINDOW)
            ma_last = float(ma[-1]) if len(ma) > 0 else float("nan")
            eps_now = agent.epsilon_by_step(agent.total_steps)
            print(
                f"Episode {ep:4d} | Return {ep_return:7.2f} | "
                f"MA{cfg.SOLVED_MA_WINDOW} {ma_last:7.2f} | Eps {eps_now:5.3f} | "
                f"AvgLoss {avg_ep_loss:7.4f} | Steps {agent.total_steps}"
            )

        if ep % cfg.SAVE_EVERY_EPISODES == 0:
            ckpt_path = os.path.join(cfg.SAVE_DIR, f"dqn_cartpole_ep{ep}.pt")
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # 提前停止：看到明显收敛
        ma = moving_average(episode_returns, window=cfg.SOLVED_MA_WINDOW)
        if len(ma) > 0 and float(ma[-1]) >= cfg.SOLVED_RETURN_MA:
            print(
                f"SOLVED: MA{cfg.SOLVED_MA_WINDOW}={float(ma[-1]):.2f} "
                f">= {cfg.SOLVED_RETURN_MA} at episode {ep}."
            )
            break

    env.close()
    return agent, episode_returns, losses


def plot_curve(episode_returns: List[float], save_path: str = "dqn_cartpole_curve.png"):
    ma = moving_average(episode_returns, window=cfg.SOLVED_MA_WINDOW)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_returns) + 1), episode_returns, label="Return")
    if len(ma) > 0:
        plt.plot(range(cfg.SOLVED_MA_WINDOW, len(episode_returns) + 1), ma, label=f"MA{cfg.SOLVED_MA_WINDOW}")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN on CartPole-v1 - Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curve to: {save_path}")


def record_gif(agent: DQNAgent, gif_path: str):
    if imageio is None:
        print("imageio not installed. Run: pip install imageio")
        return

    env = make_env(render_mode="rgb_array")
    frames = []

    obs, info = env.reset(seed=cfg.SEED + 999)
    total_return = 0.0

    for t in range(cfg.DEMO_MAX_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = agent.select_action(obs, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        if terminated or truncated:
            break

    env.close()

    imageio.mimsave(gif_path, frames, fps=cfg.DEMO_FPS)
    print(f"Saved demo GIF to: {gif_path} | DemoReturn={total_return:.2f} | Frames={len(frames)}")


if __name__ == "__main__":
    agent, returns, losses = train()
    plot_curve(returns, "dqn_cartpole_curve.png")
    record_gif(agent, cfg.DEMO_GIF_PATH)
