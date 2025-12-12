"""
模型测试脚本
"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import cv2
import numpy as np
from gym.spaces import Box
from gym import RewardWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F

import time


# ======================
# 1. 配置（与训练一致）
# ======================

class SACConfig:
    # 环境相关
    frame_stack: int = 4
    frame_size_h: int = 84
    frame_size_w: int = 84

    # SAC 只需用到动作维度和设备
    device: str = "cuda"


# ======================
# 2. 环境预处理（与训练脚本一致）
# ======================

class MarioPreprocess(gym.ObservationWrapper):
    """
    灰度 + 缩放 + 帧堆叠，与训练脚本一致
    """
    def __init__(self, env, cfg: SACConfig):
        super().__init__(env)
        self.cfg = cfg
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(cfg.frame_stack, cfg.frame_size_h, cfg.frame_size_w),
            dtype=np.float32
        )
        self.frames = []

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.frames = []
        obs = self.observation(obs)
        return obs

    def observation(self, obs):
        # obs: (H,W,3) RGB
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray,
            (self.cfg.frame_size_w, self.cfg.frame_size_h),
            interpolation=cv2.INTER_AREA
        )
        normalized = resized.astype(np.float16) / 255.0
        if len(self.frames) == 0:
            self.frames = [normalized for _ in range(self.cfg.frame_stack)]
        else:
            self.frames.pop(0)
            self.frames.append(normalized)
        stacked = np.stack(self.frames, axis=0)  # (C,H,W)
        return stacked


class MarioReward(RewardWrapper):
    """
    训练时用到的奖励塑形，方便打印奖励。
    """
    def __init__(self, env, cfg: SACConfig):
        super().__init__(env)
        self.cfg = cfg
        self.prev_x = 0

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.prev_x = 0
        return obs

    def reward(self, reward):
        info = self.env.unwrapped._get_info()
        x = info.get("x_pos", self.prev_x)
        delta_x = max(0, x - self.prev_x)

        shaped = 0.01 * delta_x + 0.1 * reward
        self.prev_x = x
        # 训练时乘了 cfg.reward_scale，这里不再需要保持完全一致，
        # 因为测试只打印 reward，不参与学习。
        return shaped * 5.0  # 与训练脚本中的 reward_scale=5.0 一致


def make_env(cfg: SACConfig):
    """
    创建与训练时相同的马里奥环境：SuperMarioBros-v0 + SIMPLE_MOVEMENT + 预处理 + 奖励封装
    """
    base_env = gym_super_mario_bros.make("SuperMarioBros-v0")
    base_env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = MarioPreprocess(base_env, cfg)
    env = MarioReward(env, cfg)
    return env


# ======================
# 3. 网络结构（与训练脚本一致的 Actor）
# ======================

class MarioCNN(nn.Module):
    """
    三层卷积特征提取网络，与训练脚本一致
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x):
        x = self.conv(x)          # (B,64,7,7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))    # (B,512)
        return x


class Actor(nn.Module):
    """
    离散动作策略网络，与训练脚本一致
    """
    def __init__(self, backbone: MarioCNN, feature_dim: int, action_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, obs):
        z = self.backbone(obs)
        logits = self.head(z)              # (B, A)
        probs = F.softmax(logits, dim=-1)  # π(a|s)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample(self, obs):
        probs, log_probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()                 # (B,)
        logp_a = dist.log_prob(action)         # (B,)
        return action.unsqueeze(-1), logp_a.unsqueeze(-1), probs, log_probs


# ======================
# 4. Gym reset/step 兼容工具（与训练脚本风格一致）
# ======================

def reset_env(env):
    reset_out = env.reset()
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        next_obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        next_obs, reward, done, info = step_out
    return next_obs, reward, done, info


# ======================
# 5. 加载模型并测试若干回合
# ======================

def main():
    cfg = SACConfig()
    if not torch.cuda.is_available():
        cfg.device = "cpu"

    # 1) 创建环境
    env = make_env(cfg)
    action_dim = env.action_space.n
    print(f"Action space size: {action_dim}")
    print(f"Device: {cfg.device}")

    # 2) 构建与训练完全一致的 Actor 网络
    backbone_pi = MarioCNN(in_channels=cfg.frame_stack)
    actor = Actor(backbone_pi, feature_dim=512, action_dim=action_dim).to(cfg.device)

    # 3) 加载训练好的权重
    #请根据你的实际存储路径修改 ckpt_path
    ckpt_path = r"C:\Users\dell\Desktop\python练习\强化学习算法\SAC算法\sac_mario_results\sac_mario_final.pt"

    checkpoint = torch.load(ckpt_path, map_location=cfg.device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    print(f"Loaded trained actor from: {ckpt_path}")
    print("-" * 50)

    # 4) 运行若干回合测试
    num_episodes = 5
    max_steps_per_episode = 5000

    for ep in range(1, num_episodes + 1):
        obs = reset_env(env)
        done = False
        ep_reward = 0.0
        step_count = 0

        # 每回合单独计时
        t0 = time.time()

        while not done and step_count < max_steps_per_episode:
            env.render()

            obs_tensor = torch.as_tensor(
                obs, device=cfg.device, dtype=torch.float32
            ).unsqueeze(0)  # (1,C,H,W)

            with torch.no_grad():
                action_tensor, _, _, _ = actor.sample(obs_tensor)
                action = int(action_tensor.item())

            next_obs, reward, done, info = step_env(env, action)

            ep_reward += reward
            step_count += 1
            obs = next_obs

        t1 = time.time()
        print(f"Episode {ep}/{num_episodes} | "
              f"Steps: {step_count} | "
              f"Total Reward: {ep_reward:.2f} | "
              f"Time: {t1 - t0:.1f}s")

    env.close()
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
