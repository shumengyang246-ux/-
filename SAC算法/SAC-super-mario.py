"""
使用强化学习的SAC算法玩超级马里奥游戏，训练网络玩超级马里奥
"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace #把马里奥游戏动作映射为0-6一共7个动作
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import numpy as np
from gym.spaces import Box
from gym import RewardWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import random  # 训练时需要设置随机种子
import json
import time
import os

#环境初始化
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 超参数配置
class SACConfig:
    # 环境
    frame_stack: int = 4
    frame_size_h: int = 84 
    frame_size_w: int = 84

    # SAC 常规超参数
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 100_000

    # 目标 V 网络 Polyak 平滑
    tau: float = 0.005

    # 训练步数
    total_steps: int = 200_000       # 总训练步数
    start_steps: int = 10_000       # 先用随机策略探索
    update_after: int = 5_000
    update_every: int = 4            # 每采一条就更新一次
    gradient_steps: int = 1          # 每次交互后做多少个 SGD step

    # SAC 特有
    alpha: float = 0.2               # 温度（如果不做自动温度）
    reward_scale: float = 5.0        # 奖励缩放因子
    device: str = "cuda"
    
    # 记录参数
    save_interval: int = 5000        # 每5000步保存一次模型
    log_interval: int = 100          # 每100步记录一次损失
    eval_interval: int = 10000       # 每1万步评估一次
    
    # 输出目录
    output_dir: str = "sac_mario_results"

# 创建输出目录
os.makedirs(SACConfig.output_dir, exist_ok=True)

#游戏环境加工处理
class MarioPreprocess(gym.ObservationWrapper):
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
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) #转换为(H,W)
        resized = cv2.resize(gray, (self.cfg.frame_size_w, self.cfg.frame_size_h),
                             interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float16) / 255.0 #归一化
        # 转为(C,H,W)
        if len(self.frames) == 0:
            self.frames = [normalized for _ in range(self.cfg.frame_stack)]
        else:
            self.frames.pop(0)
            self.frames.append(normalized)
        stacked = np.stack(self.frames, axis=0)  # (C,H,W)
        return stacked

#奖励加工处理
class MarioReward(RewardWrapper):
    def __init__(self, env, cfg: SACConfig):
        super().__init__(env)
        self.cfg = cfg
        self.prev_x = 0

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)  # 兼容 Gym 新旧 API
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.prev_x = 0
        return obs

    def reward(self, reward):
        info = self.env.unwrapped._get_info()  # 拿到 info字典
        x = info.get("x_pos", self.prev_x)
        delta_x = max(0, x - self.prev_x)

        shaped = 0.01 * delta_x + 0.1 * reward
        self.prev_x = x
        return shaped * self.cfg.reward_scale

#总环境
def make_env(env):
    env= MarioPreprocess(env, SACConfig)
    env = MarioReward(env, SACConfig)
    return env

#经验池
class ReplayBuffer:
    def __init__(self, obs_shape, action_dim, size: int, device="cpu"):
        self.size = size
        self.device = device
        self.ptr = 0 #当前指针位置
        self.full = False #是否满了

        self.obs = np.zeros((size, *obs_shape), dtype=np.float32) # 状态
        self.next_obs = np.zeros((size, *obs_shape), dtype=np.float32) # 下一状态
        self.acts = np.zeros((size, 1), dtype=np.int64) # 动作
        self.rews = np.zeros((size, 1), dtype=np.float32) # 奖励
        self.done = np.zeros((size, 1), dtype=np.float32) # 是否结束

    def store(self, obs, act, rew, next_obs, done):
        """
        用于将一次交互经验数据存储到经验池中
        """
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample_batch(self, batch_size):
        """
        抽样经验
        """
        max_idx = self.size if self.full else self.ptr # 
        idx = np.random.randint(0, max_idx, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        acts = torch.as_tensor(self.acts[idx], device=self.device)
        rews = torch.as_tensor(self.rews[idx], device=self.device)
        done = torch.as_tensor(self.done[idx], device=self.device)
        return obs, acts, rews, next_obs, done

#神经网络结构——一个 soft value 网络 Vψ(s)，一个软 Q 网络 Qθ(s,a)，我们用两个 Qθ₁、Qθ₂（double Q），一个随机策略网络 πφ(a|s)，连续动作是高斯，我们改成离散 Categorical
class MarioCNN(nn.Module):
    """
    图像特征提取，三层卷积网络
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        # 计算展平后的维度，运行时可以用 dummy forward 确定
        self.fc = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x):
        x = self.conv(x)          # (B,64,7,7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))    # (B,512)
        return x

class Actor(nn.Module):
    """
    Actor网络，使用离散策略
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
        action = dist.sample()                         # (B,)
        logp_a = dist.log_prob(action)                 # (B,)
        return action.unsqueeze(-1), logp_a.unsqueeze(-1), probs, log_probs

class QNetwork(nn.Module):
    """
    Q网络，使用两个 Q 网络，一个 Q1，一个 Q2，用于估计 Q(s,a)
    """
    def __init__(self, backbone: MarioCNN, feature_dim: int, action_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, obs):
        z = self.backbone(obs)
        q = self.head(z)      # (B, A)
        return q


class ValueNetwork(nn.Module):
    """
    V网络，使用一个 V 网络，用于估计 V(s)
    """
    def __init__(self, backbone: MarioCNN, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, obs):
        z = self.backbone(obs)
        v = self.head(z)      # (B,1)
        return v

#网络初始化
cfg = SACConfig()
if not torch.cuda.is_available():  # CUDA 不可用时自动退回 CPU
    cfg.device = "cpu"
env = make_env(env)
action_dim = env.action_space.n

backbone_q1 = MarioCNN(in_channels=cfg.frame_stack)
backbone_q2 = MarioCNN(in_channels=cfg.frame_stack)
q1 = QNetwork(backbone_q1, feature_dim=512, action_dim=action_dim).to(cfg.device)
q2 = QNetwork(backbone_q2, feature_dim=512, action_dim=action_dim).to(cfg.device)

backbone_v = MarioCNN(in_channels=cfg.frame_stack)
v = ValueNetwork(backbone_v, feature_dim=512).to(cfg.device)
v_target = ValueNetwork(MarioCNN(cfg.frame_stack), feature_dim=512).to(cfg.device)
v_target.load_state_dict(v.state_dict())
v_target.requires_grad_(False)

# 策略网络
backbone_pi = MarioCNN(in_channels=cfg.frame_stack)
actor = Actor(backbone_pi, feature_dim=512, action_dim=action_dim).to(cfg.device)

actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
q1_opt = torch.optim.Adam(q1.parameters(), lr=cfg.lr)
q2_opt = torch.optim.Adam(q2.parameters(), lr=cfg.lr)
v_opt = torch.optim.Adam(v.parameters(), lr=cfg.lr)

#初始化经验池
replay_buffer = ReplayBuffer(
    env.observation_space.shape,
    action_dim,
    cfg.replay_size,
    device=cfg.device
)

#SAC算法主逻辑
def sac_update(
    actor, q1, q2, v, v_target,
    actor_opt, q1_opt, q2_opt, v_opt,
    batch, cfg: SACConfig
):
    """
    :param actor: 策略网络
    :param q1: q网络1
    :param q2: q网络2
    :param v: 价值网络
    :param v_target: 目标价值网络
    :param actor_opt: 策略网络优化器
    :param q1_opt: q网络1优化器
    :param q2_opt: q网络2优化器
    :param v_opt: 价值网络优化器
    :param batch: 从经验池采样的一批数据
    :param cfg: 超参数配置
    """
    obs, acts, rews, next_obs, done = batch
    obs = obs.to(cfg.device)
    next_obs = next_obs.to(cfg.device)
    acts = acts.to(cfg.device).long()
    rews = rews.to(cfg.device)
    done = done.to(cfg.device)

    # 1) 计算当前策略下的动作分布
    probs, log_probs = actor(obs)             # (B,A)

    # 2) Q1,Q2(s, ·) -> 取所有动作
    q1_all = q1(obs)                          # (B,A)
    q2_all = q2(obs)                          # (B,A)
    q_min_all = torch.min(q1_all, q2_all) # 取最小值作为目标q估计

    # === 5.1 更新 V 网络 ===  
    with torch.no_grad():
        inside = q_min_all - cfg.alpha * log_probs   # (B,A)
        target_v = (probs * inside).sum(dim=-1, keepdim=True)  # E_a[Q - α log π]
    v_pred = v(obs)                                  # (B,1)
    v_loss = 0.5 * F.mse_loss(v_pred, target_v)

    v_opt.zero_grad()
    v_loss.backward()
    v_opt.step()

    # === 5.2 更新 Q1,Q2 
    with torch.no_grad():
        v_next = v_target(next_obs)                  # (B,1)
        q_target = rews + cfg.gamma * (1 - done) * v_next

    q1_pred_all = q1_all.gather(1, acts)             # (B,1)
    q2_pred_all = q2_all.gather(1, acts)             # (B,1)

    q1_loss = 0.5 * F.mse_loss(q1_pred_all, q_target)
    q2_loss = 0.5 * F.mse_loss(q2_pred_all, q_target)

    q1_opt.zero_grad()
    q1_loss.backward()
    q1_opt.step()

    q2_opt.zero_grad()
    q2_loss.backward()
    q2_opt.step()

    # === 5.3 更新策略 π === 
    # 在离散动作下可以直接用 KL 的显式形式：
    # Jπ = E_s [ Σ_a π(a|s) ( α log π(a|s) - Q_min(s,a) ) ]
    probs, log_probs = actor(obs)            # 重新前向一次以获得梯度
    q1_all = q1(obs)
    q2_all = q2(obs)
    q_min_all = torch.min(q1_all, q2_all)

    inside = cfg.alpha * log_probs - q_min_all
    policy_loss = (probs * inside).sum(dim=-1).mean()

    actor_opt.zero_grad()
    policy_loss.backward()
    actor_opt.step()

    # === 5.4 软更新 target V ===
    with torch.no_grad():
        for p, p_targ in zip(v.parameters(), v_target.parameters()):
            p_targ.data.mul_(1 - cfg.tau)
            p_targ.data.add_(cfg.tau * p.data)

    return {
        "v_loss": v_loss.item(),
        "q1_loss": q1_loss.item(),
        "q2_loss": q2_loss.item(),
        "pi_loss": policy_loss.item(),
    }


# 设置可复现的随机种子
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Gym库的环境重置
def reset_env(env):
    reset_out = env.reset()
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


# Gym库的环境动作执行
def step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        next_obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        next_obs, reward, done, info = step_out
    return next_obs, reward, done, info


# 主训练循环
def main():
    set_seed(42)
    # 训练记录
    training_logs = {
        'steps': [],
        'v_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'pi_loss': [],
        'episode_rewards': [],
        'episode_lengths': []
    }
    obs = reset_env(env)
    episode_reward = 0.0
    episode_len = 0
    episode_count = 0
    loss_info = {}

    actor.train()
    q1.train()
    q2.train()
    v.train()
    v_target.train()
    
    start_time = time.time()
    print(f"Starting SAC training on Super Mario Bros")
    print(f"Total steps: {cfg.total_steps}")
    print(f"Device: {cfg.device}")
    print(f"Action space: {action_dim} actions")
    print("-" * 50)

    for step in range(cfg.total_steps):
        # 1) 初始采取随机探索
        if step < cfg.start_steps:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.as_tensor(obs, device=cfg.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _, _ = actor.sample(obs_tensor)
            action = int(action_tensor.item())

        # 2) 和环境交互，获得奖励和下一状态
        next_obs, reward, done, _ = step_env(env, action)
        replay_buffer.store(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_reward += reward
        episode_len += 1

        # 3) 策略网络更新
        if step >= cfg.update_after and step % cfg.update_every == 0:
            for _ in range(cfg.gradient_steps):
                batch = replay_buffer.sample_batch(cfg.batch_size)
                loss_info = sac_update(
                    actor, q1, q2, v, v_target,
                    actor_opt, q1_opt, q2_opt, v_opt,
                    batch, cfg
                )
            # 记录损失
                if step % cfg.log_interval == 0:
                    training_logs['steps'].append(step)
                    training_logs['v_loss'].append(loss_info['v_loss'])
                    training_logs['q1_loss'].append(loss_info['q1_loss'])
                    training_logs['q2_loss'].append(loss_info['q2_loss'])
                    training_logs['pi_loss'].append(loss_info['pi_loss'])
                    
                    # 打印进度
                    if step % (cfg.log_interval * 10) == 0:
                        elapsed_time = time.time() - start_time
                        progress = (step + 1) / cfg.total_steps * 100
                        print(f"Step {step+1:6d}/{cfg.total_steps} | "
                              f"Progress: {progress:5.1f}% | "
                              f"Time: {elapsed_time:.1f}s | "
                              f"V Loss: {loss_info['v_loss']:.3f} | "
                              f"Q1 Loss: {loss_info['q1_loss']:.3f} | "
                              f"Pi Loss: {loss_info['pi_loss']:.3f}")

        # 4) 重置回合统计
        if done:
            training_logs['episode_rewards'].append(episode_reward)
            training_logs['episode_lengths'].append(episode_len)
            episode_count += 1
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(training_logs['episode_rewards'][-10:]) if len(training_logs['episode_rewards']) >= 10 else episode_reward
                print(f"Episode {episode_count:3d} | "
                      f"Step {step:6d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Length: {episode_len:3d} | "
                      f"Avg Reward (last 10): {avg_reward:6.1f}")
            
            obs = reset_env(env)
            episode_reward = 0.0
            episode_len = 0
        
        # 5) 定期保存检查点
        if (step + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"sac_mario_checkpoint_{step + 1}.pt")
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "q1": q1.state_dict(),
                    "q2": q2.state_dict(),
                    "v": v.state_dict(),
                    "v_target": v_target.state_dict(),
                    "optimizer_actor": actor_opt.state_dict(),
                    "optimizer_q1": q1_opt.state_dict(),
                    "optimizer_q2": q2_opt.state_dict(),
                    "optimizer_v": v_opt.state_dict(),
                    "step": step,
                    "episode": episode_count,
                },
                ckpt_path
            )
            
            # 保存训练日志
            log_path = os.path.join(cfg.output_dir, "training_logs.json")
            with open(log_path, 'w') as f:
                json.dump(training_logs, f, indent=2)
            
            print(f"Checkpoint and logs saved at step {step+1}")
    
    # 训练结束，保存最终模型和日志
    final_ckpt_path = os.path.join(cfg.output_dir, "sac_mario_final.pt")
    torch.save(
        {
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "v": v.state_dict(),
            "v_target": v_target.state_dict(),
            "optimizer_actor": actor_opt.state_dict(),
            "optimizer_q1": q1_opt.state_dict(),
            "optimizer_q2": q2_opt.state_dict(),
            "optimizer_v": v_opt.state_dict(),
            "step": cfg.total_steps,
            "episode": episode_count,
        },
        final_ckpt_path
    )
    
    log_path = os.path.join(cfg.output_dir, "training_logs.json")
    with open(log_path, 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    env.close()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total episodes: {episode_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final model and logs saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()