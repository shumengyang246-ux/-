"""
绘制SAC算法训练结果曲线图，先运行训练脚本，生成训练日志后再运行绘图脚本
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(log_file_path="sac_mario_results/training_logs.json"):
    """
    绘制SAC训练结果
    
    参数:
        log_file_path: 训练日志文件的路径，替换为自己的路径
    """
    
    # 读取训练日志
    try:
        with open(log_file_path, 'r') as f:
            training_logs = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {log_file_path}")
        print("请确保训练脚本已运行并生成了日志文件")
        return
    
    # 创建图形
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('SAC Training Results - Super Mario Bros', fontsize=16, fontweight='bold')
    
    # 1. 损失函数曲线
    if training_logs['steps'] and training_logs['v_loss']:
        steps = training_logs['steps']
        
        # V损失
        ax = axes[0, 0]
        ax.plot(steps, training_logs['v_loss'], 'b-', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('V Loss')
        ax.set_title('Value Network Loss')
        ax.grid(True, alpha=0.3)
        
        # 平滑处理
        if len(steps) > 100:
            window_size = min(50, len(steps) // 10)
            smooth_v_loss = np.convolve(training_logs['v_loss'], np.ones(window_size)/window_size, mode='valid')
            smooth_steps = steps[window_size-1:]
            ax.plot(smooth_steps, smooth_v_loss, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
            ax.legend()
    
    # 2. Q损失
    if training_logs['steps'] and training_logs['q1_loss']:
        steps = training_logs['steps']
        
        ax = axes[0, 1]
        ax.plot(steps, training_logs['q1_loss'], 'g-', alpha=0.7, linewidth=1.5, label='Q1 Loss')
        if training_logs['q2_loss']:
            ax.plot(steps, training_logs['q2_loss'], 'b-', alpha=0.7, linewidth=1.5, label='Q2 Loss')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Q Loss')
        ax.set_title('Q Network Losses')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 平滑处理
        if len(steps) > 100:
            window_size = min(50, len(steps) // 10)
            smooth_q1_loss = np.convolve(training_logs['q1_loss'], np.ones(window_size)/window_size, mode='valid')
            smooth_steps = steps[window_size-1:]
            ax.plot(smooth_steps, smooth_q1_loss, 'r-', linewidth=2, label=f'Smoothed Q1')
            if training_logs['q2_loss']:
                smooth_q2_loss = np.convolve(training_logs['q2_loss'], np.ones(window_size)/window_size, mode='valid')
                ax.plot(smooth_steps, smooth_q2_loss, 'orange', linewidth=2, label=f'Smoothed Q2')
            ax.legend()
    
    # 3. 策略损失
    if training_logs['steps'] and training_logs['pi_loss']:
        steps = training_logs['steps']
        
        ax = axes[1, 0]
        ax.plot(steps, training_logs['pi_loss'], 'purple', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Network Loss')
        ax.grid(True, alpha=0.3)
        
        # 平滑处理
        if len(steps) > 100:
            window_size = min(50, len(steps) // 10)
            smooth_pi_loss = np.convolve(training_logs['pi_loss'], np.ones(window_size)/window_size, mode='valid')
            smooth_steps = steps[window_size-1:]
            ax.plot(smooth_steps, smooth_pi_loss, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
            ax.legend()
    
    # 4. 所有损失对比
    if training_logs['steps']:
        steps = training_logs['steps']
        
        ax = axes[1, 1]
        
        # 归一化损失以便比较
        if training_logs['v_loss']:
            v_loss_norm = np.array(training_logs['v_loss']) / max(training_logs['v_loss']) if max(training_logs['v_loss']) > 0 else np.array(training_logs['v_loss'])
            ax.plot(steps, v_loss_norm, 'b-', alpha=0.7, linewidth=1.5, label='V Loss (normalized)')
        
        if training_logs['q1_loss']:
            q1_loss_norm = np.array(training_logs['q1_loss']) / max(training_logs['q1_loss']) if max(training_logs['q1_loss']) > 0 else np.array(training_logs['q1_loss'])
            ax.plot(steps, q1_loss_norm, 'g-', alpha=0.7, linewidth=1.5, label='Q1 Loss (normalized)')
        
        if training_logs['pi_loss']:
            pi_loss_norm = np.array(training_logs['pi_loss']) / max(training_logs['pi_loss']) if max(training_logs['pi_loss']) > 0 else np.array(training_logs['pi_loss'])
            ax.plot(steps, pi_loss_norm, 'purple', alpha=0.7, linewidth=1.5, label='Policy Loss (normalized)')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Normalized Loss')
        ax.set_title('Normalized Loss Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 5. 回合奖励
    if training_logs['episode_rewards']:
        episodes = list(range(1, len(training_logs['episode_rewards']) + 1))
        
        ax = axes[2, 0]
        ax.plot(episodes, training_logs['episode_rewards'], 'b-', alpha=0.5, linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)
        
        # 移动平均
        if len(episodes) > 10:
            window_size = min(20, len(episodes) // 5)
            moving_avg = np.convolve(training_logs['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
            avg_episodes = episodes[window_size-1:]
            ax.plot(avg_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
            ax.legend()
    
    # 6. 回合长度
    if training_logs['episode_lengths']:
        episodes = list(range(1, len(training_logs['episode_lengths']) + 1))
        
        ax = axes[2, 1]
        ax.plot(episodes, training_logs['episode_lengths'], 'g-', alpha=0.5, linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
        
        # 移动平均
        if len(episodes) > 10:
            window_size = min(20, len(episodes) // 5)
            moving_avg = np.convolve(training_logs['episode_lengths'], np.ones(window_size)/window_size, mode='valid')
            avg_episodes = episodes[window_size-1:]
            ax.plot(avg_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
            ax.legend()
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else "."
    output_path = os.path.join(output_dir, "training_plots.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"训练结果图已保存到: {output_path}")
    
    plt.show()

def plot_loss_summary(log_file_path="sac_mario_results/training_logs.json"):
    """
    绘制损失总结图
    
    参数:
        log_file_path: 训练日志文件的路径
    """
    
    # 读取训练日志
    try:
        with open(log_file_path, 'r') as f:
            training_logs = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {log_file_path}")
        return
    
    if not training_logs['steps']:
        print("错误: 日志文件中没有训练数据")
        return
    
    # 创建单独的损失图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = training_logs['steps']
    
    # 原始损失曲线
    ax1 = axes[0, 0]
    if training_logs['v_loss']:
        ax1.plot(steps, training_logs['v_loss'], 'b-', label='V Loss', alpha=0.7)
    if training_logs['q1_loss']:
        ax1.plot(steps, training_logs['q1_loss'], 'g-', label='Q1 Loss', alpha=0.7)
    if training_logs['q2_loss']:
        ax1.plot(steps, training_logs['q2_loss'], 'c-', label='Q2 Loss', alpha=0.7)
    if training_logs['pi_loss']:
        ax1.plot(steps, training_logs['pi_loss'], 'purple', label='Policy Loss', alpha=0.7)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('SAC Loss Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 平滑后的损失曲线
    ax2 = axes[0, 1]
    if len(steps) > 100:
        window_size = min(100, len(steps) // 10)
        
        if training_logs['v_loss']:
            smooth_v = np.convolve(training_logs['v_loss'], np.ones(window_size)/window_size, mode='valid')
            ax2.plot(steps[window_size-1:], smooth_v, 'b-', label='V Loss', linewidth=2)
        
        if training_logs['q1_loss']:
            smooth_q1 = np.convolve(training_logs['q1_loss'], np.ones(window_size)/window_size, mode='valid')
            ax2.plot(steps[window_size-1:], smooth_q1, 'g-', label='Q1 Loss', linewidth=2)
        
        if training_logs['q2_loss']:
            smooth_q2 = np.convolve(training_logs['q2_loss'], np.ones(window_size)/window_size, mode='valid')
            ax2.plot(steps[window_size-1:], smooth_q2, 'c-', label='Q2 Loss', linewidth=2)
        
        if training_logs['pi_loss']:
            smooth_pi = np.convolve(training_logs['pi_loss'], np.ones(window_size)/window_size, mode='valid')
            ax2.plot(steps[window_size-1:], smooth_pi, 'purple', label='Policy Loss', linewidth=2)
        
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Smoothed Loss')
        ax2.set_title(f'Smoothed Loss Functions (window={window_size})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 损失分布直方图
    ax3 = axes[1, 0]
    losses = []
    labels = []
    colors = []
    
    if training_logs['v_loss']:
        losses.append(training_logs['v_loss'])
        labels.append('V Loss')
        colors.append('blue')
    
    if training_logs['q1_loss']:
        losses.append(training_logs['q1_loss'])
        labels.append('Q1 Loss')
        colors.append('green')
    
    if training_logs['pi_loss']:
        losses.append(training_logs['pi_loss'])
        labels.append('Policy Loss')
        colors.append('purple')
    
    if losses:
        # 计算平均损失
        avg_losses = [np.mean(loss) for loss in losses]
        std_losses = [np.std(loss) for loss in losses]
        
        x_pos = np.arange(len(labels))
        bars = ax3.bar(x_pos, avg_losses, yerr=std_losses, 
                      align='center', alpha=0.7, color=colors, 
                      capsize=5, edgecolor='black')
        
        ax3.set_ylabel('Average Loss')
        ax3.set_title('Average Loss Values with Std Dev')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels)
        
        # 在柱状图上显示数值
        for i, (bar, avg) in enumerate(zip(bars, avg_losses)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{avg:.3f}', ha='center', va='bottom')
    
    # 训练进度总结
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 收集统计信息
    stats_text = "Training Statistics:\n\n"
    
    if training_logs['steps']:
        total_steps = max(training_logs['steps'])
        stats_text += f"Total Steps: {total_steps:,}\n"
    
    if training_logs['episode_rewards']:
        total_episodes = len(training_logs['episode_rewards'])
        stats_text += f"Total Episodes: {total_episodes}\n"
        
        if total_episodes > 0:
            avg_reward = np.mean(training_logs['episode_rewards'])
            max_reward = np.max(training_logs['episode_rewards'])
            min_reward = np.min(training_logs['episode_rewards'])
            
            stats_text += f"\nReward Statistics:\n"
            stats_text += f"  Average: {avg_reward:.2f}\n"
            stats_text += f"  Maximum: {max_reward:.2f}\n"
            stats_text += f"  Minimum: {min_reward:.2f}\n"
    
    if training_logs['v_loss'] and training_logs['pi_loss']:
        final_v_loss = training_logs['v_loss'][-1] if training_logs['v_loss'] else 0
        final_pi_loss = training_logs['pi_loss'][-1] if training_logs['pi_loss'] else 0
        final_q1_loss = training_logs['q1_loss'][-1] if training_logs['q1_loss'] else 0
        
        stats_text += f"\nFinal Loss Values:\n"
        stats_text += f"  V Loss: {final_v_loss:.4f}\n"
        stats_text += f"  Q1 Loss: {final_q1_loss:.4f}\n"
        stats_text += f"  Policy Loss: {final_pi_loss:.4f}\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SAC Training Summary - Super Mario Bros', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else "."
    output_path = os.path.join(output_dir, "loss_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"损失总结图已保存到: {output_path}")
    
    plt.show()

# 主执行函数
if __name__ == "__main__":
    # 默认日志文件路径
    log_file = "sac_mario_results/training_logs.json"
    
    # 如果文件不存在，尝试在当前目录查找
    if not os.path.exists(log_file):
        print(f"在默认路径未找到日志文件: {log_file}")
        print("正在搜索其他可能的位置...")
        
        possible_paths = [
            "./training_logs.json",
            "../sac_mario_results/training_logs.json",
            "training_logs.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                log_file = path
                print(f"找到日志文件: {log_file}")
                break
        
        if not os.path.exists(log_file):
            print("未找到训练日志文件。请先运行训练脚本。")
            exit(1)
    
    print(f"使用日志文件: {log_file}")
    
    # 绘制完整的训练结果图
    plot_training_results(log_file)
    
    # 绘制损失总结图
    plot_loss_summary(log_file)
    
    print("\n绘图完成！")