"""
Parse loss and average reward from log.txt and plot them against training steps.

Usage:
    python plot_log_metrics.py

Outputs:
    - Displays a matplotlib window with two subplots.
    - Saves the figure to training_metrics.png in the same directory.
"""
from pathlib import Path
import re

import matplotlib.pyplot as plt


def load_metrics(log_path: Path):
    """Extract steps, losses, and avg rewards from the log."""
    pattern = re.compile(
        r"Step\s+(\d+),\s*Loss:\s*([-\d.eE]+),\s*Avg Reward:\s*([-\d.eE]+)",
        re.IGNORECASE,
    )
    steps, losses, rewards = [], [], []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step, loss, reward = match.groups()
                steps.append(int(step))
                losses.append(float(loss))
                rewards.append(float(reward))

    return steps, losses, rewards


def plot_metrics(steps, losses, rewards):
    if not steps:
        raise ValueError("No step entries were found in the log.")

    fig, (ax_loss, ax_reward) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
    )

    ax_loss.plot(steps, losses, color="#1f77b4", linewidth=1.5)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss vs Step")
    ax_loss.grid(True, linestyle="--", alpha=0.4)

    ax_reward.plot(steps, rewards, color="#d62728", linewidth=1.5)
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Average Reward")
    ax_reward.set_title("Average Reward vs Step")
    ax_reward.grid(True, linestyle="--", alpha=0.4)

    fig.savefig("training_metrics.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    log_file = Path("log.txt")
    if not log_file.exists():
        raise FileNotFoundError(f"Could not find {log_file}")

    steps_data, losses_data, rewards_data = load_metrics(log_file)
    plot_metrics(steps_data, losses_data, rewards_data)
