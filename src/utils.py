import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import yaml
import os


def save_model(model, path):
    """Save model weights"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """Load model weights"""
    model.load_state_dict(torch.load(path, map_location='cpu'))
    print(f"Model loaded from {path}")
    return model


def save_demonstrations(demos, path):
    """Save demonstration data"""
    with open(path, 'wb') as f:
        pickle.dump(demos, f)
    print(f"Saved {len(demos)} demonstrations to {path}")


def load_demonstrations(path):
    """Load demonstration data"""
    with open(path, 'rb') as f:
        demos = pickle.load(f)
    print(f"Loaded {len(demos)} demonstrations from {path}")
    return demos


def load_config():
    """Load configuration file"""
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)


def plot_training_curves(rewards, losses=None, title="Training Results", save_path=None):
    """Plot training rewards and losses"""
    fig, axes = plt.subplots(1, 2 if losses else 1, figsize=(12, 4))

    if losses:
        # Plot rewards
        axes[0].plot(rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].grid(True, alpha=0.3)

        # Plot losses
        axes[1].plot(losses)
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].grid(True, alpha=0.3)
    else:
        # Plot only rewards
        axes.plot(rewards)
        axes.set_xlabel('Episode')
        axes.set_ylabel('Reward')
        axes.set_title(title)
        axes.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()