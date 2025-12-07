import sys
import os

# Force CPU-only mode to avoid CUDA library issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.seed_utils import set_seed


def evaluate_agent(agent, env, num_episodes=20):
    """Evaluate an agent's performance"""
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            # Different agents have different select_action methods
            if hasattr(agent, 'select_action'):
                # BC or DQN agent
                action = agent.select_action(state)
            else:
                # PPO agent (model)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    if hasattr(agent, 'shared'):
                        # PPO network
                        action_logits, _ = agent(state_tensor)
                    else:
                        # Simple network
                        action_logits = agent(state_tensor)
                    action = action_logits.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                break

        rewards.append(episode_reward)

    return rewards


def evaluate_all(env_name='CartPole-v1', seed=42):
    """Evaluate all trained agents"""
    # Set seeds for reproducibility
    set_seed(seed)
    
    print("=" * 60)
    print(f"FINAL EVALUATION (seed={seed})")
    print("=" * 60)

    # Create environment
    try:
        from src.environments import EnvironmentWrapper
    except ImportError:
        import gymnasium as gym

        class EnvironmentWrapper:
            def __init__(self, env_name):
                self.env = gym.make(env_name)

            def get_state_dim(self):
                return self.env.observation_space.shape[0]

            def get_action_dim(self):
                return self.env.action_space.n

            def reset(self):
                obs, _ = self.env.reset()
                return obs

            def step(self, action):
                return self.env.step(action)

            def close(self):
                self.env.close()

    env = EnvironmentWrapper(env_name)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    results = {}

    # Test BC-only
    print("\n[1] Testing BC-only...")
    bc_path = f"bc_{env_name}.pth"
    if os.path.exists(bc_path):
        # Create BC agent (matching src/networks.py architecture)
        class BehaviorCloningAgent:
            def __init__(self, state_dim, action_dim):
                self.device = torch.device("cpu")
                self.policy_network = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),  # Changed from 64 to 128
                    nn.ReLU(),
                    nn.Linear(128, action_dim)  # Changed from 64 to 128
                ).to(self.device)

            def select_action(self, state):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_logits = self.policy_network(state_tensor)
                return action_logits.argmax().item()

        bc_agent = BehaviorCloningAgent(state_dim, action_dim)
        bc_agent.policy_network.load_state_dict(torch.load(bc_path, map_location='cpu'))
        bc_agent.policy_network.eval()

        bc_rewards = evaluate_agent(bc_agent, env, 20)
        results['BC-only'] = bc_rewards
        print(f"  Avg reward: {np.mean(bc_rewards):.2f} ± {np.std(bc_rewards):.2f}")
    else:
        print(f"  WARNING: BC model not found")

    # Test Pure RL
    print("\n[2] Testing Pure RL...")
    pure_rl_path = f"pure_rl_{env_name}.pth"
    if os.path.exists(pure_rl_path):
        # Create PPO network (matching src/networks.py architecture)
        class PPONetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),  # Changed from 64 to 128
                    nn.ReLU(),
                )
                self.actor = nn.Linear(128, action_dim)  # Changed from 64 to 128
                self.critic = nn.Linear(128, 1)  # Changed from 64 to 128

            def forward(self, x):
                features = self.shared(x)
                action_logits = self.actor(features)
                value = self.critic(features)
                return action_logits, value

        pure_rl_model = PPONetwork(state_dim, action_dim)
        pure_rl_model.load_state_dict(torch.load(pure_rl_path, map_location='cpu'))
        pure_rl_model.eval()

        pure_rl_rewards = evaluate_agent(pure_rl_model, env, 20)
        results['Pure RL'] = pure_rl_rewards
        print(f"  Avg reward: {np.mean(pure_rl_rewards):.2f} ± {np.std(pure_rl_rewards):.2f}")
    else:
        print(f"  WARNING: Pure RL model not found")

    # Test BC→RL
    print("\n[3] Testing BC→RL...")
    bc_rl_path = f"bc_rl_{env_name}.pth"
    if os.path.exists(bc_rl_path):
        # Create PPO network (matching src/networks.py architecture)
        class PPONetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),  # Changed from 64 to 128
                    nn.ReLU(),
                )
                self.actor = nn.Linear(128, action_dim)  # Changed from 64 to 128
                self.critic = nn.Linear(128, 1)  # Changed from 64 to 128

            def forward(self, x):
                features = self.shared(x)
                action_logits = self.actor(features)
                value = self.critic(features)
                return action_logits, value

        bc_rl_model = PPONetwork(state_dim, action_dim)
        bc_rl_model.load_state_dict(torch.load(bc_rl_path, map_location='cpu'))
        bc_rl_model.eval()

        bc_rl_rewards = evaluate_agent(bc_rl_model, env, 20)
        results['BC→RL'] = bc_rl_rewards
        print(f"  Avg reward: {np.mean(bc_rl_rewards):.2f} ± {np.std(bc_rl_rewards):.2f}")
    else:
        print(f"  WARNING: BC→RL model not found")

    env.close()

    # Create evaluation visualization - THE TESTING GRAPH!
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        labels = list(results.keys())
        data = [results[label] for label in labels]
        colors = ['lightgreen', 'lightblue', 'lightsalmon']
        
        # Panel 1: Box plot (Left)
        ax1 = axes[0]
        bp = ax1.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showmeans=True)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Test Performance Distribution\n(20 Episodes Each)', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax1.tick_params(axis='x', rotation=15)
        
        # Panel 2: Bar chart - MAIN TESTING GRAPH (Middle)
        ax2 = axes[1]
        means = [np.mean(results[label]) for label in labels]
        stds = [np.std(results[label]) for label in labels]
        x_pos = np.arange(len(labels))
        
        bars = ax2.bar(x_pos, means, yerr=stds, alpha=0.8, 
                      color=colors[:len(labels)], edgecolor='black', linewidth=2,
                      capsize=10, error_kw={'linewidth': 3})
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 10,
                    f'{mean:.1f}', ha='center', va='bottom', 
                    fontsize=13)
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'±{std:.1f}', ha='center', va='center', 
                    fontsize=11, style='italic')
        
        ax2.set_ylabel('Mean Reward ± Std Dev', fontsize=12, fontweight='bold')
        ax2.set_title('Final Test Performance\n(After All Training)', 
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=15)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add threshold lines if applicable
        if env_name == 'CartPole-v1':
            ax2.axhline(195, color='red', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Threshold (195)')
            ax2.axhline(500, color='green', linestyle='--', linewidth=2,
                       alpha=0.5, label='Perfect (500)')
            ax2.legend()
        
        # Panel 3: Individual episodes scatter (Right)
        ax3 = axes[2]
        for i, (label, color) in enumerate(zip(labels, colors[:len(labels)])):
            rewards = results[label]
            x = [i] * len(rewards)
            ax3.scatter(x, rewards, alpha=0.6, s=80, color=color, 
                       edgecolors='black', linewidth=1, label=label)
            ax3.hlines(np.mean(rewards), i-0.3, i+0.3, colors='red', linewidth=3)
        
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=15)
        ax3.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
        ax3.set_title('All Test Episodes\n(Each Dot = 1 Episode)', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.legend(loc='lower right')
        
        plt.suptitle(f'Final Testing Evaluation - {env_name}\n(Performance After All Training Complete)', 
                    fontsize=15, fontweight='bold', y=0.98)
        
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'images/evaluation_{env_name}.png', dpi=150, bbox_inches='tight')
        print(f"\nEvaluation plot saved: images/evaluation_{env_name}.png")
        plt.close()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return evaluate_all(env_name)


if __name__ == "__main__":
    evaluate_all()

