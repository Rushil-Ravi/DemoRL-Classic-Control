import sys
import os
import random
# Force CPU-only mode to avoid CUDA library issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.seed_utils import set_seed

# Import from src if available, otherwise define locally
try:
    from src.environments import EnvironmentWrapper
    from src.networks import QNetwork

    print("Using src module imports")
except ImportError:
    import gymnasium as gym

    print("WARNING: Using local implementations")


    class EnvironmentWrapper:
        def __init__(self, env_name):
            self.env = gym.make(env_name)
            self.name = env_name

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


    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim  # Store action_dim

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Improved hyperparameters for more consistent training
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_frequency = 4  # Update every N steps
        self.target_update_frequency = 100  # Update target network every N steps
        self.steps = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)  # Use stored action_dim

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values (using target network for stability)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))

        # Compute loss (Huber loss for more stability)
        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network periodically (not randomly!)
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


def train_expert(env_name='CartPole-v1', num_episodes=500, seed=42):
    """Train expert DQN agent"""
    print(f"Training Expert Agent on {env_name} (seed={seed})")
    
    # Set seeds for reproducibility
    set_seed(seed)

    # Initialize environment
    env = EnvironmentWrapper(env_name)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Initialize agent
    agent = DQNAgent(state_dim, action_dim)

    # Training loop
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            # Select action
            action = agent.select_action(state)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.append((state, action, reward, next_state, done))

            # Update agent
            agent.update()

            # Update state
            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        # Logging
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward} | "
                  f"Avg (20): {avg_reward:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    # Save model
    model_path = f"expert_{env_name}.pth"
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"Expert model saved: {model_path}")

    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)

    # Plot results with improved styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Episode rewards with rolling average
    ax1.plot(episode_rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    
    # Add rolling average
    window = 20
    if len(episode_rewards) >= window:
        rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), rolling_avg, 
                color='darkblue', linewidth=2, label=f'{window}-Episode Moving Average')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title(f'Expert (DQN) Training Progress - {env_name}', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right plot: Reward distribution (last 50 episodes)
    recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
    ax2.hist(recent_rewards, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(np.mean(recent_rewards), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(recent_rewards):.1f}')
    ax2.set_xlabel('Episode Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Reward Distribution (Last {len(recent_rewards)} Episodes)', 
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(f'images/expert_training_{env_name}.png', dpi=150, bbox_inches='tight')
    print(f"Training plot saved: images/expert_training_{env_name}.png")
    plt.close()

    env.close()

    print(f"\nExpert training complete!")
    print(f"Final average reward (last 20 episodes): {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")

    return agent


# Make sure this function exists for main.py to call
def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return train_expert(env_name)


if __name__ == "__main__":
    train_expert()