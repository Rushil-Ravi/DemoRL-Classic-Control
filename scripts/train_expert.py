import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Import from src if available, otherwise define locally
try:
    from src.environments import EnvironmentWrapper
    from src.networks import QNetwork

    print("✅ Using src module imports")
except ImportError:
    import gymnasium as gym

    print("⚠️ Using local implementations")


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
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, x):
            return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def select_action(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.q_network.net[-1].out_features)

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

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        if np.random.random() < 0.01:  # 1% chance each update
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


def train_expert(env_name='CartPole-v1', num_episodes=200):
    """Train expert DQN agent"""
    print(f"🚀 Training Expert Agent on {env_name}")

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
    print(f"💾 Expert model saved: {model_path}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Expert Training on {env_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'expert_training_{env_name}.png')
    plt.show()

    env.close()

    print(f"\n✅ Expert training complete!")
    print(f"Final average reward: {np.mean(episode_rewards[-20:]):.2f}")

    return agent


# Make sure this function exists for main.py to call
def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return train_expert(env_name)


if __name__ == "__main__":
    train_expert()