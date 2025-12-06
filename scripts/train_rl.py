import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def train_agent(env_name, use_bc_init=False):
    """Train a PPO agent, optionally initialized with BC weights"""
    print(f"🤖 Training {'BC→RL' if use_bc_init else 'Pure RL'} on {env_name}")

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

    # PPO Network
    class PPONetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.actor = nn.Linear(64, action_dim)
            self.critic = nn.Linear(64, 1)

        def forward(self, x):
            features = self.shared(x)
            action_logits = self.actor(features)
            value = self.critic(features)
            return action_logits, value

    # Create PPO agent
    model = PPONetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Load BC weights if requested
    if use_bc_init:
        bc_path = f"bc_{env_name}.pth"
        if os.path.exists(bc_path):
            try:
                # Load BC network
                bc_network = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim)
                )
                bc_network.load_state_dict(torch.load(bc_path, map_location='cpu'))

                # Transfer weights to shared layers
                model.shared.load_state_dict(bc_network[:4].state_dict())
                print("✅ Initialized with BC weights")
            except:
                print("⚠️ Could not load BC weights, using random initialization")
        else:
            print(f"⚠️ BC model not found: {bc_path}")

    # Training parameters
    num_episodes = 300 if env_name == 'CartPole-v1' else 500
    gamma = 0.99
    clip_epsilon = 0.2

    rewards = []

    for episode in range(num_episodes):
        # Collect trajectory
        states = []
        actions = []
        rewards_list = []
        values = []
        log_probs = []
        dones = []

        state = env.reset()
        episode_reward = 0

        while True:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_logits, value = model(state_tensor)
                action_dist = torch.distributions.Categorical(logits=action_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Store
            states.append(state)
            actions.append(action.item())
            rewards_list.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

        # Compute advantages (simple version)
        advantages = []
        returns = []
        R = 0

        for r, done in zip(reversed(rewards_list), reversed(dones)):
            if done:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)

        returns = np.array(returns)
        values_array = np.array(values)
        advantages = returns - values_array

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(np.array(actions))
        old_log_probs_tensor = torch.FloatTensor(np.array(log_probs))
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)

        # PPO update (single epoch for simplicity)
        for _ in range(4):  # PPO epochs
            action_logits, values_pred = model(states_tensor)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = action_dist.log_prob(actions_tensor)
            entropy = action_dist.entropy().mean()

            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surrogate1 = ratio * advantages_tensor
            surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values_pred, returns_tensor)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward} | "
                  f"Avg (50): {avg_reward:.1f}")

    env.close()

    # Save model
    suffix = 'bc_rl' if use_bc_init else 'pure_rl'
    model_path = f"{suffix}_{env_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")

    return rewards


def compare_methods(env_name='CartPole-v1'):
    """Compare Pure RL vs BC→RL"""
    print("=" * 60)
    print("🏆 Comparing Pure RL vs BC→RL")
    print("=" * 60)

    # Train both methods
    print("\nTraining Pure RL...")
    pure_rl_rewards = train_agent(env_name, use_bc_init=False)

    print("\nTraining BC→RL...")
    bc_rl_rewards = train_agent(env_name, use_bc_init=True)

    # Plot comparison
    plt.figure(figsize=(10, 6))

    # Smooth curves
    window = 10
    smooth_pure = np.convolve(pure_rl_rewards, np.ones(window) / window, mode='valid')
    smooth_bc = np.convolve(bc_rl_rewards, np.ones(window) / window, mode='valid')

    plt.plot(smooth_pure, label='Pure RL', linewidth=2, alpha=0.8)
    plt.plot(smooth_bc, label='BC→RL', linewidth=2, alpha=0.8)

    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (window=10)')
    plt.title(f'Comparison on {env_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'comparison_{env_name}.png')
    plt.show()

    # Statistics
    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)

    print(f"\nPure RL (last 100 episodes):")
    print(f"  Mean: {np.mean(pure_rl_rewards[-100:]):.2f}")
    print(f"  Std: {np.std(pure_rl_rewards[-100:]):.2f}")
    print(f"  Max: {np.max(pure_rl_rewards[-100:]):.2f}")

    print(f"\nBC→RL (last 100 episodes):")
    print(f"  Mean: {np.mean(bc_rl_rewards[-100:]):.2f}")
    print(f"  Std: {np.std(bc_rl_rewards[-100:]):.2f}")
    print(f"  Max: {np.max(bc_rl_rewards[-100:]):.2f}")

    # Sample efficiency analysis
    threshold = 195 if env_name == 'CartPole-v1' else 200

    def episodes_to_threshold(rewards, threshold):
        for i, reward in enumerate(rewards):
            if reward >= threshold:
                return i + 1
        return len(rewards)

    pure_episodes = episodes_to_threshold(pure_rl_rewards, threshold)
    bc_episodes = episodes_to_threshold(bc_rl_rewards, threshold)

    print(f"\n⏱️  SAMPLE EFFICIENCY:")
    print(f"  Pure RL reached threshold in {pure_episodes} episodes")
    print(f"  BC→RL reached threshold in {bc_episodes} episodes")

    if bc_episodes < pure_episodes:
        improvement = (pure_episodes - bc_episodes) / pure_episodes * 100
        print(f"  ✅ BC→RL is {improvement:.1f}% faster!")
    else:
        print("  ⚠️  No improvement in sample efficiency")

    print("\n✅ Comparison complete!")


def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return compare_methods(env_name)


if __name__ == "__main__":
    compare_methods()