import sys
import os

# Force CPU-only mode to avoid CUDA library issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.seed_utils import set_seed


def train_agent(env_name, use_bc_init=False, seed=42):
    """Train a PPO agent, optionally initialized with BC weights"""
    # Set seed for reproducibility
    set_seed(seed)
    
    print(f"Training {'BC→RL' if use_bc_init else 'Pure RL'} on {env_name} (seed={seed})")

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

    # PPO Network (matching src/networks.py architecture)
    class PPONetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),  # Changed from 64 to 128 to match src/networks.py
                nn.ReLU(),
            )
            self.actor = nn.Linear(128, action_dim)  # Changed from 64 to 128
            self.critic = nn.Linear(128, 1)  # Changed from 64 to 128

        def forward(self, x):
            features = self.shared(x)
            action_logits = self.actor(features)
            value = self.critic(features)
            return action_logits, value

    # Create PPO agent with proper learning rate from config
    model = PPONetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)  # Fixed: was 1e-4, should be 3e-4 from config

    # Load BC weights if requested
    if use_bc_init:
        bc_path = f"bc_{env_name}.pth"
        if os.path.exists(bc_path):
            try:
                # Load BC network (matching src/networks.py architecture)
                bc_network = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),  # Changed from 64 to 128
                    nn.ReLU(),
                    nn.Linear(128, action_dim)  # Changed from 64 to action_dim
                )
                bc_network.load_state_dict(torch.load(bc_path, map_location='cpu'))

                # Transfer weights to shared layers AND actor
                model.shared.load_state_dict(bc_network[:4].state_dict())
                
                print("Initialized with BC weights (including actor head)")
            except Exception as e:
                print(f"WARNING: Could not load BC weights: {e}, using random initialization")
        else:
            print(f"WARNING: BC model not found: {bc_path}")

    # Training parameters
    num_episodes = 800 if env_name == 'CartPole-v1' else 1000  # Increased for better convergence
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

        # Normalize advantages (more carefully)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(np.array(actions))
        old_log_probs_tensor = torch.FloatTensor(np.array(log_probs))
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)

        # PPO update (using config value)
        for _ in range(4):  # Changed from 10 to 4 (from config)
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
    print(f"Model saved: {model_path}")

    return rewards


def compare_methods(env_name='CartPole-v1', seed=42):
    """Compare BC-only, Pure RL, and BC→RL"""
    # Set seed for reproducibility
    set_seed(seed)
    
    print("=" * 60)
    print(f"Comparing BC-only, Pure RL, and BC→RL (seed={seed})")
    print("=" * 60)

    # Evaluate BC-only if available
    bc_only_rewards = None
    bc_path = f"bc_{env_name}.pth"
    if os.path.exists(bc_path):
        print("\nEvaluating BC-only agent...")
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
        
        # Create BC agent (matching src/networks.py architecture)
        class BehaviorCloningAgent:
            def __init__(self, state_dim, action_dim):
                self.device = torch.device("cpu")
                self.policy_network = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),  # Changed from 64 to 128
                    nn.ReLU(),
                    nn.Linear(128, action_dim)  # Changed from 64 to action_dim
                ).to(self.device)
            
            def select_action(self, state):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_logits = self.policy_network(state_tensor)
                return action_logits.argmax().item()
        
        bc_agent = BehaviorCloningAgent(state_dim, action_dim)
        bc_agent.policy_network.load_state_dict(torch.load(bc_path, map_location='cpu'))
        bc_agent.policy_network.eval()
        
        # Evaluate BC agent
        bc_only_rewards = []
        num_eval = 50  # Evaluate for 50 episodes to match training length
        for _ in range(num_eval):
            state = env.reset()
            episode_reward = 0
            
            while True:
                action = bc_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            bc_only_rewards.append(episode_reward)
        
        env.close()
        print(f"BC-only avg reward: {np.mean(bc_only_rewards):.2f}")
    else:
        print("\nWARNING: BC model not found, skipping BC-only evaluation")

    # Train both RL methods
    print("\nTraining Pure RL...")
    pure_rl_rewards = train_agent(env_name, use_bc_init=False)

    print("\nTraining BC→RL...")
    bc_rl_rewards = train_agent(env_name, use_bc_init=True)

    # Plot comparison with improved styling including BC-only
    # Create visualization with 3 subplots (1 row x 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left: Training progress comparison
    ax1 = axes[0]
    
    # Plot BC-only as horizontal line if available
    if bc_only_rewards is not None:
        bc_mean = np.mean(bc_only_rewards)
        ax1.axhline(bc_mean, color='green', linestyle='-', linewidth=2.5, 
                   alpha=0.7, label=f'BC-only (Mean: {bc_mean:.1f})', zorder=5)
        ax1.fill_between(range(len(pure_rl_rewards)), 
                        bc_mean - np.std(bc_only_rewards),
                        bc_mean + np.std(bc_only_rewards),
                        color='green', alpha=0.1)
    
    # Plot training curves
    ax1.plot(pure_rl_rewards, alpha=0.3, color='blue', linewidth=1)
    ax1.plot(bc_rl_rewards, alpha=0.3, color='orange', linewidth=1)
    
    # Smooth curves
    window = 10
    smooth_pure = np.convolve(pure_rl_rewards, np.ones(window) / window, mode='valid')
    smooth_bc = np.convolve(bc_rl_rewards, np.ones(window) / window, mode='valid')
    
    ax1.plot(range(window-1, len(pure_rl_rewards)), smooth_pure, 
            label='Pure RL', linewidth=2.5, alpha=0.9, color='blue')
    ax1.plot(range(window-1, len(bc_rl_rewards)), smooth_bc, 
            label='BC→RL', linewidth=2.5, alpha=0.9, color='orange')
    
    # Add threshold line
    if env_name == 'CartPole-v1':
        threshold = 195
    elif env_name in ['LunarLander-v2', 'LunarLander-v3']:
        threshold = 200
    else:
        threshold = 200
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title(f'Training Progress: BC-only vs Pure RL vs BC→RL - {env_name}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Middle: Cumulative reward comparison
    ax2 = axes[1]
    cumulative_pure = np.cumsum(pure_rl_rewards)
    cumulative_bc = np.cumsum(bc_rl_rewards)
    
    ax2.plot(cumulative_pure, label='Pure RL', linewidth=2.5, color='blue')
    ax2.plot(cumulative_bc, label='BC→RL', linewidth=2.5, color='orange')
    
    # Add BC-only cumulative baseline
    if bc_only_rewards is not None:
        bc_cumulative = np.cumsum([np.mean(bc_only_rewards)] * len(pure_rl_rewards))
        ax2.plot(bc_cumulative, label='BC-only (constant)', linewidth=2.5, 
                color='green', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Cumulative Reward', fontsize=12)
    ax2.set_title('Cumulative Reward Over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Reward distribution comparison (final performance)
    ax3 = axes[2]
    
    if bc_only_rewards is not None:
        ax3.hist(bc_only_rewards, bins=15, alpha=0.5, color='green', 
                label='BC-only (50 episodes)', edgecolor='black')
    
    ax3.hist(pure_rl_rewards[-100:], bins=15, alpha=0.5, color='blue', 
            label='Pure RL (last 100)', edgecolor='black')
    ax3.hist(bc_rl_rewards[-100:], bins=15, alpha=0.5, color='orange', 
            label='BC→RL (last 100)', edgecolor='black')
    
    # Add mean lines
    if bc_only_rewards is not None:
        ax3.axvline(np.mean(bc_only_rewards), color='green', 
                   linestyle='--', linewidth=2.5, alpha=0.8)
    ax3.axvline(np.mean(pure_rl_rewards[-100:]), color='blue', 
               linestyle='--', linewidth=2.5, alpha=0.8)
    ax3.axvline(np.mean(bc_rl_rewards[-100:]), color='orange', 
               linestyle='--', linewidth=2.5, alpha=0.8)
    
    ax3.set_xlabel('Episode Reward', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Final Performance Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Calculate statistics for console output (but don't plot them)
    def episodes_to_threshold(rewards, threshold):
        for i, reward in enumerate(rewards):
            if reward >= threshold:
                return i + 1
        return len(rewards)
    
    pure_episodes = episodes_to_threshold(pure_rl_rewards, threshold)
    bc_episodes = episodes_to_threshold(bc_rl_rewards, threshold)
    improvement = ((pure_episodes - bc_episodes) / pure_episodes * 100) if bc_episodes < pure_episodes else 0
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'images/comparison_{env_name}.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: images/comparison_{env_name}.png")
    plt.close()

    # Print statistics to console
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    if bc_only_rewards is not None:
        print(f"\nBC-only:")
        print(f"  Mean: {np.mean(bc_only_rewards):.2f}")
        print(f"  Std: {np.std(bc_only_rewards):.2f}")
        print(f"  Max: {np.max(bc_only_rewards):.2f}")

    print(f"\nPure RL (last 100 episodes):")
    print(f"  Mean: {np.mean(pure_rl_rewards[-100:]):.2f}")
    print(f"  Std: {np.std(pure_rl_rewards[-100:]):.2f}")
    print(f"  Max: {np.max(pure_rl_rewards[-100:]):.2f}")

    print(f"\nBC→RL (last 100 episodes):")
    print(f"  Mean: {np.mean(bc_rl_rewards[-100:]):.2f}")
    print(f"  Std: {np.std(bc_rl_rewards[-100:]):.2f}")
    print(f"  Max: {np.max(bc_rl_rewards[-100:]):.2f}")

    print(f"\nSAMPLE EFFICIENCY:")
    print(f"  Pure RL reached threshold in {pure_episodes} episodes")
    print(f"  BC→RL reached threshold in {bc_episodes} episodes")

    if bc_episodes < pure_episodes:
        print(f"  BC→RL is {improvement:.1f}% faster!")
    else:
        print("  No improvement in sample efficiency")

    print("\nComparison complete!")
    
    # Print LaTeX-ready summary for paper
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.latex_utils import print_quick_summary
        print_quick_summary(env_name, pure_rl_rewards, bc_rl_rewards, bc_only_rewards)
    except Exception as e:
        print(f"Note: Could not generate LaTeX summary: {e}")


def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return compare_methods(env_name)


if __name__ == "__main__":
    compare_methods()