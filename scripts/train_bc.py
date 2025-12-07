import sys
import os

# Force CPU-only mode to avoid CUDA library issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.seed_utils import set_seed


def train_bc(env_name='CartPole-v1', seed=42):
    """Train Behavior Cloning agent"""
    # Set seeds for reproducibility
    set_seed(seed)
    
    print(f"Training Behavior Cloning on {env_name} (seed={seed})")

    # Load demonstrations
    demo_path = f"demos_{env_name}.pkl"
    if not os.path.exists(demo_path):
        print(f"ERROR: Demonstrations not found: {demo_path}")
        print("Please run collect_demos.py first!")
        return

    with open(demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    print(f"Loaded {len(demonstrations)} demonstrations")

    # Prepare data
    all_states = []
    all_actions = []

    for demo in demonstrations:
        all_states.extend(demo['states'])
        all_actions.extend(demo['actions'])

    states = np.array(all_states)
    actions = np.array(all_actions)

    print(f"Training on {len(states)} transitions")

    # Create environment to get dimensions
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

    # Behavior Cloning Agent (FIXED: takes only 2 args)
    class BehaviorCloningAgent:
        def __init__(self, state_dim, action_dim):
            """Initialize BC agent with state and action dimensions"""
            self.device = torch.device("cpu")

            # Policy network (matching src/networks.py architecture)
            self.policy_network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),  # Changed from 64 to 128
                nn.ReLU(),
                nn.Linear(128, action_dim)  # Changed from 64 to 128
            ).to(self.device)

            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3)
            self.criterion = nn.CrossEntropyLoss()

        def train(self, states, actions, epochs=50, batch_size=32):
            """Train the BC agent"""
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)

            losses = []
            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(len(states))
                epoch_loss = 0
                num_batches = 0

                for i in range(0, len(states), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_states = states_tensor[batch_indices]
                    batch_actions = actions_tensor[batch_indices]

                    # Forward pass
                    action_logits = self.policy_network(batch_states)
                    loss = self.criterion(action_logits, batch_actions)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                losses.append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            return losses

        def select_action(self, state):
            """Select action given state"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_logits = self.policy_network(state_tensor)
            return action_logits.argmax().item()

    # Create and train BC agent
    agent = BehaviorCloningAgent(state_dim, action_dim)
    losses = agent.train(states, actions, epochs=50, batch_size=32)

    # Evaluate BC agent
    test_rewards = []
    for _ in range(20):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                break

        test_rewards.append(episode_reward)

    # Save BC model
    model_path = f"bc_{env_name}.pth"
    torch.save(agent.policy_network.state_dict(), model_path)
    print(f"BC model saved: {model_path}")

    # Plot results with improved styling
    fig = plt.figure(figsize=(12, 5))
    
    # Left plot: Training loss over epochs
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(losses, color='orangered', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('BC Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add final loss annotation
    final_loss = losses[-1] if losses else 0
    ax1.annotate(f'Final: {final_loss:.4f}', 
                xy=(len(losses)-1, final_loss),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Middle plot: Performance distribution
    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(test_rewards, bins=12, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(test_rewards), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(test_rewards):.1f}')
    ax2.set_xlabel('Episode Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('BC Agent Performance Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(f'bc_training_{env_name}.png', dpi=150, bbox_inches='tight')
    print(f"BC training plot saved: bc_training_{env_name}.png")
    plt.close()

    env.close()

    print(f"\nBehavior Cloning training complete!")
    print(f"Average test reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Max test reward: {np.max(test_rewards)}")

    return agent


def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return train_bc(env_name)


if __name__ == "__main__":
    train_bc()