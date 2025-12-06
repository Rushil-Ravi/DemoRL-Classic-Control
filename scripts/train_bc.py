import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def train_bc(env_name='CartPole-v1'):
    """Train Behavior Cloning agent"""
    print(f"🎓 Training Behavior Cloning on {env_name}")

    # Load demonstrations
    demo_path = f"demos_{env_name}.pkl"
    if not os.path.exists(demo_path):
        print(f"❌ Demonstrations not found: {demo_path}")
        print("Please run collect_demos.py first!")
        return

    with open(demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    print(f"✅ Loaded {len(demonstrations)} demonstrations")

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

            # Simple policy network
            self.policy_network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
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
    print(f"💾 BC model saved: {model_path}")

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Behavior Cloning Training Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(test_rewards, bins=10, alpha=0.7)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title(f'BC Performance\nMean: {np.mean(test_rewards):.1f}, Std: {np.std(test_rewards):.1f}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'bc_training_{env_name}.png')
    plt.show()

    env.close()

    print(f"\n✅ Behavior Cloning complete!")
    print(f"Average test reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Max test reward: {np.max(test_rewards)}")

    return agent


def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return train_bc(env_name)


if __name__ == "__main__":
    train_bc()