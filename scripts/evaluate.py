import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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


def evaluate_all(env_name='CartPole-v1'):
    """Evaluate all trained agents"""
    print("=" * 60)
    print("📈 FINAL EVALUATION")
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
    print("\n1️⃣ Testing BC-only...")
    bc_path = f"bc_{env_name}.pth"
    if os.path.exists(bc_path):
        # Create BC agent
        class BehaviorCloningAgent:
            def __init__(self, state_dim, action_dim):
                self.device = torch.device("cpu")
                self.policy_network = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim)
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
        print(f"  ⚠️ BC model not found")

    # Test Pure RL
    print("\n2️⃣ Testing Pure RL...")
    pure_rl_path = f"pure_rl_{env_name}.pth"
    if os.path.exists(pure_rl_path):
        # Create PPO network
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

        pure_rl_model = PPONetwork(state_dim, action_dim)
        pure_rl_model.load_state_dict(torch.load(pure_rl_path, map_location='cpu'))
        pure_rl_model.eval()

        pure_rl_rewards = evaluate_agent(pure_rl_model, env, 20)
        results['Pure RL'] = pure_rl_rewards
        print(f"  Avg reward: {np.mean(pure_rl_rewards):.2f} ± {np.std(pure_rl_rewards):.2f}")
    else:
        print(f"  ⚠️ Pure RL model not found")

    # Test BC→RL
    print("\n3️⃣ Testing BC→RL...")
    bc_rl_path = f"bc_rl_{env_name}.pth"
    if os.path.exists(bc_rl_path):
        # Create PPO network
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

        bc_rl_model = PPONetwork(state_dim, action_dim)
        bc_rl_model.load_state_dict(torch.load(bc_rl_path, map_location='cpu'))
        bc_rl_model.eval()

        bc_rl_rewards = evaluate_agent(bc_rl_model, env, 20)
        results['BC→RL'] = bc_rl_rewards
        print(f"  Avg reward: {np.mean(bc_rl_rewards):.2f} ± {np.std(bc_rl_rewards):.2f}")
    else:
        print(f"  ⚠️ BC→RL model not found")

    env.close()

    # Create comparison plot if we have results
    if results:
        plt.figure(figsize=(12, 5))

        # Box plot
        plt.subplot(1, 2, 1)
        labels = list(results.keys())
        data = [results[label] for label in labels]

        plt.boxplot(data, labels=labels)
        plt.ylabel('Episode Reward')
        plt.title('Performance Distribution')
        plt.grid(True, alpha=0.3)

        # Bar plot with error