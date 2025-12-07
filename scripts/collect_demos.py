import sys
import os

# Force CPU-only mode to avoid CUDA library issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import torch
import torch.nn as nn
from src.seed_utils import set_seed


def collect_demos(env_name='CartPole-v1', num_episodes=50, seed=42):
    """Collect expert demonstrations"""
    # Set seeds for reproducibility
    set_seed(seed)
    
    print(f"Collecting Demonstrations from {env_name} (seed={seed})")

    # Try to import EnvironmentWrapper
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

    # Initialize environment
    env = EnvironmentWrapper(env_name)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Load expert model
    expert_path = f"expert_{env_name}.pth"

    if not os.path.exists(expert_path):
        print(f"ERROR: Expert model not found: {expert_path}")
        print("Please run train_expert.py first!")
        return

    # Create a simple QNetwork to load the expert (must match the architecture used during training)
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

    # Load expert model
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(expert_path, map_location='cpu'))
    model.eval()

    print(f"Loaded expert model from {expert_path}")

    # Collect demonstrations
    demonstrations = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        while True:
            # Get expert action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            episode_data['states'].append(state.copy())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['next_states'].append(next_state.copy())
            episode_data['dones'].append(done)

            state = next_state

            if done:
                break

        demonstrations.append(episode_data)

        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1}/{num_episodes} episodes")

    # Save demonstrations
    demo_path = f"demos_{env_name}.pkl"
    with open(demo_path, 'wb') as f:
        pickle.dump(demonstrations, f)

    env.close()

    # Statistics
    total_transitions = sum(len(d['states']) for d in demonstrations)
    avg_episode_length = total_transitions / len(demonstrations)

    print(f"\nDemonstrations collected successfully!")
    print(f"Total episodes: {len(demonstrations)}")
    print(f"Total transitions: {total_transitions}")
    print(f"Average episode length: {avg_episode_length:.1f}")
    print(f"Saved to: {demo_path}")

    return demonstrations


def main(env_name='CartPole-v1'):
    """Main function called by main.py"""
    return collect_demos(env_name)


if __name__ == "__main__":
    collect_demos()