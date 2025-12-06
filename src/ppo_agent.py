import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['learning_rate'])

        self.gamma = config['gamma']
        self.clip_epsilon = config['clip_epsilon']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        self.batch_size = config['batch_size']
        self.ppo_epochs = config['ppo_epochs']

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, value = self.policy(state)
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), value.item(), log_prob.item()

    def update(self, states, actions, old_log_probs, advantages, returns):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)

        for _ in range(self.ppo_epochs):
            action_logits, values = self.policy(states)
            dist = torch.distributions.Categorical(logits=action_logits)

            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values.squeeze(), returns)

            # Entropy
            entropy = dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        return loss.item()

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t] * (1 - dones[t])
            else:
                next_value = values[t + 1] * (1 - dones[t])

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t])
            advantages.insert(0, gae)

        return np.array(advantages)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))