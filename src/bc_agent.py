import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .networks import PolicyNetwork


class BehaviorCloningAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, states: np.ndarray, actions: np.ndarray, epochs: int = 50, batch_size: int = 32):
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)

        losses = []
        dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_states, batch_actions in dataloader:
                action_logits, _ = self.policy_network(batch_states)
                loss = self.criterion(action_logits, batch_actions)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"BC Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, _ = self.policy_network(state_tensor)
        return action_logits.argmax().item()