import torch
import torch.nn as nn


class MLPBase(nn.Module):
    def __init__(self, state_in=4, hidden_lay=64, action_out=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_in, hidden_lay),
            nn.ReLU(),
            nn.Linear(hidden_lay, hidden_lay),
            nn.ReLU(),
            nn.Linear(hidden_lay, action_out),
        )

    def forward(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state)

    def action(self, state: torch.Tensor, deterministic=True):
        logits = self.forward(state)
        if deterministic:
            return torch.argmax(logits, dim=-1).item()
        else:
            probabilities = torch.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(probabilities)
            action = distribution.sample().item()
        return action
