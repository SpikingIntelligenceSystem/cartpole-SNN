import torch
import torch.nn as nn
import snntorch as snn


class SNN_SingleStep(nn.Module):
    def __init__(self, state_in=4, hidden_lay=64, action_out=2, beta=0.9):
        super().__init__()

        self.fc1 = nn.Linear(state_in, hidden_lay)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_lay, action_out)
        self.lif2 = snn.Leaky(beta=beta)

        self.fc_out = nn.Linear(hidden_lay, action_out)

    def forward(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        cur1 = self.fc1(state)
        spk1, _ = self.lif1(cur1)

        cur2 = self.fc2(spk1)
        spk2, _ = self.lif2(cur2)

        logits = self.fc_out(spk2)
        return logits

    def action(self, state: torch.Tensor, deterministic=True):
        logits = self.forward(state)
        if deterministic:
            return torch.argmax(logits, dim=-1).item()
        else:
            probabilities = torch.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(probabilities)
            action = distribution.sample().item()
        return action
