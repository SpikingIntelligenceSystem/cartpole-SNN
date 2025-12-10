import torch
import torch.nn as nn
import snntorch as snn


class SNN_SingleStep(nn.Module):
    def __init__(self, state_in: int = 4, hidden_lay: int = 64, action_out: int = 2, beta: float = 0.9):
        super().__init__()
        self.fc1 = nn.Linear(state_in, hidden_lay)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_lay, hidden_lay)
        self.lif2 = snn.Leaky(beta=beta)

        self.fc_out = nn.Linear(hidden_lay, action_out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        cur1 = self.fc1(state)
        mem1 = torch.zeros_like(cur1)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        mem2 = torch.zeros_like(cur2)
        spk2, mem2 = self.lif2(cur2, mem2)

        logits = self.fc_out(spk2)
        return logits

    def action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        logits = self.forward(state)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())
