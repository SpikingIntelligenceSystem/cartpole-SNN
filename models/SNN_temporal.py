import torch
import torch.nn as nn
import snntorch as snn

class SNN_Temporal(nn.Module):
    def __init__(self, state_in, hidden_lay, action_out, beta=0.9):
        super().__init__()

        self.fc1 = nn.Linear(state_in, hidden_lay)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_lay, hidden_lay)
        self.lif2 = snn.Leaky(beta=beta)

        self.fc_output = nn.Linear(hidden_lay, action_out)
    
    def forward(self, spikes: torch.Tensor):
        if spikes.dim() == 2:
            spikes = spikes.unsqueeze(0)
        elif spikes.dim() == 1:
            spikes = spikes.unsqueeze(0).unsqueeze(1)
        if spikes.dim() != 3:
            raise ValueError(f"Input spike format error. Expected shape [T, batch_size, state_dim], got shape {spikes.shape}.")
        # Formats spikes to [T, batch_size, state_dim] if not already in format.

        T, batch_size, _ = spikes.shape
        device = spikes.device

        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=device)

        out_sum = torch.zeros(batch_size, self.fc_output.out_features, device=device)

        for t in range(T):
            x_t = spikes[t]
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            out_t = self.fc_output(spk2)
            out_sum += out_t
        logits = out_sum / T
        return logits

    def action(self, state: torch.Tensor, num_steps, device):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        spikes = spike_encoding(state, num_steps, device)
        with torch.no_grad():
            logits = self.forward(spikes)
            action = logits.argmax(dim=-1).item()
        return action

def spike_encoding(states: torch.Tensor, num_steps, device):
    states = states.to(device)
    states_norm = torch.tanh(states)
    spikes = states_norm.unsqueeze(0).expand(num_steps, -1, -1)
    return spikes
# Spike encoding also used in certain training scripts

    

