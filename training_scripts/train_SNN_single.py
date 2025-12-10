import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from models.SNN_single import SNN_SingleStep

"""
To run training, paste the following command into repo root after data collection:
python -m training_scripts.train_SNN_single
"""

# Configurations
batch_size = 64
learning_rate = 1e-3
num_epochs = 10
train_fraction = 0.8
seed = 0
# Configurations

data_path = Path("raw_data/cartpole_trajectories.pt")
results_dir = Path("raw_data")
results_dir.mkdir(parents=True, exist_ok=True)


class CartPoleDataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(path)
        self.states = data["states"].float()
        self.actions = data["actions"].long()
        assert self.states.shape[0] == self.actions.shape[0]

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
# Data set definition


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identify_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(model, device, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct_ids = 0
    total_ids = 0

    for states, actions in data_loader:
        states = states.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        logits = model(states)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * states.size(0)
        predicted = torch.argmax(logits, dim=1)
        correct_ids += (predicted == actions).sum().item()
        total_ids += actions.size(0)

    avg_loss = total_loss / total_ids
    accuracy = correct_ids / total_ids
    return avg_loss, accuracy


def evaluate(model, device, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_ids = 0
    total_ids = 0

    with torch.no_grad():
        for states, actions in data_loader:
            states = states.to(device)
            actions = actions.to(device)

            logits = model(states)
            loss = criterion(logits, actions)

            total_loss += loss.item() * states.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct_ids += (predicted == actions).sum().item()
            total_ids += actions.size(0)

    avg_loss = total_loss / total_ids
    accuracy = correct_ids / total_ids
    return avg_loss, accuracy


def main():
    set_seed(seed)
    device = identify_device()
    print(f"Training SNN on: {device}")

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
    full_dataset = CartPoleDataset(data_path)
    n_total = len(full_dataset)
    n_train = int(train_fraction * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset size: {n_total} (train={n_train}, val={n_val})")

    model = SNN_SingleStep(state_in=4, hidden_lay=64, action_out=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    model_path = results_dir / "snn_single_step_cartpole.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved SNN policy to: {model_path}")
    # Save model

    curves_path = results_dir / "snn_single_step_training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            },
            f,
        )
    print(f"Saved training curves to: {curves_path}")
    # Save curves


if __name__ == "__main__":
    main()
