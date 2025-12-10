import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from models.MLP_baseline import MLPBase

"""
To run training, paste the following command into repo root after data collection:
python -m training_scripts.train_MLP_baseline
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


class cartpoleDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.states = data["states"].float()
        self.actions = data["actions"].long()
        assert self.states.shape[0] == self.actions.shape[0]

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
# Data set definition


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identify_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print(f"Training on: {device}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Please run trajectory_collection.py first.")
    dataset = cartpoleDataset(data_path)
    num_total = len(dataset)
    num_train = int(train_fraction * num_total)
    num_val = num_total - num_train

    train_dataset, val_dataset = random_split(
        dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(seed),
    )
    training_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    model = MLPBase(state_in=4, hidden_lay=64, action_out=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        training_loss, training_accuracy = train(
            model, device, training_loader, optimizer, criterion)

        validation_loss, validation_accuracy = evaluate(
            model, device, validation_loader, criterion)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"Training Loss: {training_loss:.4f}/// Training Accuracy: {training_accuracy*100:.2f}%")
        print(
            f"Validation Loss: {validation_loss:.4f}/// Validation Accuracy: {validation_accuracy*100:.2f}%")
    model_save_path = results_dir / "mlp_baseline_cartpole.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    curves_save_path = results_dir / "mlp_baseline_training_curves.json"
    with open(curves_save_path, "w") as f:
        json.dump(
            {
                "training_losses": training_losses,
                "validation_losses": validation_losses,
                "training_accuracies": training_accuracies,
                "validation_accuracies": validation_accuracies,
            },
            f,
        )
    print(f"Training curves saved to {curves_save_path}")
    # Saves training curves for later analysis


if __name__ == "__main__":
    main()
