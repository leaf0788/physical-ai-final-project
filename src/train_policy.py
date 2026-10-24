from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
MODEL_PATH = OUT / "heart_policy.pt"
CURVE_PATH = OUT / "training_curve.png"


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def expert_action(state):
    """Simple expert controller used to generate imitation-learning labels.

    state = [dx_body, dy_body, distance, angle_error]
    output = [linear_velocity, angular_velocity]
    """
    dx_b, dy_b, dist, ang = state
    v = np.clip(1.8 * dist * max(np.cos(ang), 0.0), 0.0, 1.6)
    w = np.clip(4.2 * ang + 0.25 * dy_b, -3.2, 3.2)
    return np.array([v, w], dtype=np.float32)


def make_dataset(n=10000):
    states = []
    actions = []
    for _ in range(n):
        dx_b = np.random.uniform(-2.0, 2.0)
        dy_b = np.random.uniform(-2.0, 2.0)
        dist = float(np.hypot(dx_b, dy_b))
        # Keep angle_error consistent with body-frame target direction.
        ang = float(np.arctan2(dy_b, dx_b))
        s = np.array([dx_b, dy_b, dist, ang], dtype=np.float32)
        a = expert_action(s)
        states.append(s)
        actions.append(a)
    return np.stack(states), np.stack(actions)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        y = self.net(x)
        v = torch.sigmoid(y[..., 0:1]) * 1.6
        w = torch.tanh(y[..., 1:2]) * 3.2
        return torch.cat([v, w], dim=-1)


def main():
    X, Y = make_dataset()
    split = int(0.9 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(Y[:split]))
    val_x = torch.from_numpy(X[split:])
    val_y = torch.from_numpy(Y[split:])
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    model = Policy()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(1, 16):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            count += len(xb)
        tr = running / count
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(val_x), val_y).item()
        train_losses.append(tr)
        val_losses.append(va)
        if epoch in (1, 5, 10, 15):
            print(f"epoch {epoch:02d}: train={tr:.6f} val={va:.6f}")

    torch.save({"state_dict": model.state_dict(), "seed": SEED}, MODEL_PATH)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 16), train_losses, label="train")
    plt.plot(range(1, 16), val_losses, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Imitation learning: training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=160)
    print(f"saved: {MODEL_PATH}")
    print(f"saved: {CURVE_PATH}")


if __name__ == "__main__":
    main()
