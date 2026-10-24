import numpy as np
import torch
from torch import nn


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


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


def load_policy(path):
    ckpt = torch.load(path, map_location="cpu")
    model = Policy()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def heart_waypoints(n=90, scale=0.18, center=(0.0, 0.0)):
    # Parametric heart. Reverse direction so the start is near the upper indentation.
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    pts = np.column_stack([x, y]) * scale
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    return pts


def policy_state(x, y, theta, target):
    dx = float(target[0] - x)
    dy = float(target[1] - y)
    c, s = np.cos(theta), np.sin(theta)
    dx_b = c*dx + s*dy
    dy_b = -s*dx + c*dy
    dist = float(np.hypot(dx, dy))
    ang = float(wrap_angle(np.arctan2(dy, dx) - theta))
    return np.array([dx_b, dy_b, dist, ang], dtype=np.float32)
