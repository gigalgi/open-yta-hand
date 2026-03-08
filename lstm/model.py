"""
UMoBIC-Finger LSTM Joint Angle Observer
========================================
Maps fingertip position (x, y) → pulley motor angle (θ).

Trained on ArUco motion capture data:
  - ~8000 samples across full range of motion
  - Input:  fingertip (x, y) in mm, measured via Logitech C270 + ArUco markers
  - Output: θ_motor in degrees, measured via AS5600 magnetic encoder

Validation metrics:
  R² = 0.9998 | MSE = 0.6684 | RMSE = 0.8297 | MAE = 0.4291

Architecture:
  (batch, seq=1, 2) → LSTM(64) → Linear → (batch, 1)

Reference:
  Galvis Giraldo, G. — Master's Thesis, SKKU 2024
  https://dcollection.skku.edu/srch/srchDetail/000000181091
"""

import torch
import torch.nn as nn


class LSTMInverseKinematics(nn.Module):
    """
    LSTM-based inverse kinematics observer for the UMoBIC-Finger.

    The cable-driven underactuated design produces nonlinear joint coupling
    that cannot be captured analytically. This network learns the mapping
    from observed fingertip position to motor angle from real hardware data.

    Args:
        input_size  (int): Number of input features. Default: 2 (x, y).
        hidden_size (int): LSTM hidden units. Default: 64.
        output_size (int): Number of output values. Default: 1 (θ_motor).
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size).
               For single-step inference use seq_len=1.

        Returns:
            Tensor of shape (batch, output_size) — predicted motor angle in degrees.
        """
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def load_model(weights_path: str = "weights/lstm_inverse_kinematics.pth") -> LSTMInverseKinematics:
    """Load pretrained model from weights file."""
    model = LSTMInverseKinematics()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model
