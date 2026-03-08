"""
predict.py
==========
Stage 2 and 3 of 3 in the grasping pipeline.

    [1] grasp_point_generator.py  ->  (x, y) contact point
    [2] LSTMPredictor             ->  theta motor angle      <- THIS FILE
    [3] SerialBridge              ->  Arduino serial         <- THIS FILE

Usage as module (called by pipeline.py):
    from predict import LSTMPredictor, SerialBridge

Usage standalone (no hardware):
    python predict.py --dry-run --x 22.0 --y -37.0

Usage with hardware:
    python predict.py --port COM5 --x 22.0 --y -37.0
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model — single definition shared with train.py
# ---------------------------------------------------------------------------

class LSTMInverseKinematics(nn.Module):
    """
    LSTM inverse kinematics observer for the UMoBIC-Finger.
    Maps fingertip (x, y) in mm to motor pulley angle theta in degrees.
    Trained on ArUco motion capture data. R2 = 0.9998.
    """

    def __init__(self, input_size: int = 2,
                 hidden_size: int = 64,
                 output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Stage 2 — LSTM Predictor
# ---------------------------------------------------------------------------

class LSTMPredictor:
    """
    Loads pretrained LSTM weights and runs inference.

    Args:
        weights_path: Path to .pth file.
        device:       'cpu' or 'cuda'. Auto-detected if not given.
    """

    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = LSTMInverseKinematics()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"[LSTMPredictor] Loaded '{os.path.basename(weights_path)}' on {self.device}")

    def predict(self, x: float, y: float) -> float:
        """
        Predict motor angle for one fingertip position.

        Args:
            x: Fingertip x in mm (palm frame, ArUco calibrated).
            y: Fingertip y in mm.

        Returns:
            theta: Motor pulley angle in degrees.
        """
        inp = torch.tensor([[[x, y]]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.model(inp).item())

    def predict_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Predict angles for multiple (x, y) points.
        points: (N, 2) array. Returns (N,) array of degrees.
        """
        inp = torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(1)
        with torch.no_grad():
            return self.model(inp).cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Stage 3 — Serial Bridge
# Swap this class to change transport layer (ROS, socket, CAN).
# pipeline.py only calls send_angle() and read_status().
# ---------------------------------------------------------------------------

class SerialBridge:
    """
    Sends motor angle to the Arduino impedance controller over USB serial.

    Protocol:
        Python -> Arduino:  "<angle>\n"           e.g. "142.50\n"
        Arduino -> Python:  comma-separated CSV   e.g. "142.50,138.2,0.3,87.1,2.1,0.04"
                            fields: desired, current, pressure, torque, velocity, accel

    Args:
        port:     'COM5' on Windows, '/dev/ttyUSB0' on Linux/Mac.
        baudrate: Must match Serial.begin() in the .ino file. Default 115200.
        timeout:  Read timeout in seconds. Default 0.1.
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.1):
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial not installed. Run: pip install pyserial")
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self._ser: Optional[serial.Serial] = None

    def connect(self) -> None:
        """Open serial port. Waits 2 s for Arduino hardware reset on connect."""
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2.0)
        print(f"[SerialBridge] Connected  {self.port}  @ {self.baudrate} baud")

    def disconnect(self) -> None:
        """Close serial port."""
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[SerialBridge] Disconnected")

    def send_angle(self, angle: float) -> None:
        """
        Send target motor angle to the Arduino.
        The firmware reads this in readDesiredAngle() each loop cycle.

        Args:
            angle: Motor pulley target in degrees (range 0-360).
        """
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("[SerialBridge] Not connected. Call connect() first.")
        self._ser.write(f"{angle:.2f}\n".encode("utf-8"))

    def read_status(self) -> Optional[str]:
        """
        Read one status line from Arduino (non-blocking).
        Returns comma-separated string or None if no data.
        """
        if self._ser and self._ser.in_waiting > 0:
            try:
                return self._ser.readline().decode("utf-8").strip()
            except Exception:
                return None
        return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

_HERE         = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_HERE, "lstm", "weights", "lstm_inverse_kinematics.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM inference: (x,y) -> theta -> Arduino")
    parser.add_argument("--port",    type=str,   default="COM5")
    parser.add_argument("--weights", type=str,   default=_WEIGHTS_PATH)
    parser.add_argument("--x",       type=float, required=True)
    parser.add_argument("--y",       type=float, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip serial transmission (no hardware needed)")
    args = parser.parse_args()

    predictor = LSTMPredictor(args.weights)
    theta     = predictor.predict(args.x, args.y)
    print(f"[Predict]  x={args.x:.2f}  y={args.y:.2f}  ->  theta={theta:.2f} deg")

    if args.dry_run:
        print("[DryRun] Serial skipped.")
    else:
        with SerialBridge(port=args.port) as bridge:
            bridge.send_angle(theta)
            time.sleep(0.05)
            status = bridge.read_status()
            if status:
                print(f"[Arduino] {status}")
