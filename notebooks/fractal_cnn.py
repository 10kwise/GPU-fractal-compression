"""
fractal_cnn.py — CNN post-decode enhancement for fractal image compression

Loads pre-trained weights from fractal_cnn_weights.pt and provides a single
function cnn_enhance() that improves the Y channel output of decode_fractal().

Usage in the fractal notebook:
    from fractal_cnn import load_model, cnn_enhance

    model = load_model('fractal_cnn_weights.pt')  # once per session
    y_rec = decode_fractal(y_tf, (H_pad, W_pad), N_ITERATIONS)
    y_rec = cnn_enhance(y_rec, model)              # ~5ms on T4
"""

import os
import numpy as np
import torch
import torch.nn as nn


class EdgeAwareResidualCNN(nn.Module):
    """
    Lightweight residual CNN for fractal post-decode enhancement.

    Learns to predict and add back the high-frequency content
    (edges, texture) destroyed by fractal's 2× downsampling lowpass filter.

    Architecture:
        4 convolutional layers with residual learning.
        Input:  (B, 1, H, W)  — fractal-decoded Y channel
        Output: (B, 1, H, W)  — enhanced Y channel

    Parameters: ~37K (runs in <5ms per image on T4)
    """
    def __init__(self, n_channels=64, n_layers=4):
        super().__init__()
        layers = []
        # First layer: 1 input channel (Y) → n_channels
        layers.append(nn.Conv2d(1, n_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        # Final layer: n_channels → 1 (residual prediction, no activation)
        layers.append(nn.Conv2d(n_channels, 1, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Residual learning: predict the correction, add to input
        return x + self.net(x)


def load_model(weights_path='fractal_cnn_weights.pt', device=None):
    """
    Load pre-trained CNN model from weights file.

    Args:
        weights_path: Path to .pt weights file
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        model: EdgeAwareResidualCNN in eval mode
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EdgeAwareResidualCNN(n_channels=64, n_layers=4)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'[CNN] Loaded {n_params:,} params from {weights_path}')
    else:
        print(f'[CNN] WARNING: Weights not found at {weights_path}')
        print(f'[CNN] Model initialised with random weights — run training first!')

    model = model.to(device).eval()
    return model


@torch.no_grad()
def cnn_enhance(y_channel, model, device=None):
    """
    Enhance a fractal-decoded Y channel using the trained CNN.

    Args:
        y_channel: numpy array (H, W) uint8 — output of decode_fractal()
        model: loaded EdgeAwareResidualCNN (from load_model())
        device: 'cuda', 'cpu', or None (auto from model)

    Returns:
        enhanced: numpy array (H, W) uint8 — enhanced Y channel
    """
    if device is None:
        device = next(model.parameters()).device

    # Convert to tensor: (1, 1, H, W) float32 in [0, 1]
    y_float = y_channel.astype(np.float32) / 255.0
    tensor = torch.from_numpy(y_float).unsqueeze(0).unsqueeze(0).to(device)

    # Forward pass
    enhanced = model(tensor)

    # Convert back to uint8 numpy
    enhanced = enhanced.squeeze().cpu().numpy()
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

    return enhanced
