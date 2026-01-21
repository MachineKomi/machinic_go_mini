"""Neural network architecture inspired by AlphaZero/KataGo."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import json


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class PolicyHead(nn.Module):
    """Policy head outputs move probabilities."""
    
    def __init__(self, channels: int, board_size: int):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        # Output: board_size^2 positions + 1 pass move
        self.fc = nn.Linear(32 * board_size * board_size, board_size * board_size + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # Raw logits, apply softmax externally


class ValueHead(nn.Module):
    """Value head outputs position evaluation."""
    
    def __init__(self, channels: int, board_size: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class AlphaGoNetwork(nn.Module):
    """AlphaZero-style neural network for Go."""
    
    def __init__(
        self,
        board_size: int = 9,
        input_channels: int = 17,
        residual_channels: int = 128,
        num_residual_blocks: int = 6,
    ):
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.num_residual_blocks = num_residual_blocks
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, residual_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(residual_channels)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(residual_channels) for _ in range(num_residual_blocks)
        ])
        
        # Output heads
        self.policy_head = PolicyHead(residual_channels, board_size)
        self.value_head = ValueHead(residual_channels, board_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (policy_logits, value)."""
        # Input convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Output heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for a single position.
        
        Args:
            features: Board features of shape (C, H, W)
            
        Returns:
            policy: Move probabilities of shape (board_size^2 + 1,)
            value: Position evaluation in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            policy_logits, value = self(x)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
        
        return policy, value
    
    def predict_batch(self, features_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict policy and value for a batch of positions."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features_batch)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            policy_logits, values = self(x)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy()[:, 0]
        
        return policies, values
    
    def get_config(self) -> dict:
        """Get network configuration."""
        return {
            "board_size": self.board_size,
            "input_channels": self.input_channels,
            "residual_channels": self.residual_channels,
            "num_residual_blocks": self.num_residual_blocks,
        }
    
    def save(self, path: Path):
        """Save network weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "AlphaGoNetwork":
        """Load network from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        network = cls(**config)
        network.load_state_dict(checkpoint["state_dict"])
        network.to(device)
        
        return network


def create_network(
    board_size: int = 9,
    residual_channels: int = 128,
    num_residual_blocks: int = 6,
    device: str = "cpu",
) -> AlphaGoNetwork:
    """Create a new network with given configuration."""
    network = AlphaGoNetwork(
        board_size=board_size,
        residual_channels=residual_channels,
        num_residual_blocks=num_residual_blocks,
    )
    network.to(device)
    return network
