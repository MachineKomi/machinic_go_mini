"""Self-play training system inspired by AlphaZero."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import json
import time
from collections import deque

from .game import GoGame, Move, Stone, format_move
from .network import AlphaGoNetwork, create_network
from .mcts import MCTS


@dataclass
class TrainingExample:
    """Single training example from self-play."""
    features: np.ndarray  # Board features
    policy: np.ndarray    # MCTS policy target
    value: float          # Game outcome from this player's perspective


@dataclass
class GameRecord:
    """Record of a complete self-play game."""
    moves: List[Move]
    policies: List[np.ndarray]
    winner: Optional[Stone]
    black_score: float
    white_score: float


class ReplayBuffer:
    """Circular buffer for training examples."""
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples: List[TrainingExample]):
        """Add examples to buffer."""
        self.buffer.extend(examples)
    
    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: Path):
        """Save buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "features": [ex.features for ex in self.buffer],
            "policies": [ex.policy for ex in self.buffer],
            "values": [ex.value for ex in self.buffer],
        }
        np.savez_compressed(path, **data)
    
    def load(self, path: Path):
        """Load buffer from disk."""
        data = np.load(path)
        self.buffer.clear()
        
        for i in range(len(data["values"])):
            self.buffer.append(TrainingExample(
                features=data["features"][i],
                policy=data["policies"][i],
                value=data["values"][i],
            ))


class SelfPlayWorker:
    """Generates self-play games for training."""
    
    def __init__(
        self,
        network: AlphaGoNetwork,
        board_size: int = 9,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature_threshold: int = 30,
        callback: Optional[Callable] = None,
    ):
        self.network = network
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature_threshold = temperature_threshold
        self.callback = callback  # Called after each move
    
    def play_game(self) -> Tuple[GameRecord, List[TrainingExample]]:
        """Play a complete self-play game.
        
        Returns:
            record: Game record with moves and outcome
            examples: Training examples from the game
        """
        game = GoGame(board_size=self.board_size)
        
        moves = []
        policies = []
        features_history = []
        players = []  # Track which player made each move
        
        move_count = 0
        max_moves = self.board_size * self.board_size * 3  # Prevent infinite games
        
        while not game.is_game_over() and move_count < max_moves:
            # Use temperature for exploration in early game
            temperature = 1.0 if move_count < self.temperature_threshold else 0.1
            
            mcts = MCTS(
                network=self.network,
                num_simulations=self.num_simulations,
                c_puct=self.c_puct,
                temperature=temperature,
            )
            
            # Get move from MCTS
            move, policy, value = mcts.get_move(game, add_noise=True)
            
            # Store for training
            features_history.append(game.to_features())
            policies.append(policy)
            players.append(game.current_player)
            
            # Play move
            game.play_move(move)
            moves.append(move)
            move_count += 1
            
            # Callback for visualization
            if self.callback:
                self.callback(game, move, policy, value, move_count)
        
        # Get game result
        winner = game.get_winner()
        black_score, white_score = game.get_score()
        
        # Create training examples with correct values
        examples = []
        for features, policy, player in zip(features_history, policies, players):
            # Value from this player's perspective
            if winner is None:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            examples.append(TrainingExample(
                features=features,
                policy=policy,
                value=value,
            ))
            
            # Data augmentation: add rotations and reflections
            examples.extend(self._augment_example(features, policy, value))
        
        record = GameRecord(
            moves=moves,
            policies=policies,
            winner=winner,
            black_score=black_score,
            white_score=white_score,
        )
        
        return record, examples

    def _augment_example(
        self, features: np.ndarray, policy: np.ndarray, value: float
    ) -> List[TrainingExample]:
        """Create augmented examples through rotations and reflections."""
        examples = []
        board_size = features.shape[1]
        
        # Reshape policy to board (excluding pass)
        policy_board = policy[:-1].reshape(board_size, board_size)
        pass_prob = policy[-1]
        
        # 4 rotations x 2 reflections = 8 symmetries (minus original = 7)
        for k in range(1, 4):  # 90, 180, 270 degree rotations
            rot_features = np.rot90(features, k, axes=(1, 2)).copy()
            rot_policy_board = np.rot90(policy_board, k).copy()
            rot_policy = np.append(rot_policy_board.flatten(), pass_prob)
            
            examples.append(TrainingExample(
                features=rot_features,
                policy=rot_policy,
                value=value,
            ))
        
        # Horizontal flip
        flip_features = np.flip(features, axis=2).copy()
        flip_policy_board = np.flip(policy_board, axis=1).copy()
        flip_policy = np.append(flip_policy_board.flatten(), pass_prob)
        
        examples.append(TrainingExample(
            features=flip_features,
            policy=flip_policy,
            value=value,
        ))
        
        # Flip + rotations
        for k in range(1, 4):
            rot_features = np.rot90(flip_features, k, axes=(1, 2)).copy()
            rot_policy_board = np.rot90(flip_policy_board, k).copy()
            rot_policy = np.append(rot_policy_board.flatten(), pass_prob)
            
            examples.append(TrainingExample(
                features=rot_features,
                policy=rot_policy,
                value=value,
            ))
        
        return examples


class Trainer:
    """Neural network trainer."""
    
    def __init__(
        self,
        network: AlphaGoNetwork,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.network = network
        self.device = device
        self.network.to(device)
        
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )

    def train_batch(self, examples: List[TrainingExample]) -> dict:
        """Train on a batch of examples.
        
        Returns dict with loss values.
        """
        self.network.train()
        
        # Prepare batch
        features = np.stack([ex.features for ex in examples])
        target_policies = np.stack([ex.policy for ex in examples])
        target_values = np.array([ex.value for ex in examples])
        
        features = torch.from_numpy(features).to(self.device)
        target_policies = torch.from_numpy(target_policies).to(self.device)
        target_values = torch.from_numpy(target_values).float().to(self.device)
        
        # Forward pass
        policy_logits, values = self.network(features)
        
        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_logits, target_policies)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(), target_values)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
    
    def train_epoch(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        batches_per_epoch: int = 100,
    ) -> dict:
        """Train for one epoch on replay buffer."""
        total_losses = {"total_loss": 0, "policy_loss": 0, "value_loss": 0}
        
        for _ in range(batches_per_epoch):
            examples = replay_buffer.sample(batch_size)
            losses = self.train_batch(examples)
            
            for key in total_losses:
                total_losses[key] += losses[key]
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= batches_per_epoch
        
        self.scheduler.step()
        
        return total_losses


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    board_size: int = 9
    residual_channels: int = 64
    num_residual_blocks: int = 4
    num_simulations: int = 100
    games_per_iteration: int = 10
    batch_size: int = 256
    batches_per_epoch: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 50000
    checkpoint_interval: int = 5
    num_iterations: int = 100
    device: str = "cpu"
    output_dir: str = "checkpoints"
    
    def save(self, path: Path):
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load config from JSON."""
        with open(path) as f:
            return cls(**json.load(f))


class TrainingPipeline:
    """Complete training pipeline with self-play and learning."""
    
    def __init__(
        self,
        config: TrainingConfig,
        game_callback: Optional[Callable] = None,
        iteration_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.game_callback = game_callback
        self.iteration_callback = iteration_callback
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(self.output_dir / "config.json")
        
        # Initialize network
        self.network = create_network(
            board_size=config.board_size,
            residual_channels=config.residual_channels,
            num_residual_blocks=config.num_residual_blocks,
            device=config.device,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            network=self.network,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=config.device,
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
        
        # Training stats
        self.iteration = 0
        self.total_games = 0
        self.stats_history = []
    
    def run_self_play(self, num_games: int) -> List[GameRecord]:
        """Run self-play games and add to replay buffer."""
        records = []
        
        for game_num in range(num_games):
            worker = SelfPlayWorker(
                network=self.network,
                board_size=self.config.board_size,
                num_simulations=self.config.num_simulations,
                callback=self.game_callback,
            )
            
            record, examples = worker.play_game()
            records.append(record)
            self.replay_buffer.add(examples)
            self.total_games += 1
        
        return records
    
    def train_iteration(self) -> dict:
        """Run one training iteration."""
        self.iteration += 1
        start_time = time.time()
        
        # Self-play phase
        records = self.run_self_play(self.config.games_per_iteration)
        selfplay_time = time.time() - start_time
        
        # Training phase
        train_start = time.time()
        losses = self.trainer.train_epoch(
            self.replay_buffer,
            batch_size=self.config.batch_size,
            batches_per_epoch=self.config.batches_per_epoch,
        )
        train_time = time.time() - train_start
        
        # Compute stats
        wins = sum(1 for r in records if r.winner == Stone.BLACK)
        avg_moves = np.mean([len(r.moves) for r in records])
        
        stats = {
            "iteration": self.iteration,
            "total_games": self.total_games,
            "buffer_size": len(self.replay_buffer),
            "black_wins": wins,
            "white_wins": len(records) - wins,
            "avg_moves": avg_moves,
            "selfplay_time": selfplay_time,
            "train_time": train_time,
            **losses,
        }
        self.stats_history.append(stats)
        
        # Save checkpoint
        if self.iteration % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
        
        # Callback
        if self.iteration_callback:
            self.iteration_callback(stats, records)
        
        return stats

    def save_checkpoint(self):
        """Save current checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{self.iteration:04d}.pt"
        self.network.save(checkpoint_path)
        
        # Save latest as well
        latest_path = self.output_dir / "latest.pt"
        self.network.save(latest_path)
        
        # Save stats
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats_history, f, indent=2)
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint and continue training."""
        self.network = AlphaGoNetwork.load(path, device=self.config.device)
        self.trainer = Trainer(
            network=self.network,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            device=self.config.device,
        )
        
        # Try to load stats
        stats_path = self.output_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.stats_history = json.load(f)
                if self.stats_history:
                    self.iteration = self.stats_history[-1]["iteration"]
                    self.total_games = self.stats_history[-1]["total_games"]
    
    def run(self, num_iterations: Optional[int] = None):
        """Run training for specified iterations."""
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        for _ in range(num_iterations):
            self.train_iteration()


def get_available_checkpoints(checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    """Get list of available checkpoints sorted by iteration."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            iteration = int(path.stem.split("_")[1])
            checkpoints.append((iteration, path))
        except (ValueError, IndexError):
            continue
    
    return sorted(checkpoints, key=lambda x: x[0])
