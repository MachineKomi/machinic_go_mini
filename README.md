# Machinic Go Mini

A miniature implementation of AlphaZero/KataGo for the game of Go, featuring:

- **Self-play training** with MCTS and neural network guidance
- **Modern TUI** for watching training games in real-time
- **Play against AI** at different checkpoints and difficulty levels
- **Watch AI vs AI** games

## Installation

```bash
cd machinic_go_mini
pip install -e .
```

## Quick Start

### Launch the TUI

```bash
machinic-go tui
```

This opens an interactive menu where you can:
- Start new training
- Continue existing training
- Play against trained checkpoints
- Watch AI vs AI games

### Command Line Training

```bash
# Start training with default settings (9x9 board)
machinic-go train

# Custom configuration
machinic-go train --board-size 9 --channels 64 --blocks 4 --simulations 100 --games 10 --iterations 100

# Resume training
machinic-go train --resume
```

### Play Against AI

```bash
# Play against latest checkpoint
machinic-go play checkpoints/latest.pt

# Play as white with harder AI
machinic-go play checkpoints/checkpoint_0050.pt --color white --simulations 400
```

### Evaluate Checkpoints

```bash
# Evaluate against random player
machinic-go evaluate checkpoints/latest.pt --games 20
```

## Architecture

### Neural Network
- ResNet-style architecture with configurable depth
- Policy head: outputs move probabilities
- Value head: outputs position evaluation [-1, 1]
- Input features: 17 planes (current stones, history, ko, valid moves)

### MCTS
- UCB-based tree search with neural network guidance
- Dirichlet noise at root for exploration during training
- Temperature-based move selection

### Training Pipeline
1. Self-play games generate training data
2. Data augmented with board symmetries (8x)
3. Network trained on replay buffer
4. Checkpoints saved periodically

## TUI Controls

### Training View
- `P` - Pause/Resume
- `F` - Faster playback
- `D` - Slower playback
- `S` - Save checkpoint
- `Q` - Quit

### Play View
- Enter moves like `D4`, `J10`, or `pass`
- `N` - New game
- `U` - Undo (limited)
- `P` - Pass
- `H` - Get hint from AI
- `Q` - Quit

### Watch View
- `Space` - Pause/Resume
- `F` - Faster
- `D` - Slower
- `N` - New game
- `Q` - Quit

## Configuration

Training parameters can be adjusted:

| Parameter | Default | Description |
|-----------|---------|-------------|
| board_size | 9 | Board size (9, 13, or 19) |
| residual_channels | 64 | Network width |
| num_residual_blocks | 4 | Network depth |
| num_simulations | 100 | MCTS simulations per move |
| games_per_iteration | 10 | Self-play games per iteration |
| batch_size | 256 | Training batch size |
| learning_rate | 0.001 | Adam learning rate |

## References

- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero)
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) (AlphaZero)
- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) (KataGo)
