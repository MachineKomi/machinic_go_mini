"""Command-line interface for Machinic Go Mini."""

import click
from pathlib import Path
import torch


@click.group()
def main():
    """Machinic Go Mini - Self-play training for Go."""
    pass


@main.command()
@click.option("--checkpoint-dir", "-c", default="checkpoints", help="Directory for checkpoints")
def tui(checkpoint_dir: str):
    """Launch the interactive TUI."""
    from .tui.app import MachinicGoApp
    
    app = MachinicGoApp(checkpoint_dir=Path(checkpoint_dir))
    app.run()


@main.command()
@click.option("--board-size", "-b", default=9, type=int, help="Board size (9, 13, or 19)")
@click.option("--channels", "-ch", default=64, type=int, help="Residual channels")
@click.option("--blocks", "-bl", default=4, type=int, help="Number of residual blocks")
@click.option("--simulations", "-s", default=100, type=int, help="MCTS simulations per move")
@click.option("--games", "-g", default=10, type=int, help="Games per training iteration")
@click.option("--iterations", "-i", default=100, type=int, help="Number of training iterations")
@click.option("--output", "-o", default="checkpoints", help="Output directory")
@click.option("--resume", "-r", is_flag=True, help="Resume from latest checkpoint")
def train(board_size, channels, blocks, simulations, games, iterations, output, resume):
    """Train the network through self-play (headless mode)."""
    from .training import TrainingConfig, TrainingPipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")
    
    config = TrainingConfig(
        board_size=board_size,
        residual_channels=channels,
        num_residual_blocks=blocks,
        num_simulations=simulations,
        games_per_iteration=games,
        num_iterations=iterations,
        device=device,
        output_dir=output,
    )
    
    def iteration_callback(stats, records):
        click.echo(
            f"Iteration {stats['iteration']}: "
            f"games={stats['total_games']}, "
            f"loss={stats['total_loss']:.4f}, "
            f"B/W={stats['black_wins']}/{stats['white_wins']}"
        )
    
    pipeline = TrainingPipeline(config, iteration_callback=iteration_callback)
    
    if resume:
        latest = Path(output) / "latest.pt"
        if latest.exists():
            click.echo(f"Resuming from {latest}")
            pipeline.load_checkpoint(latest)
        else:
            click.echo("No checkpoint found, starting fresh")
    
    click.echo(f"Starting training for {iterations} iterations...")
    pipeline.run(iterations)
    click.echo("Training complete!")


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--board-size", "-b", default=9, type=int, help="Board size")
@click.option("--simulations", "-s", default=200, type=int, help="MCTS simulations")
@click.option("--color", "-c", default="black", type=click.Choice(["black", "white"]))
def play(checkpoint, board_size, simulations, color):
    """Play against a trained checkpoint (TUI mode)."""
    from .tui.app import MachinicGoApp
    from .tui.play_view import PlayView
    from .game import Stone
    
    player_color = Stone.BLACK if color == "black" else Stone.WHITE
    
    app = MachinicGoApp()
    app.push_screen(PlayView(
        checkpoint_path=Path(checkpoint),
        board_size=board_size,
        num_simulations=simulations,
        player_color=player_color,
    ))
    app.run()


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--board-size", "-b", default=9, type=int, help="Board size")
@click.option("--simulations", "-s", default=200, type=int, help="MCTS simulations")
@click.option("--games", "-g", default=10, type=int, help="Number of games to play")
def evaluate(checkpoint, board_size, simulations, games):
    """Evaluate a checkpoint by playing against random."""
    from .network import AlphaGoNetwork
    from .mcts import MCTS, RandomPlayer
    from .game import GoGame, Stone
    
    click.echo(f"Loading checkpoint: {checkpoint}")
    network = AlphaGoNetwork.load(Path(checkpoint))
    network.eval()
    
    mcts = MCTS(network=network, num_simulations=simulations, temperature=0.1)
    random_player = RandomPlayer()
    
    wins = {"ai_black": 0, "ai_white": 0, "random_black": 0, "random_white": 0}
    
    for game_num in range(games):
        # AI plays Black
        game = GoGame(board_size=board_size)
        while not game.is_game_over():
            if game.current_player == Stone.BLACK:
                move, _, _ = mcts.get_move(game, add_noise=False)
            else:
                move = random_player.get_move(game)
            game.play_move(move)
        
        winner = game.get_winner()
        if winner == Stone.BLACK:
            wins["ai_black"] += 1
        else:
            wins["random_white"] += 1
        
        # AI plays White
        game = GoGame(board_size=board_size)
        while not game.is_game_over():
            if game.current_player == Stone.WHITE:
                move, _, _ = mcts.get_move(game, add_noise=False)
            else:
                move = random_player.get_move(game)
            game.play_move(move)
        
        winner = game.get_winner()
        if winner == Stone.WHITE:
            wins["ai_white"] += 1
        else:
            wins["random_black"] += 1
        
        click.echo(f"Game {game_num + 1}/{games}: AI wins {wins['ai_black'] + wins['ai_white']}/{(game_num + 1) * 2}")
    
    total_ai_wins = wins["ai_black"] + wins["ai_white"]
    total_games = games * 2
    click.echo(f"\nResults: AI won {total_ai_wins}/{total_games} ({100*total_ai_wins/total_games:.1f}%)")
    click.echo(f"  As Black: {wins['ai_black']}/{games}")
    click.echo(f"  As White: {wins['ai_white']}/{games}")


@main.command()
@click.option("--checkpoint-dir", "-c", default="checkpoints", help="Checkpoint directory")
def list_checkpoints(checkpoint_dir):
    """List available checkpoints."""
    from .training import get_available_checkpoints
    
    checkpoints = get_available_checkpoints(Path(checkpoint_dir))
    
    if not checkpoints:
        click.echo("No checkpoints found.")
        return
    
    click.echo(f"Found {len(checkpoints)} checkpoints:")
    for iteration, path in checkpoints:
        click.echo(f"  Iteration {iteration}: {path}")
    
    latest = Path(checkpoint_dir) / "latest.pt"
    if latest.exists():
        click.echo(f"  Latest: {latest}")


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
def info(checkpoint):
    """Show information about a checkpoint."""
    from .network import AlphaGoNetwork
    
    network = AlphaGoNetwork.load(Path(checkpoint))
    config = network.get_config()
    
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"  Board size: {config['board_size']}x{config['board_size']}")
    click.echo(f"  Residual channels: {config['residual_channels']}")
    click.echo(f"  Residual blocks: {config['num_residual_blocks']}")
    
    total_params = sum(p.numel() for p in network.parameters())
    click.echo(f"  Total parameters: {total_params:,}")


if __name__ == "__main__":
    main()
