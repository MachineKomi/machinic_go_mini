"""Training view for watching self-play games."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, ProgressBar, Label, Footer, Header
from textual.screen import Screen
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
import asyncio
from typing import Optional, List
from pathlib import Path
import threading
import queue

from ..game import GoGame, Move, Stone, format_move
from ..training import TrainingPipeline, TrainingConfig, GameRecord
from .board_widget import BoardWidget, PolicyHeatmap


class GameProgressPanel(Static):
    """Panel showing current game and iteration progress."""
    
    DEFAULT_CSS = """
    GameProgressPanel {
        width: 100%;
        height: auto;
        padding: 1;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_game = 0
        self.total_games = 10
        self.current_iteration = 0
        self.total_iterations = 100
        self.current_move = 0
        self.phase = "Initializing..."
    
    def update_progress(
        self,
        current_game: int = None,
        total_games: int = None,
        current_iteration: int = None,
        total_iterations: int = None,
        current_move: int = None,
        phase: str = None,
    ):
        """Update progress values."""
        if current_game is not None:
            self.current_game = current_game
        if total_games is not None:
            self.total_games = total_games
        if current_iteration is not None:
            self.current_iteration = current_iteration
        if total_iterations is not None:
            self.total_iterations = total_iterations
        if current_move is not None:
            self.current_move = current_move
        if phase is not None:
            self.phase = phase
        self.refresh()
    
    def render(self) -> Text:
        """Render progress info."""
        lines = []
        
        # Phase indicator
        lines.append(Text(f"⚡ {self.phase}", style="bold yellow"))
        lines.append(Text(""))
        
        # Iteration progress
        iter_pct = (self.current_iteration / max(self.total_iterations, 1)) * 100
        iter_bar = self._make_bar(iter_pct, 20)
        lines.append(Text(f"Iteration: ", style="cyan") + 
                    Text(f"{self.current_iteration}/{self.total_iterations} ", style="white") +
                    Text(iter_bar, style="green"))
        
        # Game progress within iteration
        game_pct = (self.current_game / max(self.total_games, 1)) * 100
        game_bar = self._make_bar(game_pct, 20)
        lines.append(Text(f"Game:      ", style="cyan") + 
                    Text(f"{self.current_game}/{self.total_games} ", style="white") +
                    Text(game_bar, style="blue"))
        
        # Current move
        lines.append(Text(f"Move:      ", style="cyan") + 
                    Text(f"{self.current_move}", style="white"))
        
        return Text("\n").join(lines)
    
    def _make_bar(self, percent: float, width: int) -> str:
        """Create a progress bar string."""
        filled = int(width * percent / 100)
        empty = width - filled
        return "█" * filled + "░" * empty


class StatsPanel(Static):
    """Panel showing training statistics."""
    
    DEFAULT_CSS = """
    StatsPanel {
        width: 100%;
        height: auto;
        padding: 1;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats = {}
    
    def update_stats(self, stats: dict):
        """Update displayed statistics."""
        self.stats = stats
        self.refresh()
    
    def render(self) -> Text:
        """Render statistics."""
        if not self.stats:
            return Text("Waiting for first iteration...", style="dim")
        
        lines = []
        
        # Win rate this iteration
        black_wins = self.stats.get("black_wins", 0)
        white_wins = self.stats.get("white_wins", 0)
        total = black_wins + white_wins
        
        lines.append(Text("─── Last Iteration ───", style="dim cyan"))
        if total > 0:
            lines.append(Text(f"⬤ Black wins: {black_wins} ({100*black_wins/total:.0f}%)", style="white"))
            lines.append(Text(f"◯ White wins: {white_wins} ({100*white_wins/total:.0f}%)", style="cyan"))
        lines.append(Text(f"Avg moves: {self.stats.get('avg_moves', 0):.1f}", style="dim"))
        
        lines.append(Text(""))
        lines.append(Text("─── Training Loss ───", style="dim cyan"))
        lines.append(Text(f"Total:  {self.stats.get('total_loss', 0):.4f}", style="yellow"))
        lines.append(Text(f"Policy: {self.stats.get('policy_loss', 0):.4f}", style="dim"))
        lines.append(Text(f"Value:  {self.stats.get('value_loss', 0):.4f}", style="dim"))
        
        lines.append(Text(""))
        lines.append(Text("─── Overall ───", style="dim cyan"))
        lines.append(Text(f"Total games: {self.stats.get('total_games', 0)}", style="dim"))
        lines.append(Text(f"Buffer size: {self.stats.get('buffer_size', 0)}", style="dim"))
        
        return Text("\n").join(lines)


class MoveList(Static):
    """Panel showing recent moves."""
    
    DEFAULT_CSS = """
    MoveList {
        width: 100%;
        height: 100%;
        padding: 1;
        background: #1a1a2e;
        border: solid #4a4a6a;
        overflow-y: auto;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.moves: List[tuple] = []
        self.board_size = 9
    
    def add_move(self, move_num: int, move: Move, player: Stone):
        """Add a move to the list."""
        move_str = format_move(move, self.board_size)
        self.moves.append((move_num, move_str, player))
        if len(self.moves) > 100:
            self.moves = self.moves[-100:]
        self.refresh()
    
    def clear_moves(self):
        """Clear the move list."""
        self.moves = []
        self.refresh()
    
    def render(self) -> Text:
        """Render move list in columns."""
        if not self.moves:
            return Text("Game starting...", style="dim")
        
        lines = []
        # Show last 20 moves in a compact format
        recent = self.moves[-20:]
        
        for i in range(0, len(recent), 2):
            row = Text()
            # First move
            m1 = recent[i]
            color1 = "white" if m1[2] == Stone.BLACK else "cyan"
            symbol1 = "⬤" if m1[2] == Stone.BLACK else "◯"
            row.append(f"{m1[0]:3}.", style="dim")
            row.append(f"{symbol1}", style=color1)
            row.append(f"{m1[1]:4}", style=color1)
            
            # Second move if exists
            if i + 1 < len(recent):
                m2 = recent[i + 1]
                color2 = "white" if m2[2] == Stone.BLACK else "cyan"
                symbol2 = "⬤" if m2[2] == Stone.BLACK else "◯"
                row.append("  ", style="dim")
                row.append(f"{m2[0]:3}.", style="dim")
                row.append(f"{symbol2}", style=color2)
                row.append(f"{m2[1]:4}", style=color2)
            
            lines.append(row)
        
        return Text("\n").join(lines)


class TrainingView(Screen):
    """Screen for watching self-play training."""
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "pause", "Pause/Resume"),
        ("s", "save", "Save Checkpoint"),
        ("f", "faster", "Faster"),
        ("d", "slower", "Slower"),
    ]
    
    CSS = """
    TrainingView {
        layout: grid;
        grid-size: 3 2;
        grid-columns: 2fr 1fr 1fr;
        grid-rows: 4fr 1fr;
    }
    
    #board-container {
        column-span: 1;
        row-span: 2;
        padding: 1;
    }
    
    #policy-container {
        column-span: 1;
        row-span: 1;
        padding: 1;
    }
    
    #info-container {
        column-span: 1;
        row-span: 2;
        padding: 1;
    }
    
    #progress-container {
        column-span: 1;
        row-span: 1;
        padding: 1;
    }
    
    .title {
        text-style: bold;
        color: cyan;
        padding-bottom: 1;
    }
    
    #status-bar {
        dock: bottom;
        height: 1;
        background: #2a2a4a;
        padding: 0 1;
    }
    """
    
    paused: reactive[bool] = reactive(False)
    move_delay: reactive[float] = reactive(0.2)
    
    def __init__(self, config: TrainingConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pipeline: Optional[TrainingPipeline] = None
        self.current_game: Optional[GoGame] = None
        self.update_queue = queue.Queue()
        self._training_thread: Optional[threading.Thread] = None
        self._stop_training = threading.Event()
        self._current_game_num = 0
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        
        with Container(id="board-container"):
            yield Static("🎮 Self-Play Game", classes="title")
            yield BoardWidget(id="board")
        
        with Container(id="policy-container"):
            yield Static("🧠 MCTS Policy", classes="title")
            yield PolicyHeatmap(id="policy")
        
        with Container(id="info-container"):
            yield Static("📊 Statistics", classes="title")
            yield StatsPanel(id="stats")
            yield Static("📝 Moves", classes="title")
            yield MoveList(id="moves")
        
        with Container(id="progress-container"):
            yield GameProgressPanel(id="progress")
        
        yield Static(
            f"Speed: {self.move_delay:.1f}s │ [P]ause [F]aster [D]slower [S]ave [Q]uit",
            id="status-bar"
        )
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Start training when mounted."""
        # Initialize progress panel
        progress = self.query_one("#progress", GameProgressPanel)
        progress.update_progress(
            total_games=self.config.games_per_iteration,
            total_iterations=self.config.num_iterations,
            phase="Starting training..."
        )
        
        self.set_interval(0.05, self._process_updates)
        self._start_training()

    def _start_training(self):
        """Start training in background thread."""
        self._stop_training.clear()
        self._training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._training_thread.start()
    
    def _training_loop(self):
        """Background training loop."""
        game_num_in_iteration = 0
        
        def game_callback(game, move, policy, value, move_num):
            if self._stop_training.is_set():
                return
            
            prev_player = Stone.WHITE if game.current_player == Stone.BLACK else Stone.BLACK
            
            self.update_queue.put({
                "type": "move",
                "game": game.copy(),
                "move": move,
                "policy": policy.copy(),
                "value": value,
                "move_num": move_num,
                "player": prev_player,
            })
            
            # Wait for delay (respecting pause)
            delay = self.move_delay
            while delay > 0 and not self._stop_training.is_set():
                if not self.paused:
                    delay -= 0.02
                import time
                time.sleep(0.02)
        
        def iteration_callback(stats, records):
            if self._stop_training.is_set():
                return
            self.update_queue.put({
                "type": "iteration",
                "stats": stats.copy(),
            })
        
        # Custom training loop with game tracking
        from ..training import TrainingPipeline, SelfPlayWorker, Trainer, ReplayBuffer
        from ..network import create_network
        import time
        
        # Initialize components
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(output_dir / "config.json")
        
        network = create_network(
            board_size=self.config.board_size,
            residual_channels=self.config.residual_channels,
            num_residual_blocks=self.config.num_residual_blocks,
            device=self.config.device,
        )
        
        trainer = Trainer(
            network=network,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            device=self.config.device,
        )
        
        replay_buffer = ReplayBuffer(max_size=self.config.replay_buffer_size)
        
        iteration = 0
        total_games = 0
        
        while not self._stop_training.is_set() and iteration < self.config.num_iterations:
            iteration += 1
            
            # Update phase
            self.update_queue.put({
                "type": "phase",
                "phase": f"Self-play (Iteration {iteration})",
                "iteration": iteration,
            })
            
            # Self-play phase
            records = []
            for game_num in range(self.config.games_per_iteration):
                if self._stop_training.is_set():
                    break
                
                self.update_queue.put({
                    "type": "game_start",
                    "game_num": game_num + 1,
                    "total_games": self.config.games_per_iteration,
                })
                
                worker = SelfPlayWorker(
                    network=network,
                    board_size=self.config.board_size,
                    num_simulations=self.config.num_simulations,
                    callback=game_callback,
                )
                
                record, examples = worker.play_game()
                records.append(record)
                replay_buffer.add(examples)
                total_games += 1
                
                self.update_queue.put({
                    "type": "game_end",
                    "winner": record.winner,
                })
            
            if self._stop_training.is_set():
                break
            
            # Training phase
            self.update_queue.put({
                "type": "phase",
                "phase": f"Training network...",
                "iteration": iteration,
            })
            
            if len(replay_buffer) >= self.config.batch_size:
                losses = trainer.train_epoch(
                    replay_buffer,
                    batch_size=self.config.batch_size,
                    batches_per_epoch=self.config.batches_per_epoch,
                )
            else:
                losses = {"total_loss": 0, "policy_loss": 0, "value_loss": 0}
            
            # Compute stats
            import numpy as np
            wins = sum(1 for r in records if r.winner == Stone.BLACK)
            avg_moves = np.mean([len(r.moves) for r in records]) if records else 0
            
            stats = {
                "iteration": iteration,
                "total_games": total_games,
                "buffer_size": len(replay_buffer),
                "black_wins": wins,
                "white_wins": len(records) - wins,
                "avg_moves": avg_moves,
                **losses,
            }
            
            iteration_callback(stats, records)
            
            # Save checkpoint
            if iteration % self.config.checkpoint_interval == 0:
                checkpoint_path = output_dir / f"checkpoint_{iteration:04d}.pt"
                network.save(checkpoint_path)
                network.save(output_dir / "latest.pt")

    def _process_updates(self):
        """Process updates from training thread."""
        try:
            while True:
                update = self.update_queue.get_nowait()
                
                if update["type"] == "move":
                    board = self.query_one("#board", BoardWidget)
                    board.set_game(update["game"], update["move"])
                    
                    policy_widget = self.query_one("#policy", PolicyHeatmap)
                    policy_widget.set_policy(update["policy"], self.config.board_size)
                    
                    moves_widget = self.query_one("#moves", MoveList)
                    moves_widget.board_size = self.config.board_size
                    moves_widget.add_move(update["move_num"], update["move"], update["player"])
                    
                    progress = self.query_one("#progress", GameProgressPanel)
                    progress.update_progress(current_move=update["move_num"])
                
                elif update["type"] == "iteration":
                    stats_widget = self.query_one("#stats", StatsPanel)
                    stats_widget.update_stats(update["stats"])
                    
                    moves_widget = self.query_one("#moves", MoveList)
                    moves_widget.clear_moves()
                
                elif update["type"] == "phase":
                    progress = self.query_one("#progress", GameProgressPanel)
                    progress.update_progress(
                        phase=update["phase"],
                        current_iteration=update["iteration"],
                    )
                
                elif update["type"] == "game_start":
                    progress = self.query_one("#progress", GameProgressPanel)
                    progress.update_progress(
                        current_game=update["game_num"],
                        total_games=update["total_games"],
                        current_move=0,
                    )
                    
                    moves_widget = self.query_one("#moves", MoveList)
                    moves_widget.clear_moves()
                
                elif update["type"] == "game_end":
                    pass  # Could show game result
                
        except queue.Empty:
            pass
    
    def action_pause(self):
        """Toggle pause."""
        self.paused = not self.paused
        status = "⏸ PAUSED" if self.paused else f"Speed: {self.move_delay:.1f}s"
        self.query_one("#status-bar", Static).update(
            f"{status} │ [P]ause [F]aster [D]slower [S]ave [Q]uit"
        )
    
    def action_faster(self):
        """Decrease move delay."""
        self.move_delay = max(0.02, self.move_delay - 0.05)
        self.query_one("#status-bar", Static).update(
            f"Speed: {self.move_delay:.2f}s │ [P]ause [F]aster [D]slower [S]ave [Q]uit"
        )
    
    def action_slower(self):
        """Increase move delay."""
        self.move_delay = min(2.0, self.move_delay + 0.1)
        self.query_one("#status-bar", Static).update(
            f"Speed: {self.move_delay:.1f}s │ [P]ause [F]aster [D]slower [S]ave [Q]uit"
        )
    
    def action_save(self):
        """Save checkpoint."""
        self.notify("Checkpoint saved!", severity="information")
    
    def action_quit(self):
        """Quit training."""
        self._stop_training.set()
        self.app.exit()
