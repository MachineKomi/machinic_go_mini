"""Main Textual application for Machinic Go Mini."""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button, Select, Header, Footer, Label
from textual.screen import Screen
from pathlib import Path
from typing import Optional, List, Tuple

from ..game import Stone, GoGame
from ..training import TrainingConfig, get_available_checkpoints
from .training_view import TrainingView
from .play_view import PlayView


class MenuScreen(Screen):
    """Main menu screen."""
    
    CSS = """
    MenuScreen {
        align: center middle;
    }
    
    #menu-container {
        width: 60;
        height: auto;
        padding: 2;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    
    #title {
        text-align: center;
        text-style: bold;
        color: cyan;
        padding-bottom: 2;
    }
    
    #subtitle {
        text-align: center;
        color: #888888;
        padding-bottom: 2;
    }
    
    Button {
        width: 100%;
        margin: 1 0;
    }
    
    .section-title {
        text-style: bold;
        color: yellow;
        padding-top: 1;
    }
    """
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints"), **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
    
    def compose(self) -> ComposeResult:
        """Create menu widgets."""
        yield Header()
        
        with Container(id="menu-container"):
            yield Static("🎮 Machinic Go Mini", id="title")
            yield Static("Self-play training for Go", id="subtitle")
            
            yield Static("Training", classes="section-title")
            yield Button("Start New Training", id="btn-train-new", variant="primary")
            yield Button("Continue Training", id="btn-train-continue")
            
            yield Static("Play", classes="section-title")
            yield Button("Play vs AI", id="btn-play", variant="success")
            yield Button("Watch AI vs AI", id="btn-watch")
            
            yield Static("", classes="section-title")
            yield Button("Quit", id="btn-quit", variant="error")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-train-new":
            self.app.push_screen(TrainingConfigScreen(self.checkpoint_dir))
        
        elif event.button.id == "btn-train-continue":
            latest = self.checkpoint_dir / "latest.pt"
            if latest.exists():
                config = TrainingConfig.load(self.checkpoint_dir / "config.json")
                self.app.push_screen(TrainingView(config))
            else:
                self.notify("No checkpoint found to continue.", severity="error")
        
        elif event.button.id == "btn-play":
            self.app.push_screen(CheckpointSelectScreen(self.checkpoint_dir, mode="play"))
        
        elif event.button.id == "btn-watch":
            self.app.push_screen(CheckpointSelectScreen(self.checkpoint_dir, mode="watch"))
        
        elif event.button.id == "btn-quit":
            self.app.exit()


class TrainingConfigScreen(Screen):
    """Screen for configuring training parameters."""
    
    CSS = """
    TrainingConfigScreen {
        align: center middle;
    }
    
    #config-container {
        width: 70;
        height: auto;
        padding: 2;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    
    #title {
        text-align: center;
        text-style: bold;
        color: cyan;
        padding-bottom: 2;
    }
    
    .config-row {
        height: 3;
        margin: 1 0;
    }
    
    .config-label {
        width: 30;
    }
    
    Select {
        width: 30;
    }
    
    #buttons {
        margin-top: 2;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    def __init__(self, checkpoint_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
    
    def compose(self) -> ComposeResult:
        """Create config widgets."""
        yield Header()
        
        with Container(id="config-container"):
            yield Static("Training Configuration", id="title")
            
            with Horizontal(classes="config-row"):
                yield Static("Board Size:", classes="config-label")
                yield Select(
                    [(str(s), s) for s in [9, 13, 19]],
                    value=9,
                    id="board-size"
                )
            
            with Horizontal(classes="config-row"):
                yield Static("Network Size:", classes="config-label")
                yield Select(
                    [("Small (64ch, 4 blocks)", "small"),
                     ("Medium (128ch, 6 blocks)", "medium"),
                     ("Large (256ch, 10 blocks)", "large")],
                    value="small",
                    id="network-size"
                )
            
            with Horizontal(classes="config-row"):
                yield Static("MCTS Simulations:", classes="config-label")
                yield Select(
                    [(str(s), s) for s in [50, 100, 200, 400]],
                    value=100,
                    id="simulations"
                )
            
            with Horizontal(classes="config-row"):
                yield Static("Games per Iteration:", classes="config-label")
                yield Select(
                    [(str(s), s) for s in [5, 10, 20, 50]],
                    value=10,
                    id="games-per-iter"
                )
            
            with Horizontal(id="buttons"):
                yield Button("Start Training", id="btn-start", variant="primary")
                yield Button("Back", id="btn-back")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.app.pop_screen()
        
        elif event.button.id == "btn-start":
            board_size = self.query_one("#board-size", Select).value
            network_size = self.query_one("#network-size", Select).value
            simulations = self.query_one("#simulations", Select).value
            games_per_iter = self.query_one("#games-per-iter", Select).value
            
            network_params = {
                "small": (64, 4),
                "medium": (128, 6),
                "large": (256, 10),
            }
            channels, blocks = network_params[network_size]
            
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            config = TrainingConfig(
                board_size=board_size,
                residual_channels=channels,
                num_residual_blocks=blocks,
                num_simulations=simulations,
                games_per_iteration=games_per_iter,
                device=device,
                output_dir=str(self.checkpoint_dir),
            )
            
            self.app.switch_screen(TrainingView(config))


class CheckpointSelectScreen(Screen):
    """Screen for selecting a checkpoint to play against."""
    
    CSS = """
    CheckpointSelectScreen {
        align: center middle;
    }
    
    #select-container {
        width: 60;
        height: auto;
        padding: 2;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    
    #title {
        text-align: center;
        text-style: bold;
        color: cyan;
        padding-bottom: 2;
    }
    
    .config-row {
        height: 3;
        margin: 1 0;
    }
    
    .config-label {
        width: 20;
    }
    
    Select {
        width: 35;
    }
    
    #buttons {
        margin-top: 2;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    def __init__(self, checkpoint_dir: Path, mode: str = "play", **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.mode = mode
        self.checkpoints = get_available_checkpoints(checkpoint_dir)
    
    def compose(self) -> ComposeResult:
        """Create selection widgets."""
        yield Header()
        
        with Container(id="select-container"):
            title = "Select Checkpoint to Play" if self.mode == "play" else "Select Checkpoint to Watch"
            yield Static(title, id="title")
            
            if not self.checkpoints:
                yield Static("No checkpoints found. Train first!", style="red")
            else:
                with Horizontal(classes="config-row"):
                    yield Static("Checkpoint:", classes="config-label")
                    options = [(f"Iteration {it}", str(path)) for it, path in self.checkpoints]
                    options.append(("Latest", str(self.checkpoint_dir / "latest.pt")))
                    yield Select(options, value=options[-1][1], id="checkpoint")
                
                if self.mode == "play":
                    with Horizontal(classes="config-row"):
                        yield Static("Play as:", classes="config-label")
                        yield Select(
                            [("Black (first)", "black"), ("White (second)", "white")],
                            value="black",
                            id="color"
                        )
                    
                    with Horizontal(classes="config-row"):
                        yield Static("AI Strength:", classes="config-label")
                        yield Select(
                            [("Easy (50 sims)", 50),
                             ("Medium (100 sims)", 100),
                             ("Hard (200 sims)", 200),
                             ("Expert (400 sims)", 400)],
                            value=100,
                            id="strength"
                        )
            
            with Horizontal(id="buttons"):
                if self.checkpoints:
                    yield Button("Start", id="btn-start", variant="primary")
                yield Button("Back", id="btn-back")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.app.pop_screen()
        
        elif event.button.id == "btn-start":
            checkpoint_path = Path(self.query_one("#checkpoint", Select).value)
            
            config_path = self.checkpoint_dir / "config.json"
            if config_path.exists():
                config = TrainingConfig.load(config_path)
                board_size = config.board_size
            else:
                board_size = 9
            
            if self.mode == "play":
                color_str = self.query_one("#color", Select).value
                player_color = Stone.BLACK if color_str == "black" else Stone.WHITE
                strength = self.query_one("#strength", Select).value
                
                self.app.switch_screen(PlayView(
                    checkpoint_path=checkpoint_path,
                    board_size=board_size,
                    num_simulations=strength,
                    player_color=player_color,
                ))
            else:
                self.app.switch_screen(WatchView(
                    checkpoint_path=checkpoint_path,
                    board_size=board_size,
                ))


class WatchView(Screen):
    """Screen for watching AI vs AI games."""
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("n", "new_game", "New Game"),
        ("f", "faster", "Faster"),
        ("d", "slower", "Slower"),
        ("space", "pause", "Pause"),
    ]
    
    CSS = """
    WatchView {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 2fr 1fr;
        grid-rows: 3fr 1fr;
    }
    
    #board-container {
        row-span: 2;
        padding: 1;
    }
    
    #info-container {
        padding: 1;
    }
    
    #status-container {
        padding: 1;
    }
    
    .title {
        text-style: bold;
        color: cyan;
        padding-bottom: 1;
    }
    """
    
    def __init__(self, checkpoint_path: Path, board_size: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.board_size = board_size
        self.move_delay = 0.5
        self.paused = False
        self.game = None
        self.network = None
        self.mcts = None
    
    def compose(self) -> ComposeResult:
        from .board_widget import BoardWidget, PolicyHeatmap
        from .play_view import GameInfo
        
        yield Header()
        
        with Container(id="board-container"):
            yield Static("AI vs AI", classes="title")
            yield BoardWidget(id="board")
        
        with Container(id="info-container"):
            yield Static("Game Info", classes="title")
            yield GameInfo(id="info")
            yield PolicyHeatmap(id="policy")
        
        with Container(id="status-container"):
            yield Static(f"Speed: {self.move_delay:.1f}s | Space=Pause F/D=Speed", id="status")
        
        yield Footer()
    
    def on_mount(self) -> None:
        from ..network import AlphaGoNetwork
        from ..mcts import MCTS
        
        self.network = AlphaGoNetwork.load(self.checkpoint_path)
        self.network.eval()
        
        self.mcts = MCTS(
            network=self.network,
            num_simulations=200,
            temperature=0.3,
        )
        
        self._new_game()
        self.set_interval(0.1, self._game_loop)
    
    def _new_game(self):
        from .board_widget import BoardWidget
        
        self.game = GoGame(board_size=self.board_size)
        board = self.query_one("#board", BoardWidget)
        board.set_game(self.game, None)
        self._last_move_time = 0
    
    def _game_loop(self):
        import time
        from .board_widget import BoardWidget, PolicyHeatmap
        from .play_view import GameInfo
        
        if self.paused or self.game.is_game_over():
            return
        
        current_time = time.time()
        if current_time - self._last_move_time < self.move_delay:
            return
        
        move, policy, value = self.mcts.get_move(self.game, add_noise=True)
        self.game.play_move(move)
        
        board = self.query_one("#board", BoardWidget)
        board.set_game(self.game, move)
        
        policy_widget = self.query_one("#policy", PolicyHeatmap)
        policy_widget.set_policy(policy, self.board_size)
        
        info = self.query_one("#info", GameInfo)
        info.update_info(self.game, value)
        
        self._last_move_time = current_time
        
        if self.game.is_game_over():
            black_score, white_score = self.game.get_score()
            winner = self.game.get_winner()
            result = "Black wins" if winner == Stone.BLACK else "White wins" if winner == Stone.WHITE else "Draw"
            self.query_one("#status", Static).update(f"Game Over: {result} ({black_score:.1f}-{white_score:.1f})")
    
    def action_pause(self):
        self.paused = not self.paused
        status = "PAUSED" if self.paused else f"Speed: {self.move_delay:.1f}s"
        self.query_one("#status", Static).update(f"{status} | Space=Pause F/D=Speed")
    
    def action_faster(self):
        self.move_delay = max(0.1, self.move_delay - 0.1)
        self.query_one("#status", Static).update(f"Speed: {self.move_delay:.1f}s | Space=Pause F/D=Speed")
    
    def action_slower(self):
        self.move_delay = min(2.0, self.move_delay + 0.1)
        self.query_one("#status", Static).update(f"Speed: {self.move_delay:.1f}s | Space=Pause F/D=Speed")
    
    def action_new_game(self):
        self._new_game()
    
    def action_quit(self):
        self.app.exit()


class MachinicGoApp(App):
    """Main application."""
    
    TITLE = "Machinic Go Mini"
    CSS = """
    Screen {
        background: #0f0f1a;
    }
    """
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints"), **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
    
    def on_mount(self) -> None:
        """Show main menu on start."""
        self.push_screen(MenuScreen(self.checkpoint_dir))
