"""Play view for playing against trained checkpoints."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Input, Select, Footer, Header
from textual.screen import Screen
from textual.reactive import reactive
from textual.message import Message
from rich.text import Text
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from ..game import GoGame, Move, Stone, format_move, parse_move
from ..network import AlphaGoNetwork
from ..mcts import MCTS
from ..training import get_available_checkpoints
from .board_widget import BoardWidget, PolicyHeatmap


class GameInfo(Static):
    """Panel showing game information."""
    
    DEFAULT_CSS = """
    GameInfo {
        width: 100%;
        height: auto;
        padding: 1;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_player = Stone.BLACK
        self.move_count = 0
        self.captured_black = 0
        self.captured_white = 0
        self.ai_value = 0.0
    
    def update_info(self, game: GoGame, ai_value: float = 0.0):
        """Update game information."""
        self.current_player = game.current_player
        self.move_count = len(game.move_history)
        self.captured_black = game.captured_black
        self.captured_white = game.captured_white
        self.ai_value = ai_value
        self.refresh()
    
    def render(self) -> Text:
        """Render game info."""
        player_symbol = "● Black" if self.current_player == Stone.BLACK else "○ White"
        player_color = "white" if self.current_player == Stone.BLACK else "cyan"
        
        lines = [
            Text(f"To play: ", style="dim") + Text(player_symbol, style=player_color),
            Text(f"Move: {self.move_count}", style="dim"),
            Text(f"Captured by Black: {self.captured_white}", style="white"),
            Text(f"Captured by White: {self.captured_black}", style="cyan"),
            Text(f"AI evaluation: {self.ai_value:+.2f}", style="yellow"),
        ]
        
        return Text("\n").join(lines)


class MoveHistory(Static):
    """Panel showing move history."""
    
    DEFAULT_CSS = """
    MoveHistory {
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
        self.moves: List[Tuple[int, str, Stone]] = []
        self.board_size = 9
    
    def add_move(self, move: Move, player: Stone):
        """Add a move to history."""
        move_num = len(self.moves) + 1
        move_str = format_move(move, self.board_size)
        self.moves.append((move_num, move_str, player))
        self.refresh()
    
    def clear(self):
        """Clear history."""
        self.moves = []
        self.refresh()
    
    def render(self) -> Text:
        """Render move history."""
        if not self.moves:
            return Text("No moves yet", style="dim")
        
        lines = []
        for move_num, move_str, player in self.moves[-30:]:
            color = "white" if player == Stone.BLACK else "cyan"
            symbol = "●" if player == Stone.BLACK else "○"
            lines.append(Text(f"{move_num:3}. {symbol} {move_str}", style=color))
        
        return Text("\n").join(lines)


class PlayView(Screen):
    """Screen for playing against the AI."""
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("n", "new_game", "New Game"),
        ("u", "undo", "Undo"),
        ("p", "pass_move", "Pass"),
        ("h", "hint", "Hint"),
    ]
    
    CSS = """
    PlayView {
        layout: grid;
        grid-size: 3 2;
        grid-columns: 2fr 1fr 1fr;
        grid-rows: 3fr 1fr;
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
    
    #input-container {
        column-span: 1;
        row-span: 1;
        padding: 1;
        height: auto;
    }
    
    #move-input {
        width: 100%;
    }
    
    .title {
        text-style: bold;
        color: cyan;
        padding-bottom: 1;
    }
    
    #message {
        color: yellow;
        padding: 1;
    }
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        board_size: int = 9,
        num_simulations: int = 200,
        player_color: Stone = Stone.BLACK,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.player_color = player_color
        
        self.network: Optional[AlphaGoNetwork] = None
        self.mcts: Optional[MCTS] = None
        self.game: Optional[GoGame] = None
        self.last_move: Optional[Move] = None
        self.last_policy: Optional[np.ndarray] = None
        self.game_over = False
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        
        with Container(id="board-container"):
            yield Static("Go Board", classes="title")
            yield BoardWidget(id="board")
        
        with Container(id="policy-container"):
            yield Static("AI Thinking", classes="title")
            yield PolicyHeatmap(id="policy")
        
        with Container(id="info-container"):
            yield Static("Game Info", classes="title")
            yield GameInfo(id="info")
            yield Static("Move History", classes="title")
            yield MoveHistory(id="history")
        
        with Container(id="input-container"):
            yield Static("Enter move (e.g., D4) or 'pass':", classes="title")
            yield Input(placeholder="Your move...", id="move-input")
            yield Static("", id="message")
        
        yield Footer()

    def on_mount(self) -> None:
        """Initialize game when mounted."""
        # Load network
        self.network = AlphaGoNetwork.load(self.checkpoint_path)
        self.network.eval()
        
        self.mcts = MCTS(
            network=self.network,
            num_simulations=self.num_simulations,
            temperature=0.1,  # Near-deterministic play
        )
        
        self._new_game()
        
        # If AI plays first
        if self.player_color == Stone.WHITE:
            self.call_later(self._ai_move)
    
    def _new_game(self):
        """Start a new game."""
        self.game = GoGame(board_size=self.board_size)
        self.last_move = None
        self.last_policy = None
        self.game_over = False
        
        self._update_display()
        
        history = self.query_one("#history", MoveHistory)
        history.board_size = self.board_size
        history.clear()
        
        self.query_one("#message", Static).update("")
    
    def _update_display(self):
        """Update all display widgets."""
        board = self.query_one("#board", BoardWidget)
        board.highlight_valid = (self.game.current_player == self.player_color)
        board.set_game(self.game, self.last_move)
        
        if self.last_policy is not None:
            policy_widget = self.query_one("#policy", PolicyHeatmap)
            policy_widget.set_policy(self.last_policy, self.board_size)
        
        # Get AI evaluation
        features = self.game.to_features()
        _, value = self.network.predict(features)
        
        info = self.query_one("#info", GameInfo)
        info.update_info(self.game, value)
    
    def _play_move(self, move: Move) -> bool:
        """Play a move and update display."""
        if self.game_over:
            return False
        
        player = self.game.current_player
        
        if not self.game.play_move(move):
            return False
        
        self.last_move = move
        
        history = self.query_one("#history", MoveHistory)
        history.add_move(move, player)
        
        self._update_display()
        self._check_game_over()
        
        return True
    
    def _ai_move(self):
        """Make AI move."""
        if self.game_over or self.game.current_player == self.player_color:
            return
        
        self.query_one("#message", Static).update("AI is thinking...")
        
        # Run MCTS
        move, policy, value = self.mcts.get_move(self.game, add_noise=False)
        self.last_policy = policy
        
        self._play_move(move)
        self.query_one("#message", Static).update(f"AI played: {format_move(move, self.board_size)}")
    
    def _check_game_over(self):
        """Check if game is over and display result."""
        if self.game.is_game_over():
            self.game_over = True
            black_score, white_score = self.game.get_score()
            winner = self.game.get_winner()
            
            if winner == Stone.BLACK:
                result = f"Black wins! ({black_score:.1f} - {white_score:.1f})"
            elif winner == Stone.WHITE:
                result = f"White wins! ({white_score:.1f} - {black_score:.1f})"
            else:
                result = f"Draw! ({black_score:.1f} - {white_score:.1f})"
            
            self.query_one("#message", Static).update(f"Game Over: {result}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle move input."""
        if self.game_over:
            self.query_one("#message", Static).update("Game is over. Press N for new game.")
            event.input.value = ""
            return
        
        if self.game.current_player != self.player_color:
            self.query_one("#message", Static).update("Wait for AI to move.")
            event.input.value = ""
            return
        
        move_str = event.value.strip()
        event.input.value = ""
        
        move = parse_move(move_str, self.board_size)
        if move is None:
            self.query_one("#message", Static).update(f"Invalid move format: {move_str}")
            return
        
        if not self.game.is_valid_move(move):
            self.query_one("#message", Static).update(f"Illegal move: {move_str}")
            return
        
        self._play_move(move)
        
        if not self.game_over:
            self.call_later(self._ai_move)
    
    def action_new_game(self):
        """Start a new game."""
        self._new_game()
        if self.player_color == Stone.WHITE:
            self.call_later(self._ai_move)
    
    def action_undo(self):
        """Undo last move pair."""
        if len(self.game.move_history) < 2:
            self.query_one("#message", Static).update("Nothing to undo.")
            return
        
        # Undo both player and AI moves
        # This is simplified - a full implementation would restore game state
        self.query_one("#message", Static).update("Undo not fully implemented yet.")
    
    def action_pass_move(self):
        """Play a pass move."""
        if self.game_over:
            return
        
        if self.game.current_player != self.player_color:
            return
        
        self._play_move(Move.pass_move())
        
        if not self.game_over:
            self.call_later(self._ai_move)
    
    def action_hint(self):
        """Show AI suggestion for player's move."""
        if self.game_over or self.game.current_player != self.player_color:
            return
        
        # Run MCTS for player's position
        move, policy, value = self.mcts.get_move(self.game, add_noise=False)
        self.last_policy = policy
        self._update_display()
        
        self.query_one("#message", Static).update(
            f"Hint: {format_move(move, self.board_size)} (eval: {value:+.2f})"
        )
    
    def action_quit(self):
        """Quit the game."""
        self.app.exit()
