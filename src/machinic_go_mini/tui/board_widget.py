"""Go board widget for Textual TUI."""

from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text
from rich.style import Style
from rich.console import RenderableType
import numpy as np
from typing import Optional, Tuple, List

from ..game import GoGame, Stone, Move


class BoardWidget(Widget):
    """Widget displaying a Go board with proper grid lines."""
    
    DEFAULT_CSS = """
    BoardWidget {
        width: auto;
        height: auto;
        padding: 1;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    """
    
    hover_pos: reactive[Optional[Tuple[int, int]]] = reactive(None)
    show_coordinates: reactive[bool] = reactive(True)
    highlight_valid: reactive[bool] = reactive(False)
    
    # Stone symbols - using larger Unicode circles
    BLACK_STONE = "⬤"
    WHITE_STONE = "◯"
    BLACK_LAST = "⬤"  # Will use different color
    WHITE_LAST = "◯"  # Will use different color
    
    # Grid intersection characters
    GRID_TL = "┌"  # Top-left
    GRID_TR = "┐"  # Top-right
    GRID_BL = "└"  # Bottom-left
    GRID_BR = "┘"  # Bottom-right
    GRID_T = "┬"   # Top edge
    GRID_B = "┴"   # Bottom edge
    GRID_L = "├"   # Left edge
    GRID_R = "┤"   # Right edge
    GRID_C = "┼"   # Center
    GRID_H = "─"   # Horizontal line
    STAR = "●"     # Star point (hoshi)
    
    STAR_POINTS_9 = [(2, 2), (6, 2), (2, 6), (6, 6), (4, 4)]
    STAR_POINTS_13 = [(3, 3), (9, 3), (3, 9), (9, 9), (6, 6), (3, 6), (9, 6), (6, 3), (6, 9)]
    STAR_POINTS_19 = [(3, 3), (9, 3), (15, 3), (3, 9), (9, 9), (15, 9), (3, 15), (9, 15), (15, 15)]
    
    def __init__(self, game: Optional[GoGame] = None, **kwargs):
        super().__init__(**kwargs)
        self._game: Optional[GoGame] = game
        self._last_move: Optional[Move] = None
    
    def get_star_points(self, board_size: int) -> List[Tuple[int, int]]:
        """Get star point positions for board size."""
        if board_size == 9:
            return self.STAR_POINTS_9
        elif board_size == 13:
            return self.STAR_POINTS_13
        elif board_size == 19:
            return self.STAR_POINTS_19
        return []
    
    def _get_grid_char(self, x: int, y: int, board_size: int) -> str:
        """Get the grid character for an empty intersection."""
        is_top = (y == board_size - 1)
        is_bottom = (y == 0)
        is_left = (x == 0)
        is_right = (x == board_size - 1)
        
        if is_top and is_left:
            return self.GRID_TL
        elif is_top and is_right:
            return self.GRID_TR
        elif is_bottom and is_left:
            return self.GRID_BL
        elif is_bottom and is_right:
            return self.GRID_BR
        elif is_top:
            return self.GRID_T
        elif is_bottom:
            return self.GRID_B
        elif is_left:
            return self.GRID_L
        elif is_right:
            return self.GRID_R
        else:
            return self.GRID_C

    def render(self) -> RenderableType:
        """Render the board with grid lines."""
        if self._game is None:
            return Text("No game loaded")
        
        board_size = self._game.board_size
        star_points = self.get_star_points(board_size)
        valid_moves = set()
        
        if self.highlight_valid:
            for move in self._game.get_valid_moves():
                if not move.is_pass:
                    valid_moves.add((move.x, move.y))
        
        lines = []
        
        # Column labels
        if self.show_coordinates:
            cols = "   " + "─".join(self._col_label(x) for x in range(board_size))
            lines.append(Text(cols, style="dim cyan"))
        
        # Board rows (top to bottom = high y to low y)
        for y in range(board_size - 1, -1, -1):
            row_text = Text()
            
            if self.show_coordinates:
                row_text.append(f"{y+1:2} ", style="dim cyan")
            
            for x in range(board_size):
                stone = self._game.board[y, x]
                is_last = (self._last_move and not self._last_move.is_pass and 
                          self._last_move.x == x and self._last_move.y == y)
                is_hover = self.hover_pos == (x, y)
                is_star = (x, y) in star_points
                is_valid = (x, y) in valid_moves
                
                # Get character and style
                char, style = self._get_cell_display(
                    x, y, board_size, stone, is_last, is_hover, is_star, is_valid
                )
                
                row_text.append(char, style=style)
                
                # Add horizontal connector between cells (except last column)
                if x < board_size - 1:
                    if stone == Stone.EMPTY and self._game.board[y, x + 1] == Stone.EMPTY:
                        row_text.append(self.GRID_H, style=Style(color="#5a5a7a"))
                    else:
                        row_text.append(" ", style=Style(color="#5a5a7a"))
            
            lines.append(row_text)
        
        # Capture info
        cap_text = Text()
        cap_text.append(f"Captured: ", style="dim")
        cap_text.append(f"⬤ {self._game.captured_white}", style="white")
        cap_text.append("  ", style="dim")
        cap_text.append(f"◯ {self._game.captured_black}", style="cyan")
        lines.append(Text(""))
        lines.append(cap_text)
        
        return Text("\n").join(lines)
    
    def _col_label(self, x: int) -> str:
        """Get column label (skip I)."""
        if x >= 8:
            return chr(ord('A') + x + 1)
        return chr(ord('A') + x)
    
    def _get_cell_display(
        self,
        x: int,
        y: int,
        board_size: int,
        stone: Stone,
        is_last: bool,
        is_hover: bool,
        is_star: bool,
        is_valid: bool,
    ) -> Tuple[str, Style]:
        """Get character and style for a cell."""
        if is_hover and stone == Stone.EMPTY:
            return "◌", Style(color="yellow")
        
        if stone == Stone.BLACK:
            if is_last:
                return self.BLACK_STONE, Style(color="bright_green", bold=True)
            return self.BLACK_STONE, Style(color="white")
        
        if stone == Stone.WHITE:
            if is_last:
                return self.WHITE_STONE, Style(color="bright_green", bold=True)
            return self.WHITE_STONE, Style(color="cyan")
        
        # Empty intersection
        if is_valid:
            return "·", Style(color="green")
        if is_star:
            return "●", Style(color="#7a7a9a")
        
        # Regular grid intersection
        grid_char = self._get_grid_char(x, y, board_size)
        return grid_char, Style(color="#5a5a7a")
    
    def set_game(self, game: GoGame, last_move: Optional[Move] = None):
        """Update the displayed game."""
        self._game = game
        self._last_move = last_move
        self.refresh()


class PolicyHeatmap(Widget):
    """Widget showing MCTS policy as a heatmap."""
    
    DEFAULT_CSS = """
    PolicyHeatmap {
        width: auto;
        height: auto;
        padding: 1;
        background: #1a1a2e;
        border: solid #4a4a6a;
    }
    """
    
    HEAT_CHARS = " ░▒▓█"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._policy: Optional[np.ndarray] = None
        self._board_size: int = 9
    
    def render(self) -> RenderableType:
        """Render the policy heatmap."""
        if self._policy is None:
            return Text("No policy data", style="dim")
        
        lines = []
        
        # Column labels
        cols = "   " + "─".join(self._col_label(x) for x in range(self._board_size))
        lines.append(Text(cols, style="dim cyan"))
        
        # Reshape policy (excluding pass)
        policy_board = self._policy[:-1].reshape(self._board_size, self._board_size)
        max_prob = policy_board.max()
        
        for y in range(self._board_size - 1, -1, -1):
            row_text = Text()
            row_text.append(f"{y+1:2} ", style="dim cyan")
            
            for x in range(self._board_size):
                prob = policy_board[y, x]
                intensity = int((prob / max(max_prob, 0.001)) * (len(self.HEAT_CHARS) - 1))
                char = self.HEAT_CHARS[intensity]
                
                # Color based on probability
                if prob > 0.3:
                    style = Style(color="bright_red", bold=True)
                elif prob > 0.1:
                    style = Style(color="yellow")
                elif prob > 0.01:
                    style = Style(color="green")
                else:
                    style = Style(color="#3a3a5a")
                
                row_text.append(char, style=style)
                if x < self._board_size - 1:
                    row_text.append(" ", style=Style(color="#3a3a5a"))
            
            lines.append(row_text)
        
        # Pass probability
        pass_prob = self._policy[-1]
        lines.append(Text(""))
        lines.append(Text(f"Pass: {pass_prob:.1%}", style="dim"))
        
        return Text("\n").join(lines)
    
    def _col_label(self, x: int) -> str:
        """Get column label (skip I)."""
        if x >= 8:
            return chr(ord('A') + x + 1)
        return chr(ord('A') + x)
    
    def set_policy(self, policy: np.ndarray, board_size: int):
        """Update the displayed policy."""
        self._policy = policy
        self._board_size = board_size
        self.refresh()
