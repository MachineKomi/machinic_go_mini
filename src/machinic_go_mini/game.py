"""Go game engine with full rules implementation."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple, List
from enum import IntEnum
import copy


class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    
    @property
    def opponent(self) -> "Stone":
        if self == Stone.BLACK:
            return Stone.WHITE
        elif self == Stone.WHITE:
            return Stone.BLACK
        return Stone.EMPTY


@dataclass
class Move:
    """Represents a move in Go."""
    x: int  # -1 for pass
    y: int  # -1 for pass
    
    @property
    def is_pass(self) -> bool:
        return self.x == -1 and self.y == -1
    
    @classmethod
    def pass_move(cls) -> "Move":
        return cls(-1, -1)
    
    def to_index(self, board_size: int) -> int:
        """Convert to flat index (pass = board_size^2)."""
        if self.is_pass:
            return board_size * board_size
        return self.y * board_size + self.x
    
    @classmethod
    def from_index(cls, index: int, board_size: int) -> "Move":
        """Create move from flat index."""
        if index == board_size * board_size:
            return cls.pass_move()
        return cls(index % board_size, index // board_size)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.x == other.x and self.y == other.y


@dataclass
class GoGame:
    """Go game state with full rules."""
    board_size: int = 9
    komi: float = 6.5
    board: np.ndarray = field(default=None)
    current_player: Stone = Stone.BLACK
    move_history: List[Move] = field(default_factory=list)
    ko_point: Optional[Tuple[int, int]] = None
    consecutive_passes: int = 0
    captured_black: int = 0  # Black stones captured by White
    captured_white: int = 0  # White stones captured by Black
    _position_hashes: Set[int] = field(default_factory=set)
    
    def __post_init__(self):
        if self.board is None:
            self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self._position_hashes.add(self._hash_position())
    
    def _hash_position(self) -> int:
        """Hash current board position for superko detection."""
        return hash((self.board.tobytes(), self.current_player))
    
    def copy(self) -> "GoGame":
        """Create a deep copy of the game state."""
        new_game = GoGame(
            board_size=self.board_size,
            komi=self.komi,
            board=self.board.copy(),
            current_player=self.current_player,
            move_history=self.move_history.copy(),
            ko_point=self.ko_point,
            consecutive_passes=self.consecutive_passes,
            captured_black=self.captured_black,
            captured_white=self.captured_white,
        )
        new_game._position_hashes = self._position_hashes.copy()
        return new_game

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))
        return neighbors
    
    def _get_group(self, x: int, y: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Get connected group and its liberties using flood fill."""
        stone = self.board[y, x]
        if stone == Stone.EMPTY:
            return set(), set()
        
        group = set()
        liberties = set()
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            group.add((cx, cy))
            
            for nx, ny in self._get_neighbors(cx, cy):
                if self.board[ny, nx] == Stone.EMPTY:
                    liberties.add((nx, ny))
                elif self.board[ny, nx] == stone and (nx, ny) not in group:
                    stack.append((nx, ny))
        
        return group, liberties
    
    def _remove_group(self, group: Set[Tuple[int, int]]) -> int:
        """Remove a group from the board, return number of stones removed."""
        for x, y in group:
            self.board[y, x] = Stone.EMPTY
        return len(group)
    
    def _would_be_suicide(self, x: int, y: int, stone: Stone) -> bool:
        """Check if placing a stone would be suicide."""
        # Temporarily place the stone
        self.board[y, x] = stone
        
        # Check if the placed stone has liberties
        _, liberties = self._get_group(x, y)
        if liberties:
            self.board[y, x] = Stone.EMPTY
            return False
        
        # Check if it captures any opponent stones
        opponent = stone.opponent
        for nx, ny in self._get_neighbors(x, y):
            if self.board[ny, nx] == opponent:
                _, opp_liberties = self._get_group(nx, ny)
                if not opp_liberties:
                    self.board[y, x] = Stone.EMPTY
                    return False
        
        self.board[y, x] = Stone.EMPTY
        return True
    
    def is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid."""
        if move.is_pass:
            return True
        
        x, y = move.x, move.y
        
        # Check bounds
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
        
        # Check if position is empty
        if self.board[y, x] != Stone.EMPTY:
            return False
        
        # Check ko rule
        if self.ko_point == (x, y):
            return False
        
        # Check suicide rule
        if self._would_be_suicide(x, y, self.current_player):
            return False
        
        return True
    
    def get_valid_moves(self) -> List[Move]:
        """Get all valid moves for current player."""
        moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                move = Move(x, y)
                if self.is_valid_move(move):
                    moves.append(move)
        moves.append(Move.pass_move())
        return moves
    
    def get_valid_moves_mask(self) -> np.ndarray:
        """Get boolean mask of valid moves (including pass)."""
        mask = np.zeros(self.board_size * self.board_size + 1, dtype=bool)
        for move in self.get_valid_moves():
            mask[move.to_index(self.board_size)] = True
        return mask

    def play_move(self, move: Move) -> bool:
        """Play a move, return True if successful."""
        if not self.is_valid_move(move):
            return False
        
        if move.is_pass:
            self.consecutive_passes += 1
            self.ko_point = None
            self.move_history.append(move)
            self.current_player = self.current_player.opponent
            return True
        
        self.consecutive_passes = 0
        x, y = move.x, move.y
        
        # Place the stone
        self.board[y, x] = self.current_player
        
        # Capture opponent stones
        captured = 0
        captured_positions = []
        opponent = self.current_player.opponent
        
        for nx, ny in self._get_neighbors(x, y):
            if self.board[ny, nx] == opponent:
                group, liberties = self._get_group(nx, ny)
                if not liberties:
                    captured += self._remove_group(group)
                    captured_positions.extend(group)
        
        # Update capture counts
        if self.current_player == Stone.BLACK:
            self.captured_white += captured
        else:
            self.captured_black += captured
        
        # Check for ko
        self.ko_point = None
        if captured == 1 and len(captured_positions) == 1:
            # Check if this creates a ko situation
            _, our_liberties = self._get_group(x, y)
            if len(our_liberties) == 1:
                self.ko_point = captured_positions[0]
        
        # Record position hash for superko
        self._position_hashes.add(self._hash_position())
        
        self.move_history.append(move)
        self.current_player = self.current_player.opponent
        return True
    
    def is_game_over(self) -> bool:
        """Check if game is over (two consecutive passes)."""
        return self.consecutive_passes >= 2
    
    def _count_territory(self) -> Tuple[int, int]:
        """Count territory using Chinese rules (area scoring)."""
        visited = np.zeros_like(self.board, dtype=bool)
        black_territory = 0
        white_territory = 0
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if visited[y, x] or self.board[y, x] != Stone.EMPTY:
                    continue
                
                # Flood fill to find empty region
                region = set()
                borders = set()
                stack = [(x, y)]
                
                while stack:
                    cx, cy = stack.pop()
                    if visited[cy, cx]:
                        continue
                    if self.board[cy, cx] != Stone.EMPTY:
                        borders.add(self.board[cy, cx])
                        continue
                    
                    visited[cy, cx] = True
                    region.add((cx, cy))
                    
                    for nx, ny in self._get_neighbors(cx, cy):
                        if not visited[ny, nx]:
                            stack.append((nx, ny))
                
                # Assign territory if bordered by only one color
                if borders == {Stone.BLACK}:
                    black_territory += len(region)
                elif borders == {Stone.WHITE}:
                    white_territory += len(region)
        
        return black_territory, white_territory
    
    def get_score(self) -> Tuple[float, float]:
        """Get final score (black_score, white_score) using Chinese rules."""
        black_stones = np.sum(self.board == Stone.BLACK)
        white_stones = np.sum(self.board == Stone.WHITE)
        black_territory, white_territory = self._count_territory()
        
        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory) + self.komi
        
        return black_score, white_score
    
    def get_winner(self) -> Optional[Stone]:
        """Get winner (None if game not over)."""
        if not self.is_game_over():
            return None
        black_score, white_score = self.get_score()
        if black_score > white_score:
            return Stone.BLACK
        elif white_score > black_score:
            return Stone.WHITE
        return None  # Draw (rare with komi)
    
    def get_result(self) -> float:
        """Get game result from Black's perspective (-1, 0, or 1)."""
        winner = self.get_winner()
        if winner == Stone.BLACK:
            return 1.0
        elif winner == Stone.WHITE:
            return -1.0
        return 0.0

    def to_features(self) -> np.ndarray:
        """Convert game state to neural network input features.
        
        Features (17 planes for board_size x board_size):
        - Plane 0: Current player's stones
        - Plane 1: Opponent's stones
        - Plane 2-7: Current player's stones history (last 3 moves each)
        - Plane 8-13: Opponent's stones history (last 3 moves each)
        - Plane 14: All ones if black to play, zeros if white
        - Plane 15: Valid moves mask
        - Plane 16: Ko point (if any)
        """
        features = np.zeros((17, self.board_size, self.board_size), dtype=np.float32)
        
        # Current player's stones
        if self.current_player == Stone.BLACK:
            features[0] = (self.board == Stone.BLACK).astype(np.float32)
            features[1] = (self.board == Stone.WHITE).astype(np.float32)
        else:
            features[0] = (self.board == Stone.WHITE).astype(np.float32)
            features[1] = (self.board == Stone.BLACK).astype(np.float32)
        
        # Color to play
        if self.current_player == Stone.BLACK:
            features[14] = np.ones((self.board_size, self.board_size), dtype=np.float32)
        
        # Valid moves
        for move in self.get_valid_moves():
            if not move.is_pass:
                features[15, move.y, move.x] = 1.0
        
        # Ko point
        if self.ko_point:
            features[16, self.ko_point[1], self.ko_point[0]] = 1.0
        
        return features
    
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {Stone.EMPTY: ".", Stone.BLACK: "●", Stone.WHITE: "○"}
        cols = "  " + " ".join("ABCDEFGHJKLMNOPQRST"[:self.board_size])
        rows = []
        for y in range(self.board_size - 1, -1, -1):
            row = f"{y+1:2} "
            row += " ".join(symbols[Stone(self.board[y, x])] for x in range(self.board_size))
            rows.append(row)
        return cols + "\n" + "\n".join(rows)


def parse_move(move_str: str, board_size: int) -> Optional[Move]:
    """Parse a move string like 'D4' or 'pass'."""
    move_str = move_str.strip().lower()
    if move_str == "pass":
        return Move.pass_move()
    
    if len(move_str) < 2:
        return None
    
    col = move_str[0]
    if col < 'a' or col > 't' or col == 'i':
        return None
    
    # Convert column letter to x coordinate (skip 'i')
    x = ord(col) - ord('a')
    if col > 'i':
        x -= 1
    
    try:
        y = int(move_str[1:]) - 1
    except ValueError:
        return None
    
    if 0 <= x < board_size and 0 <= y < board_size:
        return Move(x, y)
    return None


def format_move(move: Move, board_size: int) -> str:
    """Format a move as a string like 'D4' or 'pass'."""
    if move.is_pass:
        return "pass"
    
    # Convert x to column letter (skip 'i')
    col = chr(ord('A') + move.x)
    if move.x >= 8:  # Skip 'I'
        col = chr(ord('A') + move.x + 1)
    
    return f"{col}{move.y + 1}"
