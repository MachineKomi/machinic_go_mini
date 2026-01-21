"""Monte Carlo Tree Search implementation inspired by AlphaZero."""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .game import GoGame, Move, Stone


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    game: GoGame
    parent: Optional["MCTSNode"] = None
    move: Optional[Move] = None  # Move that led to this node
    prior: float = 0.0  # Prior probability from policy network
    
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)  # move_index -> child
    visit_count: int = 0
    value_sum: float = 0.0
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0
    
    def ucb_score(self, c_puct: float = 1.5) -> float:
        """Calculate UCB score for node selection."""
        if self.parent is None:
            return 0.0
        
        # Exploration term
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        # Value from parent's perspective (negate because alternating players)
        return -self.value + exploration


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""
    
    def __init__(
        self,
        network,
        c_puct: float = 1.5,
        num_simulations: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
    ):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
    
    def search(self, game: GoGame, add_noise: bool = True) -> Tuple[np.ndarray, float]:
        """Run MCTS and return visit count distribution and root value.
        
        Args:
            game: Current game state
            add_noise: Whether to add Dirichlet noise at root (for training)
            
        Returns:
            policy: Visit count distribution over moves
            value: Root node value estimate
        """
        root = MCTSNode(game=game.copy())
        
        # Expand root
        self._expand(root)
        
        # Add Dirichlet noise at root for exploration
        if add_noise and root.is_expanded:
            self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree using UCB
            while node.is_expanded and not node.game.is_game_over():
                node = self._select_child(node)
                search_path.append(node)
            
            # Get value
            if node.game.is_game_over():
                # Terminal node: use actual game result
                value = node.game.get_result()
                # Adjust for current player
                if node.game.current_player == Stone.WHITE:
                    value = -value
            else:
                # Expand and evaluate
                value = self._expand(node)
            
            # Backpropagate
            self._backpropagate(search_path, value)
        
        # Build policy from visit counts
        policy = self._get_policy(root)
        
        return policy, root.value

    def _expand(self, node: MCTSNode) -> float:
        """Expand a node using the neural network.
        
        Returns the value estimate for the position.
        """
        features = node.game.to_features()
        policy, value = self.network.predict(features)
        
        # Mask invalid moves
        valid_mask = node.game.get_valid_moves_mask()
        policy = policy * valid_mask
        
        # Renormalize
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback to uniform over valid moves
            policy = valid_mask.astype(np.float32)
            policy = policy / policy.sum()
        
        # Create children for valid moves
        board_size = node.game.board_size
        for move_idx in range(len(policy)):
            if valid_mask[move_idx]:
                move = Move.from_index(move_idx, board_size)
                child_game = node.game.copy()
                child_game.play_move(move)
                
                child = MCTSNode(
                    game=child_game,
                    parent=node,
                    move=move,
                    prior=policy[move_idx],
                )
                node.children[move_idx] = child
        
        return value
    
    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root node priors for exploration."""
        num_children = len(node.children)
        if num_children == 0:
            return
        
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)
        
        for i, child in enumerate(node.children.values()):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = float("-inf")
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Backpropagate value through the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for alternating players
    
    def _get_policy(self, root: MCTSNode) -> np.ndarray:
        """Get policy from visit counts with temperature."""
        board_size = root.game.board_size
        num_moves = board_size * board_size + 1
        
        visits = np.zeros(num_moves, dtype=np.float32)
        for move_idx, child in root.children.items():
            visits[move_idx] = child.visit_count
        
        if self.temperature == 0:
            # Deterministic: pick best move
            policy = np.zeros_like(visits)
            policy[np.argmax(visits)] = 1.0
        else:
            # Apply temperature
            visits = visits ** (1.0 / self.temperature)
            total = visits.sum()
            if total > 0:
                policy = visits / total
            else:
                policy = np.ones_like(visits) / num_moves
        
        return policy
    
    def get_move(self, game: GoGame, add_noise: bool = False) -> Tuple[Move, np.ndarray, float]:
        """Get best move for current position.
        
        Returns:
            move: Selected move
            policy: MCTS policy distribution
            value: Position evaluation
        """
        policy, value = self.search(game, add_noise=add_noise)
        
        if self.temperature == 0:
            move_idx = np.argmax(policy)
        else:
            move_idx = np.random.choice(len(policy), p=policy)
        
        move = Move.from_index(move_idx, game.board_size)
        return move, policy, value


class RandomPlayer:
    """Random player for baseline comparison."""
    
    def get_move(self, game: GoGame) -> Move:
        """Select a random valid move."""
        valid_moves = game.get_valid_moves()
        return np.random.choice(valid_moves)


class GreedyPlayer:
    """Greedy player using only network policy (no search)."""
    
    def __init__(self, network):
        self.network = network
    
    def get_move(self, game: GoGame) -> Move:
        """Select move with highest policy probability."""
        features = game.to_features()
        policy, _ = self.network.predict(features)
        
        # Mask invalid moves
        valid_mask = game.get_valid_moves_mask()
        policy = policy * valid_mask
        
        move_idx = np.argmax(policy)
        return Move.from_index(move_idx, game.board_size)
