"""Data models for the Chess OpenEnv environment."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChessAction:
    """
    Represents a chess move action.

    Attributes:
        move: UCI format move string (e.g., "e2e4", "e7e8q" for promotion)
    """

    move: str


@dataclass
class ChessObservation:
    """
    Represents the observable state of the chess environment.

    Attributes:
        fen: Board position in FEN notation
        legal_moves: List of legal moves in UCI format
        is_check: Whether the current player is in check
        done: Whether the episode has ended
        reward: Reward value (1.0 for win, -1.0 for loss, 0.0 for draw, None otherwise)
        result: Game result string if game is over (e.g., "1-0", "0-1", "1/2-1/2")
        metadata: Additional information about the position
    """

    fen: str
    legal_moves: List[str]
    is_check: bool = False
    done: bool = False
    reward: Optional[float] = None
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChessState:
    """
    Tracks episode metadata for the chess environment.

    Attributes:
        episode_id: Unique identifier for the current episode
        step_count: Number of moves (half-moves) played in current episode
        current_player: "white" or "black"
        fen: Current position in FEN notation
        move_history: List of moves played in UCI format
    """

    episode_id: str
    step_count: int
    current_player: str
    fen: str
    move_history: List[str] = field(default_factory=list)


@dataclass
class RewardConfig:
    """
    Configuration for reward shaping in the chess environment.

    Attributes:
        win: Reward for winning the game
        loss: Reward for losing the game
        draw: Reward for drawing the game
        illegal_move: Penalty for attempting an illegal move
        use_evaluation: Whether to include position evaluation in rewards
        evaluation_scale: Scale factor for evaluation-based rewards
    """

    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    illegal_move: float = -0.1
    use_evaluation: bool = False
    evaluation_scale: float = 0.001
