from dataclasses import dataclass

# Score for checkmate.
CHECKMATE_SCORE = 10**8
# Threshold to differentiate checkmates from other moves.
CHECKMATE_THRESHOLD = 999 * (10**4)


@dataclass(slots=True)
class Config:
    """
    Configuration for the engine.
    """

    mode: str
    algorithm: str
    negamax_depth: int
    null_move: bool
    null_move_r: int
    quiescence_search_depth: int
    syzygy_path: str | None
    syzygy_pieces: int
    checkmate_score: int = CHECKMATE_SCORE
    checkmate_threshold: int = CHECKMATE_THRESHOLD
