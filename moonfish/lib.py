from typing import Optional

from chess import Board, Move

from moonfish.config import Config
from moonfish.helper import Algorithm, find_best_move, get_engine


def search_move(
    board: Board,
    algorithm: str = Algorithm.alpha_beta.value,
    depth: int = 3,
    null_move: bool = False,
    null_move_r: int = 2,
    quiescence_search_depth: int = 3,
    syzygy_path: Optional[str] = None,
    syzygy_pieces: int = 5,
) -> Move:
    """
    Searches for the best move in the given board state using the specified engine.

    Arguments:
        - board: The current chess board state.
        - engine: The chess engine to use for searching.
        - depth: The maximum depth to search.
        - null_move: Whether to use null move pruning.

    Returns:
        - The best move found in UCI format, or None if no moves are available.
    """
    config = Config(
        mode="",
        algorithm=algorithm,
        negamax_depth=depth,
        null_move=null_move,
        null_move_r=null_move_r,
        quiescence_search_depth=quiescence_search_depth,
        syzygy_path=syzygy_path,
        syzygy_pieces=syzygy_pieces,
    )
    engine = get_engine(config)
    return find_best_move(board=board, engine=engine)
