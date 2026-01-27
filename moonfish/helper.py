import os
from enum import Enum
from pathlib import Path

from chess import Board, Move, polyglot

from moonfish.config import Config
from moonfish.engines.alpha_beta import AlphaBeta
from moonfish.engines.base_engine import ChessEngine
from moonfish.engines.l1p_alpha_beta import Layer1ParallelAlphaBeta
from moonfish.engines.l2p_alpha_beta import Layer2ParallelAlphaBeta
from moonfish.engines.lazy_smp import LazySMP
from moonfish.engines.random import RandomEngine


class Algorithm(Enum):
    """Enumeration of all possible algorithms."""

    alpha_beta = "alpha_beta"
    parallel_alpha_beta_layer_1 = "parallel_alpha_beta_layer_1"
    parallel_alpha_beta_layer_2 = "parallel_alpha_beta_layer_2"
    lazy_smp = "lazy_smp"
    random = "random"


def get_engine(config: Config):
    """
    Returns the engine

    Arguments:
        - algorithm_name: the name of the algorithm we want to use.

    Returns:
        - engine: the engine we want to use.
    """
    algorithm = Algorithm[config.algorithm]

    if algorithm is Algorithm.alpha_beta:
        return AlphaBeta(config)
    elif algorithm is Algorithm.parallel_alpha_beta_layer_1:
        return Layer1ParallelAlphaBeta(config)
    elif algorithm is Algorithm.parallel_alpha_beta_layer_2:
        return Layer2ParallelAlphaBeta(config)
    elif algorithm is Algorithm.lazy_smp:
        return LazySMP(config)
    elif algorithm is Algorithm.random:
        return RandomEngine(config)
    raise Exception("algorithm not supported")


def _opening_book_path() -> str:
    """
    Resolve the opening book path, allowing overrides for CI or custom installs.
    """
    env_path = os.getenv("MOONFISH_OPENING_BOOK")
    if env_path and Path(env_path).is_file():
        return env_path

    cwd_path = Path("opening_book/cerebellum.bin")
    if cwd_path.is_file():
        return str(cwd_path)

    repo_path = Path(__file__).resolve().parents[1] / "opening_book" / "cerebellum.bin"
    if repo_path.is_file():
        return str(repo_path)

    return str(cwd_path)


def find_best_move(board: Board, engine: ChessEngine) -> Move:
    """
    Finds the best move for the given board using the given engine.

    Arguments:
        - board: the chess board state.
        - engine: the engine to use for finding the best move.

    Returns:
        - best_move: the best move found by the engine.
    """
    # try using cerebellum opening book: https://zipproth.de/Brainfish/download/
    # if it fails we search on our engine. The first (12-20) moves should be
    # available in the opening book, so our engine starts playing after that.
    try:
        best_move = polyglot.MemoryMappedReader(_opening_book_path()).find(board).move
    except (ValueError, OSError, AttributeError, IndexError):
        best_move = engine.search_move(board)
    return best_move
# Trigger benchmark 1769542464
