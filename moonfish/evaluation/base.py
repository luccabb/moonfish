"""
Base evaluator protocol.

Any evaluation function (classical, NNUE, transformer, LLM) should implement
this protocol to be usable with the alpha-beta search engine.
"""

from typing import Protocol, runtime_checkable

from chess import Board


@runtime_checkable
class Evaluator(Protocol):
    """
    Protocol for board evaluation functions.

    Implementations can range from simple piece-square tables to neural networks
    or even LLM-based evaluators. The engine will call `evaluate()` at leaf nodes
    of the search tree and in quiescence search.

    The returned score should be from the perspective of the side to move:
    - Positive = good for the side to move
    - Negative = bad for the side to move
    - 0 = roughly equal

    Scores are in centipawns (100 = 1 pawn advantage).
    """

    def evaluate(self, board: Board) -> float:
        """
        Evaluate the given board position.

        Args:
            board: The chess position to evaluate.

        Returns:
            Score in centipawns from the side-to-move's perspective.
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal state (e.g., caches, accumulators).

        Called at the start of each new search. Implementations that
        don't maintain state can make this a no-op.
        """
        ...
