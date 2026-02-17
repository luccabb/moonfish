"""
Classical evaluator wrapping the existing PeSTO evaluation.

This is the default evaluator used by the engine. It provides a baseline
for comparing against neural network evaluators.
"""

from chess import Board

from moonfish.psqt import BOARD_EVALUATION_CACHE, board_evaluation


class ClassicalEvaluator:
    """
    Classical evaluation based on PeSTO piece-square tables with
    tapered midgame/endgame scoring.
    """

    def evaluate(self, board: Board) -> float:
        """Evaluate using PeSTO piece-square tables."""
        return board_evaluation(board)

    def reset(self) -> None:
        """Clear the evaluation cache between searches."""
        BOARD_EVALUATION_CACHE.clear()
