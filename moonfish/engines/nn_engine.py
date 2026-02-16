"""
Neural network chess engine.

Uses alpha-beta search with a pluggable neural network evaluator instead of
the classical PeSTO evaluation. Any evaluator implementing the Evaluator
protocol can be used.

Example:
    from moonfish.evaluation.nn import NNEvaluator

    # Load a custom ONNX model
    evaluator = NNEvaluator.from_file("my_model.onnx")
    engine = NNEngine(config, evaluator=evaluator)
    best_move = engine.search_move(board)

    # Or use an LLM for evaluation
    def llm_eval(board):
        return call_my_llm(board.fen())

    evaluator = NNEvaluator(eval_fn=llm_eval)
    engine = NNEngine(config, evaluator=evaluator)
"""

from chess import Board, Move

from moonfish.config import Config
from moonfish.engines.alpha_beta import AlphaBeta
from moonfish.evaluation.base import Evaluator
from moonfish.evaluation.classical import ClassicalEvaluator


class NNEngine(AlphaBeta):
    """
    Chess engine that uses a pluggable evaluator for position assessment.

    Inherits the full alpha-beta search from AlphaBeta but replaces the
    evaluation function with a provided Evaluator instance. This allows
    using neural networks, LLMs, or any custom evaluation function while
    keeping the same search algorithm.
    """

    def __init__(self, config: Config, evaluator: Evaluator | None = None):
        """
        Initialize the NN engine.

        Args:
            config: Engine configuration.
            evaluator: An Evaluator instance. If None, falls back to
                ClassicalEvaluator (PeSTO tables).
        """
        super().__init__(config)
        self.evaluator = evaluator or ClassicalEvaluator()

    def eval_board(self, board: Board) -> float:
        """
        Evaluate the board using the plugged-in evaluator.

        If Syzygy tablebases are available and the position qualifies,
        tablebase probing takes priority over the evaluator.

        Args:
            board: The chess position to evaluate.

        Returns:
            Score from the side-to-move's perspective.
        """
        # Syzygy probing still takes priority for endgame positions
        if self.tablebase is not None:
            from moonfish.psqt import count_pieces

            pieces = sum(count_pieces(board))
            if pieces <= self.config.syzygy_pieces:
                try:
                    import chess.syzygy

                    dtz = self.tablebase.probe_dtz(board)
                    return dtz
                except Exception:
                    pass

        return self.evaluator.evaluate(board)

    def search_move(self, board: Board) -> Move:
        """Search for the best move, resetting evaluator state first."""
        self.evaluator.reset()
        return super().search_move(board)
