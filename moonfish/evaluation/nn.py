"""
Neural network evaluator framework.

Supports loading evaluation models in multiple formats:
- ONNX Runtime (.onnx) - recommended for deployment
- PyTorch (.pt, .pth) - for development and fine-tuning
- Custom callables - for LLMs or other exotic evaluators

The framework handles board-to-tensor conversion and model inference.
To use a custom model, either:

1. Provide an ONNX or PyTorch model file:
   evaluator = NNEvaluator.from_file("model.onnx")

2. Provide a custom callable:
   evaluator = NNEvaluator(eval_fn=my_llm_eval_function)

3. Subclass and override `_raw_evaluate`:
   class MyEvaluator(NNEvaluator):
       def _raw_evaluate(self, board):
           return my_model(board)

Board representation for neural models:
- Default: 12-plane bitboard (6 piece types x 2 colors) + metadata
- Each plane is 8x8 = 64 values (1 if piece present, 0 otherwise)
- Metadata: side to move, castling rights, en passant, halfmove clock
- Total input size: 12*64 + 5 = 773 floats
"""

from typing import Callable

import chess
from chess import Board


def board_to_tensor(board: Board) -> list[float]:
    """
    Convert a board to a flat feature vector suitable for neural network input.

    Encoding (773 features):
    - 12 bitboard planes (6 piece types x 2 colors), each 64 values: [0..767]
    - Side to move (1 = white, 0 = black): [768]
    - Castling rights (4 bools): [769..772]

    The board is always encoded from white's perspective. If it's black's turn,
    the model output should be negated by the evaluator.

    Args:
        board: The chess position to encode.

    Returns:
        List of 773 floats representing the position.
    """
    features: list[float] = []

    # 12 bitboard planes: WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):  # PAWN=1 through KING=6
            bb = board.pieces_mask(piece_type, color)
            for square in range(64):
                features.append(1.0 if bb & (1 << square) else 0.0)

    # Metadata
    features.append(1.0 if board.turn == chess.WHITE else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    return features


# Input size for the default board encoding
INPUT_SIZE = 773


class NNEvaluator:
    """
    Neural network evaluator with support for multiple model backends.

    This evaluator can use:
    - ONNX models (via onnxruntime)
    - PyTorch models (via torch)
    - Custom callables (e.g., LLM API calls)

    The framework handles board encoding and score normalization.
    Models should output a single float: positive = white is better.
    The evaluator automatically negates for black's perspective.

    Example usage:
        # With an ONNX model:
        evaluator = NNEvaluator.from_file("model.onnx")

        # With a custom function (e.g., LLM):
        def llm_eval(board: Board) -> float:
            prompt = f"Evaluate this chess position: {board.fen()}"
            return call_llm(prompt)  # returns centipawn score

        evaluator = NNEvaluator(eval_fn=llm_eval)

        # With a custom board encoder:
        evaluator = NNEvaluator(
            eval_fn=my_fn,
            board_encoder=my_custom_encoder,
        )
    """

    def __init__(
        self,
        eval_fn: Callable[[Board], float] | None = None,
        board_encoder: Callable[[Board], list[float]] | None = None,
    ):
        """
        Create a neural network evaluator.

        Args:
            eval_fn: Optional callable that takes a Board and returns a score
                from white's perspective. If provided, this is used directly
                instead of a model file.
            board_encoder: Optional custom board-to-feature-vector function.
                Defaults to the standard 773-feature encoding.
        """
        self._eval_fn = eval_fn
        self._board_encoder = board_encoder or board_to_tensor
        self._model = None
        self._backend: str | None = None

    @classmethod
    def from_file(cls, model_path: str, **kwargs) -> "NNEvaluator":
        """
        Load a neural network model from a file.

        Supported formats:
        - .onnx: Loaded via ONNX Runtime
        - .pt, .pth: Loaded via PyTorch

        Args:
            model_path: Path to the model file.
            **kwargs: Additional arguments passed to the model loader.

        Returns:
            NNEvaluator instance with the model loaded.
        """
        evaluator = cls(**kwargs)
        evaluator._load_model(model_path)
        return evaluator

    def _load_model(self, model_path: str) -> None:
        """Load a model from file, auto-detecting the backend."""
        if model_path.endswith(".onnx"):
            self._load_onnx(model_path)
        elif model_path.endswith((".pt", ".pth")):
            self._load_pytorch(model_path)
        else:
            raise ValueError(
                f"Unsupported model format: {model_path}. "
                "Supported: .onnx, .pt, .pth"
            )

    def _load_onnx(self, model_path: str) -> None:
        """Load an ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX models. "
                "Install it with: pip install onnxruntime"
            )
        self._model = ort.InferenceSession(model_path)
        self._backend = "onnx"

    def _load_pytorch(self, model_path: str) -> None:
        """Load a PyTorch model."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for PyTorch models. "
                "Install it with: pip install torch"
            )
        self._model = torch.jit.load(model_path)
        self._model.eval()
        self._backend = "pytorch"

    def _raw_evaluate(self, board: Board) -> float:
        """
        Get the raw model output for a position.

        Override this method in subclasses for custom evaluation logic.
        The returned score should be from white's perspective.

        Args:
            board: The chess position to evaluate.

        Returns:
            Score from white's perspective (positive = white is better).
        """
        if self._eval_fn is not None:
            return self._eval_fn(board)

        if self._model is None:
            raise RuntimeError(
                "No model loaded. Use NNEvaluator.from_file() or provide eval_fn."
            )

        features = self._board_encoder(board)

        if self._backend == "onnx":
            import numpy as np

            input_array = np.array([features], dtype=np.float32)
            input_name = self._model.get_inputs()[0].name
            result = self._model.run(None, {input_name: input_array})
            return float(result[0][0][0])

        elif self._backend == "pytorch":
            import torch

            input_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                result = self._model(input_tensor)
            return float(result.item())

        raise RuntimeError(f"Unknown backend: {self._backend}")

    def evaluate(self, board: Board) -> float:
        """
        Evaluate a position from the side-to-move's perspective.

        The raw model output is from white's perspective. This method
        negates the score when it's black's turn.

        Args:
            board: The chess position to evaluate.

        Returns:
            Score in centipawns from the side-to-move's perspective.
        """
        score = self._raw_evaluate(board)
        # Negate for black's perspective (model outputs white-relative scores)
        if board.turn == chess.BLACK:
            score = -score
        return score

    def reset(self) -> None:
        """Reset internal state. No-op for stateless NN evaluators."""
        pass
