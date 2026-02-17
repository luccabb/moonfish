"""
NNUE evaluator with incremental accumulator updates.

Implements a dual-perspective efficiently updatable neural network (NNUE)
following the architecture pioneered by Stockfish. The network uses:

- 768 sparse binary input features (6 piece types x 2 colors x 64 squares)
- Dual-perspective feature transformer with shared weights
- SCReLU activation (squared clipped ReLU)
- Single output neuron

Accumulators are updated incrementally during search: each move only
changes 2-4 input features, so we add/subtract weight columns rather
than recomputing the full matrix multiply.
"""

import numpy as np
from chess import Board

import chess

# Feature encoding: 768 binary features
# Index = piece_type_index * 128 + color * 64 + square
# piece_type_index: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
# color: WHITE=0, BLACK=1
# square: 0..63 (a1=0, h8=63)
NUM_FEATURES = 768
HIDDEN_SIZE = 128


def feature_index(piece_type: int, color: bool, square: int) -> int:
    """Compute the feature index for a piece on a square (white perspective)."""
    return (piece_type - 1) * 128 + int(color) * 64 + square


def feature_index_flipped(piece_type: int, color: bool, square: int) -> int:
    """Compute the feature index from black's perspective (flip colors and squares)."""
    flipped_color = not color
    flipped_square = square ^ 56  # Vertical flip: rank 0 <-> rank 7
    return (piece_type - 1) * 128 + int(flipped_color) * 64 + flipped_square


class NNUEEvaluator:
    """
    NNUE evaluator with dual-perspective accumulators and incremental updates.

    Architecture:
        Input (768 sparse binary) -> Linear(768, 128) shared weights
        Two accumulators: white perspective + black perspective
        SCReLU on each -> concat(128+128=256) -> Linear(256, 1) -> output

    The accumulator state can be saved and restored during search to avoid
    full recomputation after board.pop().
    """

    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.w1 = data["w1"].astype(np.float32)  # (768, 128)
        self.b1 = data["b1"].astype(np.float32)  # (128,)
        self.w2 = data["w2"].astype(np.float32)  # (256,) or (256, 1)
        self.b2 = data["b2"].astype(np.float32)  # (1,) or scalar

        # Flatten w2 to 1D for dot product
        self.w2 = self.w2.ravel()
        self.b2 = float(self.b2.ravel()[0])

        # Accumulators: will be set by reset()
        self.white_acc = np.copy(self.b1)
        self.black_acc = np.copy(self.b1)

    def reset(self, board: Board | None = None) -> None:
        """Full recompute of accumulators from board position."""
        self.white_acc = np.copy(self.b1)
        self.black_acc = np.copy(self.b1)
        if board is not None:
            for sq, piece in board.piece_map().items():
                w_idx = feature_index(piece.piece_type, piece.color, sq)
                b_idx = feature_index_flipped(piece.piece_type, piece.color, sq)
                self.white_acc += self.w1[w_idx]
                self.black_acc += self.w1[b_idx]

    def save_accumulators(self) -> tuple[np.ndarray, np.ndarray]:
        """Save current accumulator state (for search stack)."""
        return np.copy(self.white_acc), np.copy(self.black_acc)

    def restore_accumulators(
        self, saved: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Restore accumulator state from saved copy."""
        self.white_acc = saved[0]
        self.black_acc = saved[1]

    def _add_piece(self, piece_type: int, color: bool, square: int) -> None:
        """Add a piece to accumulators."""
        w_idx = feature_index(piece_type, color, square)
        b_idx = feature_index_flipped(piece_type, color, square)
        self.white_acc += self.w1[w_idx]
        self.black_acc += self.w1[b_idx]

    def _remove_piece(self, piece_type: int, color: bool, square: int) -> None:
        """Remove a piece from accumulators."""
        w_idx = feature_index(piece_type, color, square)
        b_idx = feature_index_flipped(piece_type, color, square)
        self.white_acc -= self.w1[w_idx]
        self.black_acc -= self.w1[b_idx]

    def update_move(self, board: Board, move: chess.Move) -> None:
        """
        Incrementally update accumulators for a move.

        Must be called BEFORE board.push(move). The board state is used
        to determine captures, castling, en passant, and promotions.

        Typically 2 feature changes (quiet move), 4 for captures/castling/promotion.
        """
        from_sq = move.from_square
        to_sq = move.to_square
        piece = board.piece_at(from_sq)
        if piece is None:
            return

        moving_type = piece.piece_type
        moving_color = piece.color

        # Handle capture (remove captured piece)
        if board.is_en_passant(move):
            # En passant: captured pawn is on a different square
            if moving_color == chess.WHITE:
                capture_sq = to_sq - 8
            else:
                capture_sq = to_sq + 8
            self._remove_piece(chess.PAWN, not moving_color, capture_sq)
        elif board.is_capture(move):
            captured = board.piece_at(to_sq)
            if captured is not None:
                self._remove_piece(captured.piece_type, captured.color, to_sq)

        # Remove piece from origin square
        self._remove_piece(moving_type, moving_color, from_sq)

        # Handle promotion
        if move.promotion is not None:
            # Add promoted piece to destination
            self._add_piece(move.promotion, moving_color, to_sq)
        else:
            # Add piece to destination
            self._add_piece(moving_type, moving_color, to_sq)

        # Handle castling: also move the rook
        if board.is_castling(move):
            if to_sq > from_sq:
                # Kingside
                rook_from = chess.square(7, chess.square_rank(from_sq))
                rook_to = chess.square(5, chess.square_rank(from_sq))
            else:
                # Queenside
                rook_from = chess.square(0, chess.square_rank(from_sq))
                rook_to = chess.square(3, chess.square_rank(from_sq))
            self._remove_piece(chess.ROOK, moving_color, rook_from)
            self._add_piece(chess.ROOK, moving_color, rook_to)

    def evaluate(self, board: Board) -> float:
        """
        Forward pass with SCReLU activation.

        Returns score from the side-to-move's perspective.
        """
        if board.turn == chess.WHITE:
            stm = self.white_acc
            nstm = self.black_acc
        else:
            stm = self.black_acc
            nstm = self.white_acc

        # SCReLU: clamp(x, 0, 1)^2
        stm_out = np.clip(stm, 0.0, 1.0)
        stm_out = stm_out * stm_out
        nstm_out = np.clip(nstm, 0.0, 1.0)
        nstm_out = nstm_out * nstm_out

        hidden = np.concatenate([stm_out, nstm_out])
        return float(np.dot(hidden, self.w2) + self.b2)
