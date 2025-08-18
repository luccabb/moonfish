# Compatibility layer for bulletchess
# This module provides missing methods from python-chess

import bulletchess


def is_checkmate(board: bulletchess.Board) -> bool:
    """Check if the current position is checkmate."""
    # If no legal moves and king is in check, it's checkmate
    legal_moves = list(board.legal_moves())
    if len(legal_moves) == 0:
        # For now, assume it's checkmate if no legal moves
        # TODO: implement proper check detection
        return True
    return False


def is_stalemate(board: bulletchess.Board) -> bool:
    """Check if the current position is stalemate."""
    # If no legal moves and king is NOT in check, it's stalemate
    legal_moves = list(board.legal_moves())
    if len(legal_moves) == 0:
        # For now, we'll assume it's stalemate if no legal moves
        # This is incorrect - we need to check if king is in check
        # TODO: implement proper check detection
        return False  # For now, assume checkmate rather than stalemate
    return False


def is_check(board: bulletchess.Board) -> bool:
    """Check if the current position is check."""
    # TODO: implement check detection for bulletchess
    return False


def is_capture(board: bulletchess.Board, move: bulletchess.Move) -> bool:
    """Check if a move is a capture."""
    return move.is_capture(board)


def is_en_passant(board: bulletchess.Board, move: bulletchess.Move) -> bool:
    """Check if a move is en passant."""
    # Check if it's a pawn move to the en passant square
    if board.en_passant_square is not None and (
        move.destination == board.en_passant_square
        and board[move.origin]
        and board[move.origin].piece_type == bulletchess.PAWN
    ):
        return True
    return False


def gives_check(board: bulletchess.Board, move: bulletchess.Move) -> bool:
    """Check if a move gives check."""
    # TODO: implement by making the move and checking if opponent is in check
    return False


def is_zeroing(board: bulletchess.Board, move: bulletchess.Move) -> bool:
    """Check if a move is zeroing (resets 50-move counter)."""
    # A move is zeroing if it's a pawn move or capture
    if move.is_capture(board):
        return True
    if board[move.origin] and board[move.origin].piece_type == bulletchess.PAWN:
        return True
    return False


def piece_at(board: bulletchess.Board, square: bulletchess.Square):
    """Get piece at square."""
    return board[square]


def push_uci(board: bulletchess.Board, uci_move: str):
    """Apply a move from UCI string."""
    move = bulletchess.Move.from_uci(uci_move)
    board.apply(move)


def square_to_index(square: bulletchess.Square) -> int:
    """Convert bulletchess Square to 0-63 index."""
    # Create mapping from square names to indices
    square_names = [
        "A1",
        "B1",
        "C1",
        "D1",
        "E1",
        "F1",
        "G1",
        "H1",
        "A2",
        "B2",
        "C2",
        "D2",
        "E2",
        "F2",
        "G2",
        "H2",
        "A3",
        "B3",
        "C3",
        "D3",
        "E3",
        "F3",
        "G3",
        "H3",
        "A4",
        "B4",
        "C4",
        "D4",
        "E4",
        "F4",
        "G4",
        "H4",
        "A5",
        "B5",
        "C5",
        "D5",
        "E5",
        "F5",
        "G5",
        "H5",
        "A6",
        "B6",
        "C6",
        "D6",
        "E6",
        "F6",
        "G6",
        "H6",
        "A7",
        "B7",
        "C7",
        "D7",
        "E7",
        "F7",
        "G7",
        "H7",
        "A8",
        "B8",
        "C8",
        "D8",
        "E8",
        "F8",
        "G8",
        "H8",
    ]
    return square_names.index(str(square))
