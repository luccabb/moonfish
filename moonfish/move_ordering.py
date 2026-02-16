import random

from chess import BLACK, Board, Move

from moonfish.psqt import (
    MG_PIECE_VALUES,
    evaluate_capture,
    evaluate_piece,
    get_phase,
)

# Simple integer piece values for fast MVV-LVA ordering in main search
# Index by piece type: 0=None, 1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING
_MVV_LVA_VALUES = (0, 100, 300, 300, 500, 900, 10000)


def organize_moves(board: Board):
    """
    This function receives a board and it returns a list of all the
    possible moves for the current player, sorted by importance.
    Captures are sorted by MVV-LVA (Most Valuable Victim - Least Valuable Attacker).
    Promotions are prioritized. Non-captures are shuffled.

    Arguments:
            - board: chess board state

    Returns:
            - legal_moves: list of all the possible moves for the current player.
    """
    non_captures = []
    captures = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        elif move.promotion is not None:
            # Promotions without capture — prioritize them
            captures.append(move)
        else:
            non_captures.append(move)

    # Sort captures by MVV-LVA: highest victim value first, then lowest attacker
    captures.sort(key=lambda m: _mvv_lva_score(board, m), reverse=True)
    random.shuffle(non_captures)
    return captures + non_captures


def _mvv_lva_score(board: Board, move: Move) -> int:
    """Fast integer MVV-LVA score for move ordering."""
    if move.promotion is not None:
        # Promotions get high score; queen promotion highest
        return _MVV_LVA_VALUES[move.promotion] + 10000

    # Victim value - attacker value (we want high victim, low attacker)
    attacker = board.piece_type_at(move.from_square)
    victim = board.piece_type_at(move.to_square)

    if victim is None:
        # En passant
        return _MVV_LVA_VALUES[1] - _MVV_LVA_VALUES[1]  # pawn captures pawn

    attacker_val = _MVV_LVA_VALUES[attacker] if attacker else 0
    victim_val = _MVV_LVA_VALUES[victim] if victim else 0
    return victim_val * 10 - attacker_val


def is_tactical_move(board: Board, move: Move) -> bool:
    """
    Check if a move is tactical (should be searched in quiescence).

    Tactical moves are:
    - Captures (change material)
    - Promotions (significant material gain)
    - Moves that give check (forcing)
    """
    return (
        board.is_capture(move) or move.promotion is not None or board.gives_check(move)
    )


def organize_moves_quiescence(board: Board):
    """
    This function receives a board and it returns a list of all the
    possible moves for the current player, sorted by importance.

    Only returns tactical moves: captures, promotions, and checks.

    Arguments:
            - board: chess board state

    Returns:
            - moves: list of tactical moves sorted by importance (MVV-LVA).
    """
    phase = get_phase(board)

    # Filter only tactical moves for quiescence search
    # (captures, promotions, checks - NOT quiet pawn pushes)
    tactical_moves = filter(
        lambda move: is_tactical_move(board, move),
        board.legal_moves,
    )

    # Sort moves by importance using MVV-LVA
    moves = sorted(
        tactical_moves,
        key=lambda move: mvv_lva(board, move, phase),
        reverse=(board.turn == BLACK),
    )
    return moves


def mvv_lva(board: Board, move: Move, phase: float) -> float:
    """
    This function receives a board and a move and it returns the
    move's value based on the phase of the game. It's based on the
    idea that the most valuable victim being captured by the least
    valuable attacker is the best move.

    Arguments:
            - board: chess board state
            - move: chess move
            - phase: current phase of the game

    Returns:
            - mvv_lva: value of the move
    """
    move_value: float = 0

    # evaluating position
    from_value = evaluate_piece(board, move.from_square, phase)
    to_value = evaluate_piece(board, move.to_square, phase)

    move_value += to_value - from_value

    # evaluating capture
    if board.is_capture(move):
        move_value += evaluate_capture(board, move, phase)

    return -move_value if board.turn else move_value
