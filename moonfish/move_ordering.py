import random

from chess import BLACK, Board, Move

from moonfish.psqt import evaluate_capture, evaluate_piece, get_phase


def organize_moves(
    board: Board,
    killers: "list[Move | None] | None" = None,
):
    """
    This function receives a board and it returns a list of all the
    possible moves for the current player, sorted by importance.
    Order: captures first, then killer moves, then remaining quiet moves.

    Arguments:
            - board: chess board state
            - killers: list of killer moves for the current ply (tried after captures)

    Returns:
            - legal_moves: list of all the possible moves for the current player.
    """
    non_captures = []
    captures = []
    killer_set = set()

    # Build set of killer moves for fast lookup
    if killers is not None:
        for k in killers:
            if k is not None:
                killer_set.add(k)

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        elif move in killer_set:
            # Skip killers from non_captures; we'll insert them after captures
            continue
        else:
            non_captures.append(move)

    random.shuffle(captures)
    random.shuffle(non_captures)

    # Insert legal killer moves between captures and quiet moves
    killer_moves = []
    if killers is not None:
        legal_moves_set = set(board.legal_moves)
        for k in killers:
            if k is not None and k in legal_moves_set and not board.is_capture(k):
                killer_moves.append(k)

    return captures + killer_moves + non_captures


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
