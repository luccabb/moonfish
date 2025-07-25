import random

from chess import BLACK, Board, Move

from moonfish.psqt import evaluate_capture, evaluate_piece, get_phase


def organize_moves(board: Board):
    """
    This function receives a board and it returns a list of all the
    possible moves for the current player, sorted by importance.
    It sends capturing moves at the starting positions in
    the array (to try to increase pruning and do so earlier).

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
        else:
            non_captures.append(move)

    random.shuffle(captures)
    random.shuffle(non_captures)
    return captures + non_captures


def organize_moves_quiescence(board: Board):
    """
    This function receives a board and it returns a list of all the
    possible moves for the current player, sorted by importance.

    Arguments:
            - board: chess board state

    Returns:
            - moves: list of all the possible moves for the current player sorted based on importance.
    """
    phase = get_phase(board)
    # filter only important moves for quiescence search
    captures = filter(
        lambda move: board.is_zeroing(move) or board.gives_check(move),
        board.legal_moves,
    )
    # sort moves by importance
    moves = sorted(
        captures,
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
