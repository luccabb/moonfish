from random import choice

from chess import Board, Move

from moonfish.config import Config


class RandomEngine:

    def __init__(self, _: Config): ...

    def search_move(self, board: Board) -> Move:
        """
        Returns a random move from the list of legal moves.

        Arguments:
            - board: chess board state.

        Returns:
            - move: a random move from the list of legal moves.
        """
        move = choice([move for move in board.legal_moves])
        return move
