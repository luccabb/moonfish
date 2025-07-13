from copy import copy
from multiprocessing.managers import DictProxy
from typing import Dict, Optional, Tuple

import chess.syzygy
from chess import Board, Move

from moonfish.config import Config
from moonfish.engines.random import choice
from moonfish.move_ordering import organize_moves, organize_moves_quiescence
from moonfish.psqt import board_evaluation, count_pieces

CACHE_KEY = Dict[Tuple[str, int, bool, float, float], Tuple[float | int, Optional[str]]]


class AlphaBeta:
    """
    A class that implements alpha-beta search algorithm.
    """

    def __init__(self, config: Config):
        self.config = config

    def random_move(self, board: Board) -> Move:
        move = choice([move for move in board.legal_moves])
        return move

    def eval_board(self, board: Board) -> float:
        """
        This function evaluates the board based on the current
        position of the pieces and returns a score for the board.

        Arguments:
            - board: chess board state

        Returns:
            - score: the score for the current board
        """
        pieces = sum(count_pieces(board))

        if pieces <= self.config.syzygy_pieces and self.config.syzygy_path:
            with chess.syzygy.open_tablebase(self.config.syzygy_path) as tablebase:
                try:
                    eval = tablebase.probe_dtz(board)
                    return eval
                except (chess.syzygy.MissingTableError, KeyError):
                    # syzygy tablebase position not found, continue with evaluation
                    pass

        return board_evaluation(board)

    def quiescence_search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """
        This functions extends our search for important
        positions (such as: captures, pawn moves, promotions),
        by using a reduced search tree.

        Arguments:
            - board: chess board state
            - alpha: best score for the maximizing player (best choice
                (highest value)  we've found along the path for max)
            - beta: best score for the minimizing player (best choice
                (lowest value) we've found along the path for min).
                When Alpha is higher than or equal to Beta, we can prune
                the search tree;    because it means that the maximizing
                player won't find a better move in this branch.
            - depth: how many depths we want to calculate for this board

        Returns:
            - best_score: returns best move's score.
        """
        if board.is_stalemate():
            return 0

        if board.is_checkmate():
            return -self.config.checkmate_score

        stand_pat = self.eval_board(board)

        # recursion base case
        if depth == 0:
            return stand_pat

        # beta-cutoff
        if stand_pat >= beta:
            return beta

        # alpha update
        alpha = max(alpha, stand_pat)

        # get moves for quiescence search
        moves = organize_moves_quiescence(board)

        for move in moves:
            # make move and get score
            board.push(move)
            score = -self.quiescence_search(
                board=board,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
            )
            board.pop()

            # beta-cutoff
            if score >= beta:
                return beta

            # alpha-update
            alpha = max(alpha, score)

        return alpha

    def negamax(
        self,
        board: Board,
        depth: int,
        null_move: bool,
        cache: DictProxy | CACHE_KEY,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> Tuple[float | int, Optional[str]]:
        """
        This functions receives a board, depth and a player; and it returns
        the best move for the current board based on how many depths we're looking ahead
        and which player is playing. Alpha and beta are used to prune the search tree.
        Alpha represents the best score for the maximizing player (best choice (highest value)  we've found
        along the path for max) and beta represents the best score for the minimizing player
        (best choice (lowest value) we've found along the path for min). When Alpha is higher
        than or equal to Beta, we can prune the search tree; because it means that the
        maximizing player won't find a better move in this branch.

        OBS:
            - We only need to evaluate the value for leaf nodes because they are our final states
            of the board and therefore we need to use their values to base our decision of what is
            the best move.

        Arguments:
            - board: chess board state
            - depth: how many depths we want to calculate for this board
            - null_move: if we want to use null move pruning
            - cache: a shared hash table to store the best
                move for each board state and depth.
            - alpha: best score for the maximizing player (best choice
                (highest value)  we've found along the path for max)
            - beta: best score for the minimizing player (best choice
                (lowest value) we've found along the path for min).

        Returns:
            - best_score, best_move: returns best move that it found and its value.
        """
        cache_key = (board.fen(), depth, null_move, alpha, beta)
        # check if board was already evaluated
        if cache_key in cache:
            return cache[cache_key]

        if board.is_checkmate():
            cache[cache_key] = (-self.config.checkmate_score, None)
            return (-self.config.checkmate_score, None)

        if board.is_stalemate():
            cache[cache_key] = (0, None)
            return (0, None)

        # recursion base case
        if depth <= 0:
            # evaluate current board
            board_score = self.quiescence_search(
                board=board,
                depth=copy(self.config.quiescence_search_depth),
                alpha=alpha,
                beta=beta,
            )
            cache[cache_key] = (board_score, None)
            return board_score, None

        # null move prunning
        if (
            self.config.null_move
            and depth >= (self.config.null_move_r + 1)
            and not board.is_check()
        ):
            board_score = self.eval_board(board)
            if board_score >= beta:
                board.push(Move.null())
                board_score = -self.negamax(
                    board=board,
                    depth=depth - 1 - self.config.null_move_r,
                    null_move=False,
                    cache=cache,
                    alpha=-beta,
                    beta=-beta + 1,
                )[0]
                board.pop()
                if board_score >= beta:
                    cache[cache_key] = (beta, None)
                    return beta, None

        best_move = None

        # initializing best_score
        best_score = float("-inf")
        moves = organize_moves(board)

        for move in moves:
            # make the move
            board.push(move)

            board_score = -self.negamax(
                board=board,
                depth=depth - 1,
                null_move=null_move,
                cache=cache,
                alpha=-beta,
                beta=-alpha,
            )[0]
            if board_score > self.config.checkmate_threshold:
                board_score -= 1
            if board_score < -self.config.checkmate_threshold:
                board_score += 1

            # take move back
            board.pop()

            # beta-cutoff
            if board_score >= beta:
                cache[cache_key] = (board_score, move)
                return board_score, move

            # update best move
            if board_score > best_score:
                best_score = board_score
                best_move = move

            # setting alpha variable to do pruning
            alpha = max(alpha, board_score)

            # alpha beta pruning when we already found a solution that is at least as
            # good as the current one those branches won't be able to influence the
            # final decision so we don't need to waste time analyzing them
            if alpha >= beta:
                break

        # if no best move, make a random one
        if not best_move:
            best_move = self.random_move(board).uci()

        # save result before returning
        cache[cache_key] = (best_score, best_move)
        return best_score, best_move

    def search_move(self, board: Board) -> Optional[str]:
        # create shared cache
        cache: CACHE_KEY = {}

        return self.negamax(
            board, copy(self.config.negamax_depth), self.config.null_move, cache
        )[1]
