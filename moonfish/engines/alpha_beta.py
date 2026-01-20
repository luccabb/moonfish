from copy import copy
from enum import IntEnum
from multiprocessing.managers import DictProxy
from typing import Dict, Optional, Tuple, Union

import chess.polyglot
import chess.syzygy
from chess import Board, Move
from moonfish.config import Config
from moonfish.engines.random import choice
from moonfish.move_ordering import organize_moves, organize_moves_quiescence
from moonfish.psqt import board_evaluation, count_pieces


class Bound(IntEnum):
    """Transposition table bound types."""

    EXACT = 0  # Score is exact (PV node, score was within alpha-beta window)
    LOWER_BOUND = 1  # Score is at least this value (failed high / beta cutoff)
    UPPER_BOUND = 2  # Score is at most this value (failed low)


# Depth value for terminal positions (checkmate/stalemate) - always usable
DEPTH_MAX = 10000

# Cache: zobrist_hash -> (score, best_move, bound_type, depth)
CACHE_TYPE = Dict[int, Tuple[Union[float, int], Optional[Move], Bound, int]]


class AlphaBeta:
    """
    A class that implements alpha-beta search algorithm.
    """

    def __init__(self, config: Config):
        self.config = config

        # Open Syzygy tablebase once at initialization (not on every eval)
        self.tablebase = None
        if config.syzygy_path:
            try:
                self.tablebase = chess.syzygy.open_tablebase(config.syzygy_path)
            except Exception:
                # Tablebase path invalid or not accessible
                self.tablebase = None

    def close(self):
        """Close the tablebase if open. Call when done with the engine."""
        if self.tablebase is not None:
            self.tablebase.close()
            self.tablebase = None

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

        # Use pre-opened tablebase for endgame positions
        if pieces <= self.config.syzygy_pieces and self.tablebase is not None:
            try:
                dtz = self.tablebase.probe_dtz(board)
                return dtz
            except (chess.syzygy.MissingTableError, KeyError):
                # Position not in tablebase, fall through to normal evaluation
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
        cache: DictProxy | CACHE_TYPE,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> Tuple[float | int, Optional[Move]]:
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
        original_alpha = alpha
        cache_key = chess.polyglot.zobrist_hash(board)

        # Check transposition table
        if cache_key in cache:
            cached_score, cached_move, cached_bound, cached_depth = cache[cache_key]

            # Only use score if cached search was at least as deep as we need
            # Use cached result if:
            # - EXACT: score is exact
            # - LOWER_BOUND and score >= beta: true score is at least cached, causes cutoff
            # - UPPER_BOUND and score <= alpha: true score is at most cached, no improvement
            if cached_depth >= depth and (
                cached_bound == Bound.EXACT
                or (cached_bound == Bound.LOWER_BOUND and cached_score >= beta)
                or (cached_bound == Bound.UPPER_BOUND and cached_score <= alpha)
            ):
                return cached_score, cached_move

        if board.is_checkmate():
            cache[cache_key] = (
                -self.config.checkmate_score,
                None,
                Bound.EXACT,
                DEPTH_MAX,
            )
            return (-self.config.checkmate_score, None)

        if board.is_stalemate():
            cache[cache_key] = (0, None, Bound.EXACT, DEPTH_MAX)
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
            cache[cache_key] = (board_score, None, Bound.EXACT, depth)
            return board_score, None

        # null move pruning
        if (
            self.config.null_move
            and null_move
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
                    # Null move confirmed beta cutoff - this is a lower bound
                    cache[cache_key] = (beta, None, Bound.LOWER_BOUND, depth)
                    return beta, None

        best_move = None
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

            # update best move
            if board_score > best_score:
                best_score = board_score
                best_move = move

            # beta-cutoff: opponent won't allow this position
            if best_score >= beta:
                # LOWER_BOUND: true score is at least best_score
                cache[cache_key] = (best_score, best_move, Bound.LOWER_BOUND, depth)
                return best_score, best_move

            # update alpha
            alpha = max(alpha, best_score)

        # if no best move, make a random one
        if not best_move:
            best_move = self.random_move(board)

        # Determine bound type based on whether we improved alpha
        if best_score <= original_alpha:
            # Failed low: we didn't find anything better than what we already had
            bound = Bound.UPPER_BOUND
        else:
            # Score is exact: we found a score within the window
            bound = Bound.EXACT

        cache[cache_key] = (best_score, best_move, bound, depth)
        return best_score, best_move

    def search_move(self, board: Board) -> Move:
        # create shared cache
        cache: CACHE_TYPE = {}

        best_move = self.negamax(
            board, copy(self.config.negamax_depth), self.config.null_move, cache
        )[1]
        assert best_move is not None, "Best move from root should not be None"
        return best_move
