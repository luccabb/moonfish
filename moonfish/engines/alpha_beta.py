from copy import copy
from enum import IntEnum
from multiprocessing.managers import DictProxy

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
CACHE_TYPE = dict[int, tuple[float | int, Move | None, Bound, int]]

INF = float("inf")
NEG_INF = float("-inf")
NULL_MOVE = Move.null()


class AlphaBeta:
    """
    A class that implements alpha-beta search algorithm.
    """

    def __init__(self, config: Config):
        self.config = config
        self.nodes: int = 0

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
        # Short-circuit: only count pieces if a tablebase is loaded
        if self.tablebase is not None:
            pieces = sum(count_pieces(board))
            if pieces <= self.config.syzygy_pieces:
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
        in_check = board.is_check()

        self.nodes += 1

        if board.is_checkmate():
            return -self.config.checkmate_score

        if board.is_stalemate():
            return 0

        # Draw detection: fifty-move rule, insufficient material
        # Note: Repetition is checked after making moves, not here
        if board.is_fifty_moves() or board.is_insufficient_material():
            return 0

        stand_pat = self.eval_board(board)

        # When in check, we can't use stand-pat for pruning (position is unstable)
        # We must search all evasions. However, still respect depth limit.
        if in_check:
            # In check: search all evasions, but don't use stand-pat for cutoffs
            if depth <= 0:
                # At depth limit while in check: return evaluation
                # (not ideal but prevents infinite recursion)
                return stand_pat

            best_score = NEG_INF
            moves = list(board.legal_moves)  # All evasions
        else:
            # Not in check: normal quiescence behavior
            # recursion base case
            if depth <= 0:
                return stand_pat

            # beta-cutoff: position is already good enough
            if stand_pat >= beta:
                return beta

            # Use stand-pat as baseline (we can always choose not to capture)
            best_score = stand_pat
            alpha = max(alpha, stand_pat)

            # Only tactical moves when not in check
            moves = organize_moves_quiescence(board)

        for move in moves:
            # make move and get score
            board.push(move)

            # Check if this move leads to a repetition (draw)
            if board.is_repetition(2):
                score = 0.0  # Draw score
            else:
                score = -self.quiescence_search(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                )

            board.pop()

            if score > best_score:
                best_score = score

            # beta-cutoff
            if score >= beta:
                return beta

            # alpha-update
            alpha = max(alpha, score)

        return best_score

    def negamax(
        self,
        board: Board,
        depth: int,
        null_move: bool,
        cache: DictProxy | CACHE_TYPE,
        alpha: float = NEG_INF,
        beta: float = INF,
        ply: int = 0,
        killers: list | None = None,
    ) -> tuple[float | int, Move | None]:
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
        tt_move = None  # Best move from transposition table (for move ordering)

        self.nodes += 1

        # Check transposition table
        if cache_key in cache:
            cached_score, cached_move, cached_bound, cached_depth = cache[cache_key]

            # Save TT move for ordering even if we can't use the score
            tt_move = cached_move

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
                board,
                self.config.quiescence_search_depth,
                alpha,
                beta,
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
                board.push(NULL_MOVE)
                board_score = -self.negamax(
                    board,
                    depth - 1 - self.config.null_move_r,
                    False,
                    cache,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    killers,
                )[0]
                board.pop()
                if board_score >= beta:
                    # Null move confirmed beta cutoff - this is a lower bound
                    cache[cache_key] = (beta, None, Bound.LOWER_BOUND, depth)
                    return beta, None

        best_move = None
        best_score = NEG_INF
        ply_killers = killers[ply] if killers and ply < len(killers) else None
        moves = organize_moves(board, ply_killers)

        # Put TT move first if available (best move from previous search)
        if tt_move is not None and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)

        in_check = board.is_check()

        # Futility pruning: if static eval is far below alpha, quiet moves won't help
        # Only compute if we might use it (low depth, not in check)
        futility_margin = 0
        can_futility_prune = False
        if depth <= 2 and not in_check:
            static_eval = self.eval_board(board)
            # Margin increases with depth: depth 1 = 100cp, depth 2 = 200cp
            futility_margin = 100 * depth
            can_futility_prune = static_eval + futility_margin < alpha

        for move_index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            is_promotion = move.promotion is not None

            # Futility pruning: skip quiet moves that can't raise alpha
            # Conditions: low depth, not PV move, quiet move, not in check
            if (
                can_futility_prune
                and move_index > 0  # Don't prune first move
                and not is_capture
                and not gives_check
                and not is_promotion
            ):
                continue

            # make the move
            board.push(move)

            # Late Move Reductions (LMR):
            # Reduce search depth for late quiet moves that are unlikely to be good
            # Conditions: sufficient depth, late move, quiet (no capture/check/promotion)
            reduction = 0
            if (
                depth >= 3
                and move_index >= 3
                and not is_capture
                and not gives_check
                and not is_promotion
                and not in_check
            ):
                # Simple reduction: reduce by 1 ply
                reduction = 1

            # Principal Variation Search (PVS):
            # For the first move, search with full window
            # For subsequent moves, search with zero window first
            if move_index == 0:
                # First move: full window search
                board_score = -self.negamax(
                    board,
                    depth - 1,
                    null_move,
                    cache,
                    -beta,
                    -alpha,
                    ply + 1,
                    killers,
                )[0]
            else:
                # Later moves: zero window search (with LMR reduction if applicable)
                board_score = -self.negamax(
                    board,
                    depth - 1 - reduction,
                    null_move,
                    cache,
                    -alpha - 1,  # Zero window
                    -alpha,
                    ply + 1,
                    killers,
                )[0]

                # If zero window search found a promising move, re-search with full window
                if board_score > alpha and (board_score < beta or reduction > 0):
                    board_score = -self.negamax(
                        board,
                        depth - 1,  # Full depth (no reduction)
                        null_move,
                        cache,
                        -beta,
                        -alpha,
                        ply + 1,
                        killers,
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
                # Update killer moves for quiet moves that cause beta cutoff
                # Add to killers if not already there (keep 2 killers per ply)
                if (
                    killers
                    and not is_capture
                    and ply < len(killers)
                    and move not in killers[ply]
                ):
                    killers[ply].insert(0, move)
                    if len(killers[ply]) > 2:
                        killers[ply].pop()

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
        """
        Search for the best move using iterative deepening with aspiration windows.

        Iterative deepening searches depth 1, then 2, then 3, etc.
        This improves move ordering (TT move from depth N-1 is tried first at depth N)
        and allows for future time management (can stop early if time runs out).

        Aspiration windows: after depth 1, use a narrow window around the previous
        score. If the search fails outside the window, re-search with a wider window.
        """
        self.nodes = 0
        # Create shared cache - persists across all depths
        cache: CACHE_TYPE = {}
        best_move = None
        target_depth = self.config.negamax_depth
        prev_score = None

        # Killer moves table: 2 killers per ply, persists across iterations
        # Max ply is roughly target_depth + quiescence_depth + some buffer
        max_ply = target_depth + self.config.quiescence_search_depth + 10
        killers: list = [[] for _ in range(max_ply)]

        # Aspiration window parameters
        INITIAL_WINDOW = 50  # Initial window size (centipawns)

        # Iterative deepening: search depth 1, 2, 3, ... up to target
        for depth in range(1, target_depth + 1):
            # Use aspiration windows after first iteration
            if prev_score is None or depth <= 1:
                # First iteration: full window
                alpha = NEG_INF
                beta = INF
            else:
                # Subsequent iterations: narrow window around previous score
                alpha = prev_score - INITIAL_WINDOW
                beta = prev_score + INITIAL_WINDOW

            # Aspiration window loop: widen window on fail high/low
            window = INITIAL_WINDOW
            while True:
                score, move = self.negamax(
                    board,
                    depth,
                    self.config.null_move,
                    cache,
                    alpha=alpha,
                    beta=beta,
                    ply=0,
                    killers=killers,
                )

                # Check if we need to re-search with wider window
                if score <= alpha:
                    # Failed low: widen window on the low side
                    window *= 2
                    # prev_score is guaranteed non-None after depth 1
                    assert prev_score is not None
                    alpha = prev_score - window
                    if window > 500:  # Give up and use full window
                        alpha = NEG_INF
                elif score >= beta:
                    # Failed high: widen window on the high side
                    window *= 2
                    # prev_score is guaranteed non-None after depth 1
                    assert prev_score is not None
                    beta = prev_score + window
                    if window > 500:  # Give up and use full window
                        beta = INF
                else:
                    # Score is within window, we're done
                    break

                # Safety: if window is fully open, we must accept the result
                if alpha == NEG_INF and beta == INF:
                    break

            prev_score = score

            # Update best move from completed search
            if move is not None:
                best_move = move

        assert best_move is not None, "Best move from root should not be None"
        return best_move
