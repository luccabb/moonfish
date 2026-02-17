"""
NNUE chess engine with accumulator state management.

Uses alpha-beta search with an NNUE evaluator. Maintains accumulator state
during search by saving/restoring before push/pop, enabling incremental
updates instead of full recomputation at every node.
"""

from multiprocessing.managers import DictProxy

import chess
from chess import Board, Move

from moonfish.config import Config
from moonfish.engines.alpha_beta import AlphaBeta, CACHE_KEY, INF, NEG_INF, NULL_MOVE
from moonfish.evaluation.classical import ClassicalEvaluator
from moonfish.evaluation.nn import NNUEEvaluator
from moonfish.move_ordering import organize_moves, organize_moves_quiescence
from moonfish.psqt import count_pieces


class NNEngine(AlphaBeta):
    """
    Chess engine using NNUE evaluation with incremental accumulator updates.

    Overrides the search methods from AlphaBeta to save/restore NNUE
    accumulator state around board.push/pop, enabling efficient incremental
    updates during the search tree traversal.
    """

    def __init__(
        self,
        config: Config,
        evaluator: NNUEEvaluator | ClassicalEvaluator | None = None,
    ):
        super().__init__(config)
        self.evaluator = evaluator or ClassicalEvaluator()
        self._use_nnue = isinstance(self.evaluator, NNUEEvaluator)

    def eval_board(self, board: Board) -> float:
        """Evaluate board using the NNUE evaluator (or classical fallback)."""
        if self.tablebase is not None:
            pieces = sum(count_pieces(board))
            if pieces <= self.config.syzygy_pieces:
                try:
                    dtz = self.tablebase.probe_dtz(board)
                    return dtz
                except (chess.syzygy.MissingTableError, KeyError):
                    pass

        return self.evaluator.evaluate(board)

    def _push_move(self, board: Board, move: Move) -> tuple | None:
        """Save accumulator state, apply incremental update, push move."""
        saved = None
        if self._use_nnue:
            nnue = self.evaluator
            saved = nnue.save_accumulators()
            if move != NULL_MOVE:
                nnue.update_move(board, move)
        board.push(move)
        return saved

    def _pop_move(self, board: Board, saved: tuple | None) -> None:
        """Pop move and restore accumulator state."""
        board.pop()
        if saved is not None and self._use_nnue:
            self.evaluator.restore_accumulators(saved)

    def quiescence_search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Quiescence search with accumulator save/restore."""
        import time

        in_check = board.is_check()
        self.nodes += 1

        if self._stop_time and (self.nodes & 511) == 0:
            if time.perf_counter() >= self._stop_time:
                self._time_abort = True
        if self._time_abort:
            return self.eval_board(board)

        if board.is_checkmate():
            return -self.config.checkmate_score
        if board.is_stalemate():
            return 0
        if board.is_fifty_moves() or board.is_insufficient_material():
            return 0

        stand_pat = self.eval_board(board)

        if in_check:
            if depth <= 0:
                return stand_pat
            best_score = NEG_INF
            moves = list(board.legal_moves)
        else:
            if depth <= 0:
                return stand_pat
            if stand_pat >= beta:
                return beta
            best_score = stand_pat
            alpha = max(alpha, stand_pat)
            moves = organize_moves_quiescence(board)

        for move in moves:
            saved = self._push_move(board, move)

            if board.is_repetition(2):
                score = 0.0
            else:
                score = -self.quiescence_search(board, depth - 1, -beta, -alpha)

            self._pop_move(board, saved)

            if score > best_score:
                best_score = score
            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return best_score

    def negamax(
        self,
        board: Board,
        depth: int,
        null_move: bool,
        cache: DictProxy | CACHE_KEY,
        alpha: float = NEG_INF,
        beta: float = INF,
    ) -> tuple[float | int, Move | None]:
        """Negamax search with accumulator save/restore."""
        import time

        cache_key = (board._transposition_key(), depth, null_move, alpha, beta)
        self.nodes += 1

        if self._stop_time and (self.nodes & 511) == 0:
            if time.perf_counter() >= self._stop_time:
                self._time_abort = True
        if self._time_abort:
            return 0, None

        if cache_key in cache:
            return cache[cache_key]

        if board.is_checkmate():
            cache[cache_key] = (-self.config.checkmate_score, None)
            return (-self.config.checkmate_score, None)
        if board.is_stalemate():
            cache[cache_key] = (0, None)
            return (0, None)

        if depth <= 0:
            board_score = self.quiescence_search(
                board, self.config.quiescence_search_depth, alpha, beta
            )
            cache[cache_key] = (board_score, None)
            return board_score, None

        # Null move pruning
        if (
            self.config.null_move
            and depth >= (self.config.null_move_r + 1)
            and not board.is_check()
        ):
            board_score = self.eval_board(board)
            if board_score >= beta:
                saved = self._push_move(board, NULL_MOVE)
                board_score = -self.negamax(
                    board,
                    depth - 1 - self.config.null_move_r,
                    False,
                    cache,
                    -beta,
                    -beta + 1,
                )[0]
                self._pop_move(board, saved)
                if board_score >= beta:
                    cache[cache_key] = (beta, None)
                    return beta, None

        best_move = None
        best_score = NEG_INF
        moves = organize_moves(board)

        for move in moves:
            saved = self._push_move(board, move)

            board_score = -self.negamax(
                board, depth - 1, null_move, cache, -beta, -alpha
            )[0]
            if board_score > self.config.checkmate_threshold:
                board_score -= 1
            if board_score < -self.config.checkmate_threshold:
                board_score += 1

            self._pop_move(board, saved)

            if board_score >= beta:
                cache[cache_key] = (board_score, move)
                return board_score, move

            if board_score > best_score:
                best_score = board_score
                best_move = move

            alpha = max(alpha, board_score)
            if alpha >= beta:
                break

        if not best_move:
            best_move = self.random_move(board)

        cache[cache_key] = (best_score, best_move)
        return best_score, best_move

    def search_move(self, board: Board) -> Move:
        """Search for the best move, initializing NNUE accumulators first."""
        self.nodes = 0
        self._time_abort = False
        self._stop_time = 0.0

        if self._use_nnue:
            self.evaluator.reset(board)
        elif hasattr(self.evaluator, "reset"):
            self.evaluator.reset()

        cache: CACHE_KEY = {}
        best_move = self.negamax(
            board, self.config.negamax_depth, self.config.null_move, cache
        )[1]
        assert best_move is not None, "Best move from root should not be None"
        return best_move

    def search_move_timed(self, board: Board, time_limit_s: float) -> Move:
        """Search with iterative deepening + time limit, maintaining accumulators."""
        import time

        self.nodes = 0
        self._time_abort = False
        self._stop_time = time.perf_counter() + time_limit_s

        if self._use_nnue:
            self.evaluator.reset(board)
        elif hasattr(self.evaluator, "reset"):
            self.evaluator.reset()

        cache: CACHE_KEY = {}
        best_move = None
        MAX_DEPTH = 100

        for depth in range(1, MAX_DEPTH + 1):
            score, move = self.negamax(
                board, depth, self.config.null_move, cache
            )

            if self._time_abort:
                break

            if move is not None:
                best_move = move

            elapsed = time.perf_counter() - (self._stop_time - time_limit_s)
            remaining = self._stop_time - time.perf_counter()
            if remaining < elapsed * 3:
                break

        self._stop_time = 0.0
        self._time_abort = False

        if best_move is None:
            best_move = self.random_move(board)
        return best_move
