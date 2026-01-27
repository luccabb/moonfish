from copy import copy
from multiprocessing import cpu_count, Manager, Pool

import chess.polyglot
from chess import Board, Move
from moonfish.engines.alpha_beta import AlphaBeta


class LazySMP(AlphaBeta):

    def search_move(self, board: Board) -> Move:
        # start multiprocessing
        nprocs = cpu_count()
        with Pool(processes=nprocs) as pool, Manager() as manager:
            shared_cache = manager.dict()
            # executing negamax in parallel N times
            # all processes share the cache for faster convergence
            # starmap blocks until all processes are done
            pool.starmap(
                self.negamax,
                [
                    (
                        board,
                        copy(self.config.negamax_depth),
                        self.config.null_move,
                        shared_cache,
                    )
                    for _ in range(nprocs)
                ],
            )

            # return best move for our original board
            # cache key is now just the zobrist hash
            cache_key = chess.polyglot.zobrist_hash(board)
            return shared_cache[cache_key][1]
