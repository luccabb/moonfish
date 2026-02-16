from multiprocessing import cpu_count, Manager, Pool

from chess import Board, Move
from moonfish.engines.alpha_beta import AlphaBeta


class LazySMP(AlphaBeta):

    def search_move(self, board: Board) -> Move:
        # start multiprocessing
        nprocs = cpu_count()
        pool = Pool(processes=nprocs)
        manager = Manager()
        shared_cache = manager.dict()
        # executing all the moves at layer 1 in parallel
        # starmap blocks until all process are done
        pool.starmap(
            self.negamax,
            [
                (
                    board,
                    self.config.negamax_depth,
                    self.config.null_move,
                    shared_cache,
                )
                for _ in range(nprocs)
            ],
        )

        # return best move from the transposition table
        tt_key = board._transposition_key()
        return shared_cache[tt_key][1]
