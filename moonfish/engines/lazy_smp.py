from multiprocessing import Manager, Pool, cpu_count

from chess import Board, Move

from moonfish.engines.alpha_beta import INF, NEG_INF, AlphaBeta


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

        # return best move for our original board
        return shared_cache[
            (
                board._transposition_key(),
                self.config.negamax_depth,
                self.config.null_move,
                NEG_INF,
                INF,
            )
        ][1]
