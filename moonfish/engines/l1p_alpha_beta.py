from multiprocessing import cpu_count, Manager, Pool

from chess import Board, Move
from moonfish.engines.alpha_beta import AlphaBeta


class Layer1ParallelAlphaBeta(AlphaBeta):
    """
    This class implements a parallel search
    algorithm starting from the first layer.
    """

    def search_move(self, board: Board) -> Move:
        self.nodes = 0
        # start multiprocessing
        nprocs = cpu_count()

        with Pool(processes=nprocs) as pool, Manager() as manager:
            shared_cache = manager.dict()

            # creating list of moves at layer 1
            moves = list(board.legal_moves)
            arguments = []
            for move in moves:
                board.push(move)
                arguments.append(
                    (
                        board.copy(),
                        self.config.negamax_depth - 1,
                        self.config.null_move,
                        shared_cache,
                    )
                )
                board.pop()

            # executing all the moves at layer 1 in parallel
            # starmap blocks until all processes are done
            processes = pool.starmap(self.negamax, arguments)
            results = []

            # inserting move information in the results
            # negamax returns (score, best_move) - we negate score since
            # it's from opponent's perspective
            for i in range(len(processes)):
                score = -processes[i][0]  # Negate: opponent's -> our perspective
                results.append((score, processes[i][1], moves[i]))

            # sorting results by score (descending) and getting best move
            results.sort(key=lambda a: a[0], reverse=True)
            best_move = results[0][2]
            return best_move
