from collections import defaultdict
from copy import copy
from functools import partial
from multiprocessing import Manager, Pool, cpu_count
from multiprocessing.managers import DictProxy
from typing import List, Tuple

from chess import Board

from moonfish.config import Config
from moonfish.engines.alpha_beta import AlphaBeta


def LAYER_SIGNAL_CORRECTION(data):
    return data if data[3] == 2 else (-data[0], *data[1:])


def CHECKMATE_CORRECTION(data, threshold):
    return (data[0] + 1, *data[1:]) if (data[0] > threshold and data[3] == 1) else data


class Layer2ParallelAlphaBeta(AlphaBeta):
    """
    This class implements a parallel search
    algorithm starting from the second layer.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.checkmate_correction = partial(
            CHECKMATE_CORRECTION, threshold=self.config.checkmate_threshold
        )

    def generate_board_and_moves(
        self, og_board: Board, board_to_move_that_generates_it: DictProxy, layer: int
    ) -> List[Tuple[Board, Board, int]]:
        """
        Generate all possible boards with their layer depth for each board.

        Args:
            og_board: Board to generate moves for.
            board_to_move_that_generates_it: Dictionary that maps board to move that generates it.
            layer: Layer depth of the board.

        Returns:
            List of tuples with:
                - generated board (original board with move applied)
                - original board
                - layer depth
        """
        boards_and_moves = []
        board = copy(og_board)

        # if board has no legal moves, we leave it as is
        # we need to run this board through negamax to get its value
        if not og_board.legal_moves:
            boards_and_moves.append((og_board, og_board, layer))

        # get first layer move that generates current board
        first_move = board_to_move_that_generates_it.get(og_board.fen())

        # generating all possible moves
        for move in board.legal_moves:

            board.push(move)

            # save first layer move that generates current board
            if first_move:
                board_to_move_that_generates_it[board.fen()] = first_move
            else:
                board_to_move_that_generates_it[board.fen()] = move
            # add new board, original board, and current layer to our output
            boards_and_moves.append((copy(board), og_board, layer + 1))

            board.pop()

        return boards_and_moves

    def search_move(self, board: Board) -> str:
        START_LAYER = 2
        # start multiprocessing
        nprocs = cpu_count()
        pool = Pool(processes=nprocs)
        manager = Manager()
        shared_cache = manager.dict()

        # pointer that help us in finding the best next move
        board_to_move_that_generates_it = manager.dict()

        # starting board list
        board_list = [(board, board, 0)]

        # generating all possible boards for up to 2 moves ahead
        for _ in range(START_LAYER):
            arguments = [
                (board, board_to_move_that_generates_it, layer)
                for board, _, layer in board_list
            ]
            processes = pool.starmap(self.generate_board_and_moves, arguments)
            board_list = [board for board in sum(processes, [])]

        negamax_arguments = [
            (
                board,
                copy(self.config.negamax_depth) - START_LAYER,
                self.config.null_move,
                shared_cache,
            )
            for board, _, _ in board_list
        ]

        parallel_layer_result = pool.starmap(self.negamax, negamax_arguments)

        # grouping output based on the  board that generates it
        groups = defaultdict(list)

        # adding information about the board and layer
        # that generates the results and separating them
        # into groups based on the root board
        for i in range(len(parallel_layer_result)):
            groups[board_list[i][1].fen()].append(
                (*parallel_layer_result[i], board_list[i][0], board_list[i][2])
            )

        best_boards = []

        for group in groups.values():
            # layer and checkmate corrections
            # they are needed to adjust for
            # boards from different layers
            group = list(map(LAYER_SIGNAL_CORRECTION, group))
            group = list(map(self.checkmate_correction, group))
            # get best move from group
            group.sort(key=lambda a: a[0])
            best_boards.append(group[0])

        # get best board
        best_boards.sort(key=lambda a: a[0], reverse=True)
        best_board = best_boards[0][2].fen()

        # get move that results in best board
        best_move = board_to_move_that_generates_it[best_board]

        return best_move
