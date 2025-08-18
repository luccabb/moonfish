import unittest

from bulletchess import Board, Move

from moonfish.config import Config
from moonfish.helper import get_engine
from parameterized import parameterized

test_boards = [
    (Board.from_fen("4r2K/8/8/8/8/7r/8/3k4 w - - 0 1"), 1, [Move.from_uci("h8g7")]),
    (Board.from_fen("8/8/8/6K1/8/5r2/8/2k3r1 w - - 0 1"), 4, [Move.from_uci("g5h4")]),
    (
        Board.from_fen("7K/8/8/8/6Q1/8/4N3/7k w - - 0 1"),
        1,
        [Move.from_uci("g4h3"), Move.from_uci("g4g1")],
    ),
    (Board.from_fen("7K/8/8/8/3N4/7Q/7p/7k w - - 0 1"), 2, [Move.from_uci("h3f1")]),
    (Board.from_fen("7K/8/8/8/8/1R6/5R2/3k4 w - - 0 1"), 2, [Move.from_uci("b3b1")]),
    (
        Board.from_fen("7K/8/8/8/8/1R6/R7/2k5 w - - 0 1"),
        3,
        [
            Move.from_uci("h8h7"),
            Move.from_uci("a2h2"),
            Move.from_uci("b3b4"),
            Move.from_uci("b3b5"),
            Move.from_uci("b3b6"),
            Move.from_uci("b3b7"),
            Move.from_uci("b3b4"),
            Move.from_uci("a2g2"),
            Move.from_uci("a2f2"),
            Move.from_uci("b3b8"),
            Move.from_uci("h8g8"),
            Move.from_uci("h8g7"),
            Move.from_uci("b3b2"),
        ],
    ),
    (
        Board.from_fen("rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQk - 0 1"),
        2,
        [Move.from_uci("d1h5")],
    ),
    (
        Board.from_fen("r3r3/3Q2P1/k2p4/B2P4/8/P2N4/1PP2K1P/2R5 w - - 0 1"),
        3,
        [Move.from_uci("d7c6")],
    ),
    (Board.from_fen("4R2k/8/8/8/8/7R/8/3K4 b - - 0 1"), 1, [Move.from_uci("h8g7")]),
    (Board.from_fen("8/8/8/6k1/8/5R2/8/2K3R1 b - - 0 1"), 4, [Move.from_uci("g5h4")]),
    (
        Board.from_fen("7k/8/8/8/6q1/8/4n3/7K b - - 0 1"),
        1,
        [Move.from_uci("g4h3"), Move.from_uci("g4g1")],
    ),
    (Board.from_fen("7k/8/8/8/3n4/7q/7P/7K b - - 0 1"), 2, [Move.from_uci("h3f1")]),
    (Board.from_fen("7k/8/8/8/8/1r6/5r2/3K4 b - - 0 1"), 2, [Move.from_uci("b3b1")]),
    (
        Board.from_fen("7k/8/8/8/8/1r6/r7/2K5 b - - 0 1"),
        3,
        [
            Move.from_uci("h8h7"),
            Move.from_uci("a2h2"),
            Move.from_uci("b3b4"),
            Move.from_uci("b3b5"),
            Move.from_uci("b3b6"),
            Move.from_uci("b3b7"),
            Move.from_uci("b3b4"),
            Move.from_uci("a2g2"),
            Move.from_uci("a2f2"),
            Move.from_uci("b3b8"),
            Move.from_uci("h8g8"),
            Move.from_uci("h8g7"),
            Move.from_uci("b3b2"),
        ],
    ),
    (
        Board.from_fen("rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 1"),
        2,
        [Move.from_uci("d8h4")],
    ),
    (
        Board.from_fen("2r5/1pp2k1p/p2n4/8/b2p4/K2P4/3q2p1/R3R3 b - - 0 1"),
        3,
        [Move.from_uci("d2c3")],
    ),
]


class Testunctions(unittest.TestCase):

    @parameterized.expand(test_boards)
    def test_alpha_beta(self, board, depth, expected_result):
        config = Config(
            mode="uci",
            algorithm="alpha_beta",
            negamax_depth=depth,
            null_move=False,
            null_move_r=2,
            quiescence_search_depth=3,
            syzygy_path=None,
            syzygy_pieces=5,
        )

        engine = get_engine(config)
        result = engine.search_move(board)
        self.assertIn(result, expected_result)

    @parameterized.expand(test_boards)
    def test_parallel_alpha_beta_layer_1(self, board, depth, expected_result):
        config = Config(
            mode="uci",
            algorithm="parallel_alpha_beta_layer_1",
            negamax_depth=depth,
            null_move=False,
            null_move_r=2,
            quiescence_search_depth=3,
            syzygy_path=None,
            syzygy_pieces=5,
        )
        engine = get_engine(config)
        result = engine.search_move(board)
        self.assertIn(result, expected_result)

    @parameterized.expand(test_boards)
    def test_parallel_alpha_beta_layer_2(self, board, depth, expected_result):
        config = Config(
            mode="uci",
            algorithm="parallel_alpha_beta_layer_2",
            negamax_depth=depth,
            null_move=True,
            null_move_r=2,
            quiescence_search_depth=3,
            syzygy_path=None,
            syzygy_pieces=5,
        )
        engine = get_engine(config)
        result = engine.search_move(board)
        self.assertIn(result, expected_result)

    @parameterized.expand(test_boards)
    def test_lazy_smp(self, board, depth, expected_result):
        config = Config(
            mode="uci",
            algorithm="lazy_smp",
            negamax_depth=depth,
            null_move=False,
            null_move_r=2,
            quiescence_search_depth=3,
            syzygy_path=None,
            syzygy_pieces=5,
        )

        engine = get_engine(config)
        result = engine.search_move(board)
        self.assertIn(result, expected_result)


if __name__ == "__main__":
    unittest.main()
