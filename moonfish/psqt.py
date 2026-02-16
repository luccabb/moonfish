# flake8: noqa

import chess
import chess.syzygy

############
# I'm using Pesto Evaluation function:
# https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
# values for Piece-Square Tables from Rofchade:
# http://www.talkchess.com/forum3/viewtopic.php?f=2&t=68311&start=19
############
# Piece values indexed by piece type (0=unused, 1=PAWN, 2=KNIGHT, ..., 6=KING)
MG_PIECE_VALUES = (0, 82, 337, 365, 477, 1025, 24000)

EG_PIECE_VALUES = (0, 94, 281, 297, 512, 936, 24000)

# fmt: off
MG_PAWN = [
    0,   0,   0,   0,   0,   0,  0,   0,
    98, 134,  61,  95,  68, 126, 34, -11,
    -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0]

EG_PAWN = [
    0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0]

MG_KNIGHT = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23]

EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64]

MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21]

EG_BISHOP = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17]

MG_ROOK = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26]

EG_ROOK = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20]

MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50]

EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41]

MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14]

EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43]
# fmt: on

# Indexed by piece type (0=unused, 1=PAWN, 2=KNIGHT, ..., 6=KING)
MG_PESTO: tuple[list[int], ...] = (
    [0],
    MG_PAWN,
    MG_KNIGHT,
    MG_BISHOP,
    MG_ROOK,
    MG_QUEEN,
    MG_KING,
)

EG_PESTO: tuple[list[int], ...] = (
    [0],
    EG_PAWN,
    EG_KNIGHT,
    EG_BISHOP,
    EG_ROOK,
    EG_QUEEN,
    EG_KING,
)

############
# Evaluation bonuses/penalties beyond PST
############

# Bishop pair bonus (having both bishops)
MG_BISHOP_PAIR_BONUS = 30
EG_BISHOP_PAIR_BONUS = 50

# Rook on open file (no pawns on file)
MG_ROOK_OPEN_FILE = 20
EG_ROOK_OPEN_FILE = 10

# Rook on semi-open file (no friendly pawns on file)
MG_ROOK_SEMI_OPEN_FILE = 10
EG_ROOK_SEMI_OPEN_FILE = 5

# Isolated pawn penalty (no friendly pawns on adjacent files)
MG_ISOLATED_PAWN = -10
EG_ISOLATED_PAWN = -20

# Doubled pawn penalty (multiple friendly pawns on the same file)
MG_DOUBLED_PAWN = -10
EG_DOUBLED_PAWN = -15

# Passed pawn bonus by rank (rank 1 to 8, index 0 unused)
# Bonus increases as pawn advances; huge in endgame
# Index = rank for white (rank 2..7 are the meaningful ones; 1 and 8 don't exist for pawns)
MG_PASSED_PAWN_BONUS = (0, 0, 5, 10, 20, 40, 60, 0)
EG_PASSED_PAWN_BONUS = (0, 0, 10, 20, 40, 70, 120, 0)

# File masks: for each file (0-7), a set of square indices on that file
FILE_SQUARES: tuple[set[int], ...] = tuple(
    {file + rank * 8 for rank in range(8)} for file in range(8)
)

# Adjacent files for each file (0-7)
ADJACENT_FILES: tuple[tuple[int, ...], ...] = tuple(
    tuple(f for f in (file - 1, file + 1) if 0 <= f <= 7) for file in range(8)
)

############
# Tapered Evaluation: https://www.chessprogramming.org/Tapered_Eval
# Phase values are used to determine on what phase of the game
# we're currently at.
############
PAWN_PHASE = 0
KNIGHT_PHASE = 1
BISHOP_PHASE = 1
ROOK_PHASE = 2
QUEEN_PHASE = 4
TOTAL_PHASE = (
    PAWN_PHASE * 16
    + KNIGHT_PHASE * 4
    + BISHOP_PHASE * 4
    + ROOK_PHASE * 4
    + QUEEN_PHASE * 2
)

PHASE_VALUES = [
    PAWN_PHASE,
    PAWN_PHASE,
    KNIGHT_PHASE,
    KNIGHT_PHASE,
    BISHOP_PHASE,
    BISHOP_PHASE,
    ROOK_PHASE,
    ROOK_PHASE,
    QUEEN_PHASE,
    QUEEN_PHASE,
]


def count_pieces(board: chess.Board) -> list[int]:
    """
    Counts the number of each piece on the board.

    :param
        board: The board to count the pieces on.
    :return:
        A list of tuples containing the number of pieces of that type
        and their phase value.
    """

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    return [
        wp,
        bp,
        wn,
        bn,
        wb,
        bb,
        wr,
        br,
        wq,
        bq,
    ]


def get_phase(board: chess.Board) -> float:
    """
    Calculates the phase of the game based on the number of pieces
    on the board.

    :param
        pieces: A list of tuples containing the number of pieces of
        that type and their phase value.
    :return:
        The phase of the game.
    """
    pieces = count_pieces(board)
    phase: float = TOTAL_PHASE

    for piece_count, piece_phase in zip(pieces, PHASE_VALUES):
        phase -= piece_count * piece_phase

    phase = (phase * 256 + (TOTAL_PHASE / 2)) / TOTAL_PHASE
    return phase


BOARD_EVALUATION_CACHE = {}


def board_evaluation_cache(fun):

    def inner(board: chess.Board):
        key = board._transposition_key()
        if key not in BOARD_EVALUATION_CACHE:
            BOARD_EVALUATION_CACHE[key] = fun(board)
        return BOARD_EVALUATION_CACHE[key]

    return inner


@board_evaluation_cache
def board_evaluation(board: chess.Board) -> float:
    """
    Evaluates the board using PeSTO PST values plus structural bonuses:
    - Pawn structure (passed, isolated, doubled pawns)
    - Bishop pair bonus
    - Rook on open/semi-open file

    Arguments:
        - board: current board state.

    Returns:
        - total_value(int): integer representing current value for this board.
    """

    phase = get_phase(board)

    mg_white = 0
    mg_black = 0
    eg_white = 0
    eg_black = 0

    # Track piece locations for structural evaluation
    white_pawns: list[int] = []
    black_pawns: list[int] = []
    white_rooks: list[int] = []
    black_rooks: list[int] = []
    white_bishop_count = 0
    black_bishop_count = 0

    # iterate only occupied squares via piece_map()
    for square, piece in board.piece_map().items():
        pt = piece.piece_type
        if piece.color == chess.WHITE:
            mg_white += MG_PESTO[pt][square ^ 56] + MG_PIECE_VALUES[pt]
            eg_white += EG_PESTO[pt][square ^ 56] + EG_PIECE_VALUES[pt]
            if pt == chess.PAWN:
                white_pawns.append(square)
            elif pt == chess.ROOK:
                white_rooks.append(square)
            elif pt == chess.BISHOP:
                white_bishop_count += 1
        else:
            mg_black += MG_PESTO[pt][square] + MG_PIECE_VALUES[pt]
            eg_black += EG_PESTO[pt][square] + EG_PIECE_VALUES[pt]
            if pt == chess.PAWN:
                black_pawns.append(square)
            elif pt == chess.ROOK:
                black_rooks.append(square)
            elif pt == chess.BISHOP:
                black_bishop_count += 1

    # Pawn file sets for structural analysis
    white_pawn_files = set()
    black_pawn_files = set()
    white_pawns_per_file: dict[int, int] = {}
    black_pawns_per_file: dict[int, int] = {}

    for sq in white_pawns:
        f = sq % 8
        white_pawn_files.add(f)
        white_pawns_per_file[f] = white_pawns_per_file.get(f, 0) + 1

    for sq in black_pawns:
        f = sq % 8
        black_pawn_files.add(f)
        black_pawns_per_file[f] = black_pawns_per_file.get(f, 0) + 1

    # --- Pawn structure ---
    # White pawns
    for sq in white_pawns:
        f = sq % 8
        r = sq // 8 + 1  # rank 1-8 (rank 1 = row 0)

        # Doubled pawn: more than one pawn on same file
        if white_pawns_per_file[f] > 1:
            mg_white += MG_DOUBLED_PAWN
            eg_white += EG_DOUBLED_PAWN

        # Isolated pawn: no friendly pawns on adjacent files
        has_adjacent = any(af in white_pawn_files for af in ADJACENT_FILES[f])
        if not has_adjacent:
            mg_white += MG_ISOLATED_PAWN
            eg_white += EG_ISOLATED_PAWN

        # Passed pawn: no enemy pawns on same or adjacent files that can block/capture
        is_passed = True
        check_files = (f,) + ADJACENT_FILES[f]
        for cf in check_files:
            for bsq in black_pawns:
                bf = bsq % 8
                br = bsq // 8 + 1
                if bf == cf and br > r:  # black pawn ahead of white pawn
                    is_passed = False
                    break
            if not is_passed:
                break
        if is_passed and 2 <= r <= 7:
            mg_white += MG_PASSED_PAWN_BONUS[r]
            eg_white += EG_PASSED_PAWN_BONUS[r]

    # Black pawns
    for sq in black_pawns:
        f = sq % 8
        r = sq // 8 + 1  # rank 1-8

        # Doubled pawn
        if black_pawns_per_file[f] > 1:
            mg_black += MG_DOUBLED_PAWN
            eg_black += EG_DOUBLED_PAWN

        # Isolated pawn
        has_adjacent = any(af in black_pawn_files for af in ADJACENT_FILES[f])
        if not has_adjacent:
            mg_black += MG_ISOLATED_PAWN
            eg_black += EG_ISOLATED_PAWN

        # Passed pawn (for black, no white pawns ahead = lower rank number)
        is_passed = True
        check_files = (f,) + ADJACENT_FILES[f]
        for cf in check_files:
            for wsq in white_pawns:
                wf = wsq % 8
                wr = wsq // 8 + 1
                if wf == cf and wr < r:  # white pawn ahead of black pawn
                    is_passed = False
                    break
            if not is_passed:
                break
        if is_passed and 2 <= r <= 7:
            # For black, rank 7 is closest to promotion (like white rank 2)
            # Mirror the rank: black rank 7 -> index 2, rank 2 -> index 7
            bonus_rank = 9 - r
            mg_black += MG_PASSED_PAWN_BONUS[bonus_rank]
            eg_black += EG_PASSED_PAWN_BONUS[bonus_rank]

    # --- Bishop pair bonus ---
    if white_bishop_count >= 2:
        mg_white += MG_BISHOP_PAIR_BONUS
        eg_white += EG_BISHOP_PAIR_BONUS
    if black_bishop_count >= 2:
        mg_black += MG_BISHOP_PAIR_BONUS
        eg_black += EG_BISHOP_PAIR_BONUS

    # --- Rook on open/semi-open file ---
    for sq in white_rooks:
        f = sq % 8
        if f not in white_pawn_files:
            if f not in black_pawn_files:
                # Open file: no pawns at all
                mg_white += MG_ROOK_OPEN_FILE
                eg_white += EG_ROOK_OPEN_FILE
            else:
                # Semi-open file: no friendly pawns
                mg_white += MG_ROOK_SEMI_OPEN_FILE
                eg_white += EG_ROOK_SEMI_OPEN_FILE

    for sq in black_rooks:
        f = sq % 8
        if f not in black_pawn_files:
            if f not in white_pawn_files:
                mg_black += MG_ROOK_OPEN_FILE
                eg_black += EG_ROOK_OPEN_FILE
            else:
                mg_black += MG_ROOK_SEMI_OPEN_FILE
                eg_black += EG_ROOK_SEMI_OPEN_FILE

    # calculate board score based on phase
    if board.turn == chess.WHITE:
        mg_score = mg_white - mg_black
        eg_score = eg_white - eg_black
    else:
        mg_score = mg_black - mg_white
        eg_score = eg_black - eg_white
    eval = ((mg_score * (256 - phase)) + (eg_score * phase)) / 256

    return eval


def evaluate_piece(board: chess.Board, square: chess.Square, phase: float) -> float:
    """
    Evaluates a piece on a given square.

    Arguments:
        - board: current board state.
        - square: square to evaluate.
        - phase: current phase of the game.

    Returns:
        - value(float): float representing
        current value for this piece on this square.
    """
    mg_score = 0
    eg_score = 0

    # get mid and end game score for single piece
    piece = board.piece_at(square)
    if piece is not None:
        if piece.color == chess.WHITE:
            mg_score += (
                MG_PESTO[piece.piece_type][square ^ 56]
                + MG_PIECE_VALUES[piece.piece_type]
            )
            eg_score += (
                EG_PESTO[piece.piece_type][square ^ 56]
                + EG_PIECE_VALUES[piece.piece_type]
            )
        if piece.color == chess.BLACK:
            mg_score += (
                MG_PESTO[piece.piece_type][square] + MG_PIECE_VALUES[piece.piece_type]
            )
            eg_score += (
                EG_PESTO[piece.piece_type][square] + EG_PIECE_VALUES[piece.piece_type]
            )

    # evaluate piece value based on phase
    eval = ((mg_score * (256 - phase)) + (eg_score * phase)) / 256
    return eval


def evaluate_capture(board: chess.Board, move: chess.Move, phase: float) -> float:
    """
    Evaluates a capture move based phase of the game.

    Arguments:
        - board: current board state.
        - move: move to evaluate.
        - phase: current phase of the game.

    Returns:
        - value(float): float representing
        value for this capture.
    """
    mg_score = 0
    eg_score = 0

    # en passant score
    if board.is_en_passant(move):
        # En passant capture evaluation
        capturing_piece = chess.PAWN
        captured_piece = chess.PAWN
    else:
        capturing_piece = board.piece_at(move.from_square).piece_type  # type: ignore
        captured_piece = board.piece_at(move.to_square).piece_type  # type: ignore

    # get mid and end game difference of scores between captured
    # and capturing piece
    if capturing_piece is not None and captured_piece is not None:
        mg_score += MG_PIECE_VALUES[captured_piece] - MG_PIECE_VALUES[capturing_piece]
        eg_score += EG_PIECE_VALUES[captured_piece] - EG_PIECE_VALUES[capturing_piece]

    # evaluate capture based on game's phase
    eval = ((mg_score * (256 - phase)) + (eg_score * phase)) / 256
    return eval
