#!/usr/bin/env python3
"""
Generate NNUE training data using Stockfish evaluations.

Produces labeled chess positions for training the NNUE evaluation network.
Each position is labeled with Stockfish's evaluation at the specified depth
and its WDL (Win/Draw/Loss) probabilities.

Requires: python-chess, numpy, and Stockfish binary in PATH or via --stockfish.

Usage:
    uv run python scripts/generate_training_data.py \
        --num-positions 1000000 \
        --depth 8 \
        --output data/training_data.npz \
        --workers 4
"""

import argparse
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np


# 48 bench positions from Stockfish (same as moonfish bench)
SEED_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
    "8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1",
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",
    "8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1",
    "8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124",
    "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
    "r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
]


def feature_index(piece_type: int, color: bool, square: int) -> int:
    """Compute NNUE feature index for a piece (white perspective)."""
    return (piece_type - 1) * 128 + int(color) * 64 + square


def encode_position(board: chess.Board) -> list[int]:
    """Return active feature indices for a position (white perspective)."""
    features = []
    for sq, piece in board.piece_map().items():
        features.append(feature_index(piece.piece_type, piece.color, sq))
    return sorted(features)


def generate_positions_from_seed(
    seed_fen: str,
    stockfish_path: str,
    num_games: int,
    depth: int,
    max_positions_per_seed: int,
) -> list[tuple[list[int], int, float]]:
    """
    Generate positions from a seed by playing random/Stockfish games.

    Returns list of (feature_indices, score_cp, wdl_win_prob) tuples.
    """
    results = []
    seen_fens = set()

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"Error opening Stockfish: {e}", file=sys.stderr)
        return results

    try:
        for game_idx in range(num_games):
            if len(results) >= max_positions_per_seed:
                break

            board = chess.Board(seed_fen)

            # Play random opening moves (5-10 moves)
            num_random = random.randint(5, 10)
            for _ in range(num_random):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(random.choice(legal))

            # Play a game mixing Stockfish moves with occasional random moves
            ply = 0
            next_random_ply = random.randint(8, 12)

            while not board.is_game_over() and ply < 300:
                ply += 1

                # Should we add this position to training data?
                fen_key = board.board_fen() + " " + ("w" if board.turn else "b")
                if (
                    ply >= 1
                    and not board.is_check()
                    and fen_key not in seen_fens
                ):
                    try:
                        info = engine.analyse(
                            board,
                            chess.engine.Limit(depth=depth),
                        )
                        score = info["score"].white()

                        # Skip mate scores
                        if score.is_mate():
                            pass
                        else:
                            cp = score.score()
                            if cp is not None and abs(cp) <= 10000:
                                # Get WDL if available
                                wdl = info.get("wdl")
                                if wdl is not None:
                                    # WDL is (wins, draws, losses) per mille from side to move
                                    w, d, l = wdl
                                    # Convert to win probability from white's perspective
                                    if board.turn == chess.WHITE:
                                        win_prob = w / 1000.0
                                    else:
                                        win_prob = l / 1000.0
                                else:
                                    # Estimate from score using sigmoid
                                    win_prob = 1.0 / (1.0 + 10.0 ** (-cp / 400.0))

                                features = encode_position(board)
                                results.append((features, cp, win_prob))
                                seen_fens.add(fen_key)

                                if len(results) >= max_positions_per_seed:
                                    break
                    except Exception:
                        pass

                # Make next move
                legal = list(board.legal_moves)
                if not legal:
                    break

                if ply >= next_random_ply:
                    # Inject a random move for diversity
                    board.push(random.choice(legal))
                    next_random_ply = ply + random.randint(8, 12)
                else:
                    # Use Stockfish at low depth for speed
                    try:
                        result = engine.play(
                            board,
                            chess.engine.Limit(depth=random.randint(1, 4)),
                        )
                        board.push(result.move)
                    except Exception:
                        board.push(random.choice(legal))
    finally:
        engine.quit()

    return results


def worker_fn(args):
    """Worker function for multiprocessing."""
    seed_fens, stockfish_path, games_per_seed, depth, positions_per_seed = args
    all_results = []
    for fen in seed_fens:
        results = generate_positions_from_seed(
            fen, stockfish_path, games_per_seed, depth, positions_per_seed,
        )
        all_results.extend(results)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Generate NNUE training data using Stockfish"
    )
    parser.add_argument(
        "--num-positions", type=int, default=1_000_000,
        help="Target number of positions to generate",
    )
    parser.add_argument(
        "--depth", type=int, default=8,
        help="Stockfish analysis depth for labeling",
    )
    parser.add_argument(
        "--output", type=str, default="data/training_data.npz",
        help="Output .npz file path",
    )
    parser.add_argument(
        "--stockfish", type=str, default="stockfish",
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, os.cpu_count() // 2),
        help="Number of worker processes",
    )
    args = parser.parse_args()

    # Verify Stockfish is available
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        engine.quit()
    except Exception as e:
        print(f"Error: Cannot find Stockfish at '{args.stockfish}': {e}")
        print("Install Stockfish or provide path with --stockfish")
        sys.exit(1)

    print(f"Generating {args.num_positions} positions at depth {args.depth}")
    print(f"Using {args.workers} workers")
    print(f"Output: {args.output}")

    seed_fens = SEED_POSITIONS
    positions_per_seed = (args.num_positions // len(seed_fens)) + 1
    games_per_seed = max(10, positions_per_seed // 30)

    # Split seeds across workers
    chunks = [[] for _ in range(args.workers)]
    for i, fen in enumerate(seed_fens):
        chunks[i % args.workers].append(fen)

    worker_args = [
        (chunk, args.stockfish, games_per_seed, args.depth, positions_per_seed)
        for chunk in chunks
        if chunk
    ]

    start_time = time.time()

    if args.workers == 1:
        all_results = worker_fn(worker_args[0])
    else:
        with multiprocessing.Pool(args.workers) as pool:
            results_list = pool.map(worker_fn, worker_args)
        all_results = []
        for r in results_list:
            all_results.extend(r)

    elapsed = time.time() - start_time
    print(f"Generated {len(all_results)} positions in {elapsed:.1f}s")

    if not all_results:
        print("No positions generated!")
        sys.exit(1)

    # Deduplicate by keeping first occurrence
    seen = set()
    unique_results = []
    for features, score, wdl in all_results:
        key = tuple(features)
        if key not in seen:
            seen.add(key)
            unique_results.append((features, score, wdl))

    print(f"After dedup: {len(unique_results)} unique positions")

    # Convert to numpy arrays
    # Store as variable-length feature lists using a padded array
    max_features = max(len(f) for f, _, _ in unique_results)
    n = len(unique_results)

    # Feature indices (padded with -1)
    feature_indices = np.full((n, max_features), -1, dtype=np.int16)
    feature_counts = np.zeros(n, dtype=np.int8)
    scores = np.zeros(n, dtype=np.int16)
    wdl_probs = np.zeros(n, dtype=np.float32)

    for i, (features, score, wdl) in enumerate(unique_results):
        feature_indices[i, :len(features)] = features
        feature_counts[i] = len(features)
        scores[i] = score
        wdl_probs[i] = wdl

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        feature_indices=feature_indices,
        feature_counts=feature_counts,
        scores=scores,
        wdl_probs=wdl_probs,
    )

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved to {args.output} ({file_size:.1f} MB)")
    print(f"Positions: {n}, Score range: [{scores.min()}, {scores.max()}]")


if __name__ == "__main__":
    main()
