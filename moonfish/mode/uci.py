import sys

from chess import WHITE, Board, STARTING_FEN
from moonfish.config import Config
from moonfish.helper import find_best_move, get_engine

# UCI based on Sunfish Engine: https://github.com/thomasahle/sunfish/blob/master/uci.py


def _parse_go_params(params: list[str]) -> dict[str, int]:
    """Parse 'go' command parameters into a dict."""
    result: dict[str, int] = {}
    i = 0
    while i < len(params):
        key = params[i]
        if key in ("wtime", "btime", "winc", "binc", "movetime", "movestogo", "depth"):
            if i + 1 < len(params):
                try:
                    result[key] = int(params[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
        elif key == "infinite":
            result["infinite"] = 1
        i += 1
    return result


def _calculate_time_limit(
    go_params: dict[str, int], side_to_move_is_white: bool
) -> float | None:
    """
    Calculate time limit in seconds from UCI go parameters.
    Returns None if the engine should use fixed depth.
    """
    # Fixed time per move (movetime)
    if "movetime" in go_params:
        # Use movetime minus a 1-second safety margin for Python/GC overhead
        movetime_ms = go_params["movetime"]
        return max((movetime_ms - 1000) / 1000.0, movetime_ms / 1000.0 * 0.5)

    # Clock-based time management
    time_key = "wtime" if side_to_move_is_white else "btime"
    inc_key = "winc" if side_to_move_is_white else "binc"

    if time_key in go_params:
        remaining_ms = go_params[time_key]
        increment_ms = go_params.get(inc_key, 0)
        moves_to_go = go_params.get("movestogo", 30)

        # Allocate time: remaining / moves_to_go + most of the increment
        time_for_move_ms = remaining_ms / moves_to_go + increment_ms * 0.8

        # Don't use more than 25% of remaining time on one move
        time_for_move_ms = min(time_for_move_ms, remaining_ms * 0.25)

        # Safety margin for Python/GC overhead
        time_for_move_ms = max(time_for_move_ms - 500, 10)

        return time_for_move_ms / 1000.0

    return None


def main(config: Config):
    """
    Start the command line user interface (UCI based).
    """
    # init board and engine
    board = Board()
    engine = get_engine(config)

    # keep listening to UCI commands
    while True:
        # get command from stdin
        uci_command = input().strip()
        uci_parameters = uci_command.split(" ")

        if uci_command == "quit":
            sys.exit()

        elif uci_command == "uci":
            # engine details
            print("id name Moonfish")
            print("id author luccabb")
            print("uciok")

        elif uci_command == "isready":
            # engine ready to receive commands
            print("readyok")

        elif uci_command == "ucinewgame":
            # start new game
            board = Board()

        elif uci_command.startswith("position"):
            moves_idx = uci_command.find("moves")

            # get moves from UCI command
            if moves_idx >= 0:
                moveslist = uci_command[moves_idx:].split()[1:]
            else:
                moveslist = []

            # get FEN from uci command
            if uci_parameters[1] == "fen":
                if moves_idx >= 0:
                    fenpart = uci_command[:moves_idx]
                    _, _, fen = fenpart.split(" ", 2)
                else:
                    fen = " ".join(uci_parameters[2:])

            elif uci_parameters[1] == "startpos":
                fen = STARTING_FEN
            else:
                raise SyntaxError("UCI Syntax error.")

            # start board and make moves
            board = Board(fen)
            for move in moveslist:
                board.push_uci(move)

        elif uci_command.startswith("go"):
            go_params = _parse_go_params(uci_parameters[1:])

            # If depth is specified in go command, use it
            if "depth" in go_params:
                config.negamax_depth = go_params["depth"]

            # Calculate time limit from go parameters
            time_limit = _calculate_time_limit(
                go_params, board.turn == WHITE
            )

            if time_limit is not None and hasattr(engine, "search_move_timed"):
                best_move = engine.search_move_timed(board, time_limit)
            else:
                best_move = find_best_move(
                    board=board,
                    engine=engine,
                )
            print(f"bestmove {best_move}")
