import sys

from chess import STARTING_FEN, Board

from moonfish.config import Config
from moonfish.helper import find_best_move, get_engine

# UCI based on Sunfish Engine: https://github.com/thomasahle/sunfish/blob/master/uci.py


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
            best_move = find_best_move(
                board=board,
                engine=engine,
            )
            print(f"bestmove {best_move}")
