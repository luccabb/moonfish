<p align="center">
    <img src="moonfish.png" alt="moonfish" width="200"/>
</p>

# Moonfish Engine ([~2000 Elo Rating Lichess.org](https://lichess.org/@/moonfish_bot))

Moonfish is a didactic Python chess engine designed to showcase parallel search algorithms and modern chess programming techniques. Built with code readability as a priority, Moonfish makes advanced concepts easily accessible providing a more approachable alternative to cpp engines. 

The engine achieves approximately ~2000 Elo when playing against Lichess Stockfish bots (beats level 5 and loses to level 6) and includes comprehensive test suites including the Bratko-Kopec tactical test positions.

# Quickstart

## Requirements
- Python 3.10

## Installation and usage
Install the python library:
```shell
pip install moonfish
```

From python:
```python
$ python
>>> import chess
>>> import moonfish
>>> board = chess.Board()
>>> moonfish.search_move(board)
Move.from_uci('g1f3')
```

You can also call the CLI, the CLI works as an [UCI](http://wbec-ridderkerk.nl/html/UCIProtocol.html) Compatible Engine:
```shell
$ moonfish --mode=uci
uci # <- user input
id name Moonfish
id author luccabb
uciok
```

You can also run it as an API:
```shell
moonfish --mode=api
```

Then send a request:
```shell
$ curl "http://localhost:5000/?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201&depth=4&quiescence_search_depth=3&null_move=True&null_move_r=2&algorithm=alpha_beta"
{
  "body": {
    "move": "e2e4"
  },
  "headers": {
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,GET",
    "Access-Control-Allow-Origin": "*"
  },
  "statusCode": 200
}
```

## Features

### Search Algorithms
- **Alpha-Beta Pruning** - Negamax with α-β cutoffs
- **Lazy SMP** - Shared memory parallel search utilizing all CPU cores  
- **Layer-based Parallelization** - Distributing work at specific search depths
- **Null Move Pruning** - Skip moves to detect zugzwang positions
- **Quiescence Search** - Extended search for tactical positions

### Evaluation & Optimization
- **PeSTO Evaluation** - Piece-square tables (PST) with tapered evaluation. [Using Rofchade's PST](https://talkchess.com/viewtopic.php?t=68311&start=19).
- **Transposition Tables** - Caching to avoid redundant calculations
- **Move Ordering** - MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
- **Syzygy Tablebase** support for perfect endgame play
- **Opening Book** integration (Cerebellum format)

### Engine Interfaces
- **UCI Protocol** - Compatible with popular chess GUIs
- **Web API** - RESTful interface for online integration
- **Lichess Bot** - Ready for deployment on Lichess.org

## Lichess-bot Python Bridge

This engine implements the UCI protocol and can be used as a bot on [Lichess](https://lichess.org). You can use the python bridge between Lichess Bot API and the engine: [https://github.com/ShailChoksi/lichess-bot](https://github.com/ShailChoksi/lichess-bot).

To run it as a bot you'll need to produce a python executable. [PyInstaller](https://pyinstaller.readthedocs.io/en/stable/) can produce it by running the following command:

```shell
python3 -m PyInstaller main.py
```

This creates a `build` and `dist` folder. The `dist` folder contains the main executable in a folder called `main`. All the files inside `main` need to be copied over to `/lichess-bot/engines` for it to work. You can checkout [/lichess](lichess/README.md) for further lichess setup.

## Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--mode` | Engine Mode | `uci` | `uci`, `api` |
| `--algorithm` | Search algorithm | `alpha_beta` | `alpha_beta`, `lazy_smp`, `parallel_alpha_beta_layer_1` |
| `--depth` | Search depth | `3` | `1-N` |
| `--null-move` | Whether to use null move pruning | `False` | `True`, `False` |
| `--null-mov-r` | Null move reduction factor | `2` | `1-N` |
| `--quiescence-search-depth` | Max depth of quiescence search | `3` | `1-N` |
| `--syzygy-path` | Tablebase directory | `None` | Valid path |

## Contributing

We welcome contributions, feel free to open PRs/Issues! Areas of interest:
- New search algorithms
- Improved evaluation functions
- Time constrained search (e.g. find the best move in 40s)
- Additional test positions
- Github CI testing
- Different evaluation functions
- Neural Net integration
- Performance benchmarking on different hardware
- Improving caching

## References

- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [python-chess library](https://python-chess.readthedocs.io/)
- [Lazy SMP Algorithm](https://www.chessprogramming.org/Lazy_SMP)
- [UCI Protocol Specification](http://wbec-ridderkerk.nl/html/UCIProtocol.html)
- [Rofchade](https://talkchess.com/viewtopic.php?t=68311&start=19)
- [THE BRATKO-KOPEC TEST RECALIBRATED](https://www.sci.brooklyn.cuny.edu/~kopec/Publications/THE%20BRATKO-KOPEC%20TEST%20RECALIBRATED.htm)

## License

MIT License - see [LICENSE](LICENSE) file for details.
