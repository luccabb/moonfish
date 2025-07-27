
# Contributing to Moonfish

Thank you for your interest in contributing to Moonfish! This guide will help you get started with developing this Python chess engine, whether you're looking to fix bugs, add new features, or improve existing algorithms.

## What is Moonfish?

Moonfish is a didactic chess engine designed to showcase parallel search algorithms and modern chess programming techniques. With approximately 2000 Elo strength, it demonstrates concepts like:

- Alpha-beta pruning with advanced optimizations
- Parallel search algorithms (Lazy SMP, layer-based parallelization)
- Chess-specific optimizations (null move pruning, quiescence search)
- Modern evaluation techniques (piece-square tables, tapered evaluation)

## Getting Started

### Prerequisites

- Python 3.10 or higher

## Development Environment Setup

1. **Clone the repository:**
   ```shell
   $ git clone https://github.com/luccabb/moonfish.git
   $ cd moonfish
   ```

2. **Set up the development environment:**
   ```shell
   # Create virtual environment and install dependencies
   $ make install
   
   # Activate the environment
   $ . .venv/bin/activate || source .venv/bin/activate
   ```

3. **Verify the installation:**
   ```shell
   # Test UCI mode
   $ moonfish --mode=uci
   uci
   id name Moonfish
   id author luccabb
   uciok
   
   # View all available options
   $ moonfish --help
   ```

## Running Tests

### Unit Tests

Unit tests are testing the basic functionality of the engine,
with key positions and moves.

```shell
python -m unittest tests/test.py
```

### [Bratko-Kopec Test](https://www.chessprogramming.org/Bratko-Kopec_Test)

The [Bratko-Kopec](https://www.chessprogramming.org/Bratko-Kopec_Test) test suite evaluates the engine's performance in terms of both speed and tactical/positional strength.

```shell
python -m tests.test_bratko_kopec
```

## Engine Configuration

Moonfish uses a configuration system that allows fine-tuning of search algorithms and evaluation parameters. Understanding these options is crucial for development and optimization.

### Configuration Parameters

The `Config` class in `moonfish/config.py` defines all engine parameters:

#### Core Search Settings

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

### Configuration Examples

#### CLI
```bash
moonfish --algorithm=alpha_beta --depth=3 --null-move=false --quiescence-search-depth=2
```

#### API Usage with Configuration
```bash
curl "http://localhost:5000/?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201&depth=4&algorithm=lazy_smp&null_move=true&null_move_r=2&quiescence_search_depth=3"
```

## Lichess-bot Python Bridge

This engine implements the UCI protocol and can be used as a bot on [Lichess](https://lichess.org). You can use the python bridge between Lichess Bot API and the engine: [https://github.com/ShailChoksi/lichess-bot](https://github.com/ShailChoksi/lichess-bot).

To run it as a bot you'll need to produce a python executable. We use [PyInstaller](https://pyinstaller.readthedocs.io/en/stable/) via our Makefile:

```shell
make build-lichess
```

This creates a `build` and `dist` folder. The `dist` folder contains the main executable in a folder called `main`. All the files inside `main` need to be copied over to `/lichess-bot/engines` for it to work. You can checkout [/lichess](lichess/README.md) for further lichess setup.

## Adding a New Chess Engine

Want to implement a new search algorithm or evaluation technique? This guide walks you through creating a new engine from scratch.

### Step 1: Create Your Engine File

Create a new file in `moonfish/engines/` for your engine:

```bash
touch moonfish/engines/my_new_engine.py
```

### Step 2: Implement the Engine Class

Your engine must implement the `ChessEngine` protocol. Here's a basic template:

```python
from chess import Board
from moonfish.engines.base_engine import ChessEngine
from moonfish.config import Config

class MyNewEngine:
    """
    Brief description of your algorithm.
    
    Example: Implements Monte Carlo Tree Search with UCB1 selection.
    """
    
    def __init__(self, config: Config):
        self.config = config
        # Initialize any data structures your algorithm needs
    
    def search_move(self, board: Board) -> str:
        """
        Main search method - must return a UCI move string.
        
        Arguments:
            board: Current chess position
            
        Returns:
            UCI move string (e.g., "e2e4", "g1f3")
        """
        # Your algorithm implementation here
        # Must return a valid UCI move string
        best_move = self.my_algorithm(board)
        return best_move.uci()  # Always return UCI string
```

### Step 3: Register Your Engine

Add your engine to the algorithm registry in `moonfish/helper.py`:

1. **Add the import:**
```python
from moonfish.engines.my_new_engine import MyNewEngine
```

2. **Add to Algorithm enum:**
```python
class Algorithm(Enum):
    # ... existing algorithms
    my_new_algorithm = "my_new_algorithm"
```

3. **Add to engine factory:**
```python
def get_engine(config: Config):
    # ... existing conditions
    elif algorithm is Algorithm.my_new_algorithm:
        return MyNewEngine(config)
```

You could also optionally integrate your engine with the existing test files and see how it performs against other methods, or test it in real games using the Lichess API integration.
