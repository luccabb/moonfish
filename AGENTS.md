# Agents in Moonfish Chess Engine

This document describes the different search agents (engines) implemented in the Moonfish chess engine codebase.

## Overview

Moonfish implements multiple chess search algorithms as "agents" that can be selected via the `--algorithm` parameter. Each agent represents a different approach to finding the best chess move.

## Available Agents

### 1. Alpha-Beta (`alpha_beta`)
**Location**: `moonfish/engines/alpha_beta.py:16`

The core minimax search algorithm with alpha-beta pruning optimization.

**Features**:
- Negamax implementation with α-β cutoffs
- Quiescence search for tactical positions
- Null move pruning to detect zugzwang
- Transposition table caching
- Syzygy tablebase integration
- MVV-LVA move ordering

**Usage**:
```shell
moonfish --algorithm=alpha_beta --depth=4
```

### 2. Lazy SMP (`lazy_smp`)
**Location**: `moonfish/engines/lazy_smp.py:9`

Parallel search implementation utilizing shared memory multiprocessing.

**Features**:
- Inherits all Alpha-Beta capabilities
- Distributes search across all CPU cores
- Shared transposition table between processes
- Automatic process count detection (`cpu_count()`)

**Usage**:
```shell
moonfish --algorithm=lazy_smp --depth=4
```

### 3. Layer-1 Parallel Alpha-Beta (`parallel_alpha_beta_layer_1`)
**Location**: `moonfish/engines/l1p_alpha_beta.py`

Parallelization at the first search layer, distributing root moves across processes.

### 4. Layer-2 Parallel Alpha-Beta (`parallel_alpha_beta_layer_2`)
**Location**: `moonfish/engines/l2p_alpha_beta.py`

Extended parallelization to the second search layer for deeper work distribution.

### 5. Random Agent (`random`)
**Location**: `moonfish/engines/random.py`

Simple random move selection for testing and baseline comparison.

## Agent Architecture

All agents implement the `ChessEngine` protocol defined in `moonfish/engines/base_engine.py:9`:

```python
class ChessEngine(Protocol):
    def __init__(self, config: Config): ...
    
    @abstractmethod
    def search_move(self, board: Board) -> Optional[str]:
        """Return the best move for the given board position."""
        raise NotImplementedError()
```

## Configuration

Agents are configured via the `Config` class in `moonfish/config.py` with parameters:

- `negamax_depth`: Search depth limit
- `null_move`: Enable null move pruning
- `null_move_r`: Null move reduction factor
- `quiescence_search_depth`: Extended tactical search depth
- `syzygy_path`: Tablebase directory path
- `checkmate_score`: Mate evaluation value

## Performance Characteristics

- **Alpha-Beta**: Single-threaded, reliable baseline
- **Lazy SMP**: Best overall performance, scales with CPU cores
- **Layer-based**: Specialized parallelization for specific search patterns
- **Random**: Instant move generation, no evaluation

## Implementation Notes

- All parallel agents use Python's `multiprocessing` module
- Shared transposition tables implemented via `Manager().dict()`
- Move ordering uses MVV-LVA heuristic for capture prioritization
- Quiescence search extends tactical analysis beyond main search depth