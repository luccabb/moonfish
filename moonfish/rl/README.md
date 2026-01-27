---
title: Moonfish Chess
emoji: ♟️
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Chess OpenEnv

A chess environment for reinforcement learning, built on [moonfish](https://github.com/luccab/moonfish) and compatible with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Features

- **Full Chess Rules**: Legal move generation, checkmate/stalemate detection, draw conditions
- **Position Evaluation**: PeSTO evaluation function from moonfish for reward shaping
- **OpenEnv Compatible**: Standard `reset()`, `step()`, `state()` interface
- **Configurable Rewards**: Win/loss/draw payoffs, illegal move penalties, evaluation-based rewards
- **HTTP API**: FastAPI server for remote training and multi-agent setups
- **Containerized**: Docker support for reproducible deployments

## Quick Start

### Local Usage (No Server)

```python
from moonfish.rl import ChessEnvironment, ChessAction

# Create environment
env = ChessEnvironment()

# Start a new game
obs = env.reset()
print(f"Legal moves: {obs.legal_moves}")

# Make a move
action = ChessAction(move="e2e4")
obs, reward, done = env.step(action)

print(f"FEN: {obs.fen}")
print(f"Reward: {reward}, Done: {done}")
```

### Client-Server Usage

Start the server:

```bash
cd moonfish/rl
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Connect with the client:

```python
from moonfish.rl import ChessEnvClient, ChessAction

client = ChessEnvClient("http://localhost:8000")

obs = client.reset()
result = client.step(ChessAction(move="e2e4"))
print(f"Reward: {result.reward}")

client.close()
```

## Data Models

### ChessAction
```python
@dataclass
class ChessAction:
    move: str  # UCI format: "e2e4", "e7e8q" (promotion)
```

### ChessObservation
```python
@dataclass
class ChessObservation:
    fen: str              # Board state in FEN notation
    legal_moves: List[str]  # Available moves in UCI format
    is_check: bool        # Current player in check
    done: bool            # Game over
    reward: Optional[float]  # Terminal reward
    result: Optional[str]    # "1-0", "0-1", "1/2-1/2"
    metadata: Dict[str, Any]  # Evaluation, material, etc.
```

### ChessState
```python
@dataclass
class ChessState:
    episode_id: str        # Unique game identifier
    step_count: int        # Half-moves played
    current_player: str    # "white" or "black"
    fen: str               # Current position
    move_history: List[str]  # All moves in UCI format
```

## Reward Configuration

```python
from moonfish.rl import ChessEnvironment, RewardConfig

config = RewardConfig(
    win=1.0,           # Reward for winning
    loss=-1.0,         # Penalty for losing
    draw=0.0,          # Reward for draw
    illegal_move=-0.1, # Penalty for illegal moves
    use_evaluation=True,  # Enable intermediate rewards
    evaluation_scale=0.0001,  # Scale for eval-based rewards
)

env = ChessEnvironment(reward_config=config)
```

## Docker

Build and run:

```bash
docker build -t chess-openenv .
docker run -p 8000:8000 chess-openenv
```

## Integration with RL Frameworks

### With TorchRL

```python
from moonfish.rl import ChessEnvironment, ChessAction

class ChessTorchRLWrapper:
    def __init__(self):
        self.env = ChessEnvironment()

    def reset(self):
        obs = self.env.reset()
        return self._obs_to_tensor(obs)

    def step(self, action_idx):
        move = self._idx_to_move(action_idx)
        obs, reward, done = self.env.step(ChessAction(move=move))
        return self._obs_to_tensor(obs), reward, done
```

### With OpenEnv Training Loop

```python
from moonfish.rl import make_env, ChessAction
import random

client = make_env("http://localhost:8000")

for episode in range(100):
    obs = client.reset()
    episode_reward = 0

    while not obs.done:
        # Your policy here (random for demo)
        move = random.choice(obs.legal_moves)
        result = client.step(ChessAction(move=move))
        obs = result.observation
        episode_reward += result.reward

    print(f"Episode {episode}: reward={episode_reward}")

client.close()
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metadata` | GET | Environment configuration |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute a move |
| `/state` | GET | Get episode metadata |

## License

MIT - See the moonfish repository for full license details.
