"""FastAPI server for the Chess OpenEnv environment."""

from pathlib import Path
from typing import Any, Dict, Optional

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from moonfish.lib import search_move
from ..models import ChessAction
from .chess_environment import ChessEnvironment


# Pydantic models for API requests/responses
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    fen: Optional[str] = None


class StepRequest(BaseModel):
    move: str


class EngineMoveRequest(BaseModel):
    fen: str
    depth: int = 2


class ObservationResponse(BaseModel):
    fen: str
    legal_moves: list[str]
    is_check: bool = False
    done: bool = False
    reward: Optional[float] = None
    result: Optional[str] = None
    metadata: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float
    done: bool


class StateResponse(BaseModel):
    episode_id: str
    step_count: int
    current_player: str
    fen: str
    move_history: list[str]


# Create FastAPI app
app = FastAPI(
    title="Chess OpenEnv",
    description="Chess environment for reinforcement learning using moonfish",
    version="1.0.0",
)

# Serve static files for the web UI
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global environment instance (for single-player mode)
# For multi-player, you'd want a session manager
_env: Optional[ChessEnvironment] = None


def get_env() -> ChessEnvironment:
    """Get or create environment instance."""
    global _env
    if _env is None:
        _env = ChessEnvironment()
    return _env


@app.get("/")
def root():
    """Serve the chess UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Moonfish Chess API", "docs": "/docs"}


@app.get("/web")
def web():
    """Serve the chess UI (for HF Spaces base_path)."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Moonfish Chess API", "docs": "/docs"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/engine-move")
def engine_move(request: EngineMoveRequest):
    """Get the best move from moonfish engine for a given position."""
    try:
        board = chess.Board(request.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    depth = max(1, min(request.depth, 4))  # Clamp depth 1-4
    move = search_move(board, depth=depth)

    return {"move": move.uci(), "fen": request.fen}


@app.get("/metadata")
def metadata():
    """Get environment metadata."""
    return get_env().get_metadata()


@app.post("/reset", response_model=ObservationResponse)
def reset(request: ResetRequest):
    """Reset the environment and start a new episode."""
    env = get_env()
    obs = env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        fen=request.fen,
    )
    return ObservationResponse(
        fen=obs.fen,
        legal_moves=obs.legal_moves,
        is_check=obs.is_check,
        done=obs.done,
        reward=obs.reward,
        result=obs.result,
        metadata=obs.metadata,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Execute a move and return the result."""
    env = get_env()

    try:
        action = ChessAction(move=request.move)
        obs, reward, done = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=ObservationResponse(
            fen=obs.fen,
            legal_moves=obs.legal_moves,
            is_check=obs.is_check,
            done=obs.done,
            reward=obs.reward,
            result=obs.result,
            metadata=obs.metadata,
        ),
        reward=reward,
        done=done,
    )


@app.get("/state", response_model=StateResponse)
def state():
    """Get current episode state."""
    env = get_env()
    try:
        s = env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StateResponse(
        episode_id=s.episode_id,
        step_count=s.step_count,
        current_player=s.current_player,
        fen=s.fen,
        move_history=s.move_history,
    )


def main():
    """Entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
