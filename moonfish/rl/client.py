"""Client for the Chess OpenEnv environment."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .models import ChessAction, ChessObservation, ChessState


@dataclass
class StepResult:
    """Result from a step() call."""

    observation: ChessObservation
    reward: float
    done: bool


class ChessEnvClient:
    """
    HTTP client for the Chess OpenEnv environment.

    Provides a simple interface to interact with a remote chess environment
    server for reinforcement learning.

    Example usage:
        client = ChessEnvClient("http://localhost:8000")
        obs = client.reset()
        print(f"Legal moves: {obs.legal_moves}")

        result = client.step(ChessAction(move="e2e4"))
        print(f"Reward: {result.reward}, Done: {result.done}")

        state = client.state()
        print(f"Move count: {state.step_count}")

        client.close()
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Initialize the chess environment client.

        Args:
            base_url: URL of the chess environment server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        fen: Optional[str] = None,
    ) -> ChessObservation:
        """
        Reset the environment and start a new episode.

        Args:
            seed: Random seed (optional)
            episode_id: Unique episode identifier (optional)
            fen: Starting position in FEN notation (optional)

        Returns:
            Initial observation of the board state
        """
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if fen is not None:
            payload["fen"] = fen

        response = self._client.post(f"{self.base_url}/reset", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_observation(data)

    def step(self, action: ChessAction) -> StepResult:
        """
        Execute a move in the environment.

        Args:
            action: The chess action (move in UCI format)

        Returns:
            StepResult with observation, reward, and done flag
        """
        payload = {"move": action.move}
        response = self._client.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        data = response.json()

        return StepResult(
            observation=self._parse_observation(data["observation"]),
            reward=data["reward"],
            done=data["done"],
        )

    def state(self) -> ChessState:
        """
        Get the current episode state.

        Returns:
            Current episode state with metadata
        """
        response = self._client.get(f"{self.base_url}/state")
        response.raise_for_status()
        data = response.json()

        return ChessState(
            episode_id=data["episode_id"],
            step_count=data["step_count"],
            current_player=data["current_player"],
            fen=data["fen"],
            move_history=data.get("move_history", []),
        )

    def metadata(self) -> Dict[str, Any]:
        """
        Get environment metadata.

        Returns:
            Dictionary with environment configuration
        """
        response = self._client.get(f"{self.base_url}/metadata")
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        """
        Check if the server is healthy.

        Returns:
            True if server is responding
        """
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _parse_observation(self, data: Dict[str, Any]) -> ChessObservation:
        """Parse observation from JSON response."""
        return ChessObservation(
            fen=data["fen"],
            legal_moves=data["legal_moves"],
            is_check=data.get("is_check", False),
            done=data.get("done", False),
            reward=data.get("reward"),
            result=data.get("result"),
            metadata=data.get("metadata", {}),
        )


# Convenience function for quick usage
def make_env(base_url: str = "http://localhost:8000") -> ChessEnvClient:
    """
    Create a chess environment client.

    Args:
        base_url: URL of the chess environment server

    Returns:
        ChessEnvClient instance
    """
    return ChessEnvClient(base_url)
