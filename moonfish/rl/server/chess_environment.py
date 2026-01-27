"""Chess environment for OpenEnv using moonfish."""

import random
import uuid
from typing import Any, Dict, Optional, Tuple

import chess

from moonfish.lib import search_move
from moonfish.psqt import board_evaluation, count_pieces, get_phase, MG_PIECE_VALUES
from ..models import ChessAction, ChessObservation, ChessState, RewardConfig


class ChessEnvironment:
    """
    Chess environment implementing the OpenEnv interface.

    Uses python-chess for game logic and moonfish for position evaluation.
    Designed for RL training where an agent plays as one color against
    an opponent (which can be random, moonfish engine, or self-play).
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        max_moves: int = 500,
        agent_color: Optional[
            bool
        ] = None,  # None = alternate, True = White, False = Black
        opponent: Optional[
            str
        ] = None,  # None = self-play, "moonfish" = moonfish engine, "random" = random
        opponent_depth: int = 2,  # Search depth for moonfish opponent
    ):
        """
        Initialize the chess environment.

        Args:
            reward_config: Configuration for reward shaping
            max_moves: Maximum half-moves before draw (prevents infinite games)
            agent_color: Which color the RL agent plays (None = alternates each episode)
            opponent: Opponent type - None (self-play), "moonfish", or "random"
            opponent_depth: Search depth when using moonfish as opponent
        """
        self.reward_config = reward_config or RewardConfig()
        self.max_moves = max_moves
        self.agent_color_setting = agent_color
        self.opponent = opponent
        self.opponent_depth = opponent_depth

        # Will be set on reset
        self._board: Optional[chess.Board] = None
        self._state: Optional[ChessState] = None
        self._agent_color: bool = chess.WHITE

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        fen: Optional[str] = None,
        **kwargs,
    ) -> ChessObservation:
        """
        Initialize a new chess game episode.

        Args:
            seed: Random seed (unused for now, chess is deterministic)
            episode_id: Unique identifier for this episode
            fen: Optional starting position in FEN notation

        Returns:
            Initial observation of the board state
        """
        # Create new board
        if fen:
            self._board = chess.Board(fen)
        else:
            self._board = chess.Board()

        # Determine agent color
        if self.agent_color_setting is None:
            # Alternate each episode based on episode_id hash
            if episode_id:
                self._agent_color = hash(episode_id) % 2 == 0
            else:
                self._agent_color = chess.WHITE
        else:
            self._agent_color = self.agent_color_setting

        # Initialize state
        self._state = ChessState(
            episode_id=episode_id or uuid.uuid4().hex,
            step_count=0,
            current_player="white" if self._board.turn else "black",
            fen=self._board.fen(),
            move_history=[],
        )

        # If agent plays Black and opponent is configured, opponent moves first
        if self.opponent is not None and self._agent_color == chess.BLACK:
            self._make_opponent_move()

        return self._get_observation()

    def step(
        self, action: ChessAction, timeout_s: Optional[float] = None, **kwargs
    ) -> Tuple[ChessObservation, float, bool]:
        """
        Execute a chess move and return the resulting state.

        Args:
            action: The move to make in UCI format (e.g., "e2e4")
            timeout_s: Unused timeout parameter

        Returns:
            Tuple of (observation, reward, done)
        """
        if self._board is None or self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Parse the move
        try:
            move = chess.Move.from_uci(action.move)
        except ValueError:
            # Invalid move format
            return self._handle_illegal_move(f"Invalid move format: {action.move}")

        # Check if move is legal
        if move not in self._board.legal_moves:
            return self._handle_illegal_move(f"Illegal move: {action.move}")

        # Execute the move
        self._board.push(move)
        self._state.step_count += 1
        self._state.move_history.append(action.move)
        self._state.current_player = "white" if self._board.turn else "black"
        self._state.fen = self._board.fen()

        # Calculate reward and check for game end
        reward, done = self._calculate_reward_and_done()

        # If game not over and opponent is configured, make opponent move
        if not done and self.opponent is not None:
            self._make_opponent_move()
            # Recalculate after opponent move
            opp_reward, done = self._calculate_reward_and_done()
            # Opponent's reward is negative of ours (zero-sum)
            reward += -opp_reward if done else 0

        observation = self._get_observation(done=done, reward=reward if done else None)

        return observation, reward, done

    @property
    def state(self) -> ChessState:
        """Return the current episode state."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def close(self) -> None:
        """Clean up resources."""
        self._board = None
        self._state = None

    def get_metadata(self) -> Dict[str, Any]:
        """Return environment metadata."""
        return {
            "name": "chess",
            "version": "1.0.0",
            "max_moves": self.max_moves,
            "reward_config": {
                "win": self.reward_config.win,
                "loss": self.reward_config.loss,
                "draw": self.reward_config.draw,
                "illegal_move": self.reward_config.illegal_move,
                "use_evaluation": self.reward_config.use_evaluation,
                "evaluation_scale": self.reward_config.evaluation_scale,
            },
        }

    def _get_observation(
        self,
        done: bool = False,
        reward: Optional[float] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ChessObservation:
        """Build observation from current board state."""
        assert self._board is not None

        legal_moves = [move.uci() for move in self._board.legal_moves]

        metadata: Dict[str, Any] = {}

        # Add evaluation if configured
        if self.reward_config.use_evaluation:
            metadata["evaluation"] = board_evaluation(self._board)

        # Add material count
        metadata["material"] = self._get_material_count()

        # Add game phase (0 = opening, 256 = endgame)
        metadata["phase"] = get_phase(self._board)
        metadata["fullmove_number"] = self._board.fullmove_number
        metadata["halfmove_clock"] = self._board.halfmove_clock

        if error:
            metadata["error"] = error

        # Determine result string if game is over
        if done and result is None:
            result = self._get_result_string()

        return ChessObservation(
            fen=self._board.fen(),
            legal_moves=legal_moves,
            is_check=self._board.is_check(),
            done=done,
            reward=reward,
            result=result,
            metadata=metadata,
        )

    def _calculate_reward_and_done(self) -> Tuple[float, bool]:
        """Calculate reward and check if episode is done."""
        assert self._board is not None

        # Check for game end
        if self._board.is_checkmate():
            # The side to move is checkmated, so the previous mover won
            winner = not self._board.turn
            if winner == self._agent_color:
                return self.reward_config.win, True
            else:
                return self.reward_config.loss, True

        if self._board.is_stalemate():
            return self.reward_config.draw, True

        if self._board.is_insufficient_material():
            return self.reward_config.draw, True

        if self._board.is_fifty_moves():
            return self.reward_config.draw, True

        if self._board.is_repetition(3):
            return self.reward_config.draw, True

        # Check move limit
        if self._state and self._state.step_count >= self.max_moves:
            return self.reward_config.draw, True

        # Game continues
        reward = 0.0

        # Optional: Add evaluation-based intermediate rewards
        if self.reward_config.use_evaluation:
            eval_score = board_evaluation(self._board)
            # Normalize evaluation to agent's perspective
            if self._board.turn != self._agent_color:
                eval_score = -eval_score
            reward = eval_score * self.reward_config.evaluation_scale

        return reward, False

    def _handle_illegal_move(
        self, error_msg: str
    ) -> Tuple[ChessObservation, float, bool]:
        """Handle an illegal move attempt."""
        observation = self._get_observation(done=False, error=error_msg)
        return observation, self.reward_config.illegal_move, False

    def _get_result_string(self) -> str:
        """Get the game result as a string."""
        assert self._board is not None

        if self._board.is_checkmate():
            return "1-0" if not self._board.turn else "0-1"
        return "1/2-1/2"

    def _get_material_count(self) -> Dict[str, int]:
        """Count material for both sides using moonfish piece values."""
        assert self._board is not None

        # count_pieces returns [wp, bp, wn, bn, wb, bb, wr, br, wq, bq]
        pieces = count_pieces(self._board)
        wp, bp, wn, bn, wb, bb, wr, br, wq, bq = pieces

        white = (
            wp * MG_PIECE_VALUES[chess.PAWN]
            + wn * MG_PIECE_VALUES[chess.KNIGHT]
            + wb * MG_PIECE_VALUES[chess.BISHOP]
            + wr * MG_PIECE_VALUES[chess.ROOK]
            + wq * MG_PIECE_VALUES[chess.QUEEN]
        )
        black = (
            bp * MG_PIECE_VALUES[chess.PAWN]
            + bn * MG_PIECE_VALUES[chess.KNIGHT]
            + bb * MG_PIECE_VALUES[chess.BISHOP]
            + br * MG_PIECE_VALUES[chess.ROOK]
            + bq * MG_PIECE_VALUES[chess.QUEEN]
        )

        return {"white": white, "black": black}

    def _make_opponent_move(self) -> None:
        """Make a move for the opponent using configured strategy."""
        assert self._board is not None
        assert self._state is not None

        if not list(self._board.legal_moves):
            return  # No legal moves (game should be over)

        if self.opponent == "moonfish":
            # Use moonfish engine to find best move
            move = search_move(self._board, depth=self.opponent_depth)
        elif self.opponent == "random":
            # Pick a random legal move
            move = random.choice(list(self._board.legal_moves))
        else:
            return  # No opponent configured

        # Execute opponent's move
        self._board.push(move)
        self._state.step_count += 1
        self._state.move_history.append(move.uci())
        self._state.current_player = "white" if self._board.turn else "black"
        self._state.fen = self._board.fen()
