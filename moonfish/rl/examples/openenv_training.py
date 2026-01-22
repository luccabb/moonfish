"""
OpenEnv Training Example

This example shows how to use the chess environment with the OpenEnv
client-server pattern, which is useful for:
- Distributed training across machines
- Isolated environment execution
- Integration with OpenEnv-compatible training frameworks

Usage:
    # Terminal 1: Start the server
    cd moonfish/rl
    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Terminal 2: Run this training script
    python examples/openenv_training.py
"""

import random

from moonfish.rl import ChessAction, make_env


def random_policy(legal_moves: list[str]) -> str:
    """Simple random policy for demonstration."""
    return random.choice(legal_moves)


def train_with_remote_env():
    """
    Training loop using the HTTP client (OpenEnv pattern).

    This pattern is useful when:
    - Environment runs on a different machine
    - You need environment isolation (sandboxing)
    - You're using OpenEnv-compatible training frameworks
    """
    # Connect to the environment server
    # For local testing, start the server first:
    #   python -m uvicorn moonfish.rl.server.app:app --port 8000
    client = make_env("http://localhost:8000")

    # Check server health
    if not client.health():
        print("Server not running. Start it with:")
        print("  python -m uvicorn moonfish.rl.server.app:app --port 8000")
        return

    print("Connected to chess environment server")
    print(f"Metadata: {client.metadata()}")
    print()

    # Training loop
    num_episodes = 5

    for episode in range(num_episodes):
        # Reset environment
        obs = client.reset()
        episode_reward = 0.0

        print(f"Episode {episode + 1}")

        while not obs.done:
            # Select action using policy
            move = random_policy(obs.legal_moves)
            action = ChessAction(move=move)

            # Step environment
            result = client.step(action)
            obs = result.observation
            episode_reward += result.reward

            # Safety limit
            state = client.state()
            if state.step_count > 200:
                print("  (truncated at 200 moves)")
                break

        print(
            f"  Moves: {client.state().step_count}, "
            f"Result: {obs.result or 'ongoing'}, "
            f"Reward: {episode_reward:.2f}"
        )

    # Cleanup
    client.close()
    print("\nTraining complete!")


def train_with_local_env():
    """
    Training loop using local environment (no server needed).

    This is simpler and faster for single-machine training.
    """
    from moonfish.rl import ChessEnvironment

    env = ChessEnvironment(opponent="random")

    print("Training with local environment (random opponent)")
    print()

    num_episodes = 5

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0

        while not obs.done:
            move = random_policy(obs.legal_moves)
            obs, reward, done = env.step(ChessAction(move=move))
            episode_reward += reward

            if env.state.step_count > 200:
                break

        print(
            f"Episode {episode + 1}: "
            f"Moves={env.state.step_count}, "
            f"Result={obs.result or 'ongoing'}, "
            f"Reward={episode_reward:.2f}"
        )

    env.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    import sys

    if "--remote" in sys.argv:
        print("=== Remote Environment (OpenEnv HTTP Client) ===\n")
        train_with_remote_env()
    else:
        print("=== Local Environment ===\n")
        train_with_local_env()
        print("\nTo test with HTTP client, run:")
        print(
            "  1. Start server: python -m uvicorn moonfish.rl.server.app:app --port 8000"
        )
        print("  2. Run: python examples/openenv_training.py --remote")
