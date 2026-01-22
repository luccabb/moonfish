"""
Basic usage example for the Chess OpenEnv environment.

This example shows how to use the chess environment both locally
(without a server) and via the HTTP client.
"""

import random

from moonfish.rl import ChessAction, ChessEnvironment, RewardConfig


def play_random_game():
    """Play a game with random moves to demonstrate the environment."""
    print("=== Playing a random game ===\n")

    # Create environment
    env = ChessEnvironment()

    # Reset to start a new game
    obs = env.reset(episode_id="random_game_001")

    print(f"Initial position: {obs.fen}")
    print(f"Legal moves: {len(obs.legal_moves)} available")
    print()

    move_count = 0
    total_reward = 0.0

    while not obs.done:
        # Pick a random legal move
        move = random.choice(obs.legal_moves)
        action = ChessAction(move=move)

        # Execute the move
        obs, reward, done = env.step(action)
        total_reward += reward
        move_count += 1

        if move_count <= 5 or done:
            print(f"Move {move_count}: {move}")
            print(f"  FEN: {obs.fen}")
            print(f"  Check: {obs.is_check}, Reward: {reward}")
            if move_count == 5 and not done:
                print("  ... (continuing)")
                print()

    print(f"\nGame finished after {move_count} moves")
    print(f"Result: {obs.result}")
    print(f"Total reward: {total_reward}")

    # Check final state
    state = env.state
    print(f"Episode ID: {state.episode_id}")
    print(f"Move history: {state.move_history[:10]}...")

    env.close()


def play_specific_opening():
    """Demonstrate playing specific moves (Italian Game opening)."""
    print("\n=== Playing the Italian Game opening ===\n")

    env = ChessEnvironment()
    obs = env.reset()

    opening_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]

    for i, move in enumerate(opening_moves):
        action = ChessAction(move=move)
        obs, reward, done = env.step(action)
        print(f"{i+1}. {move} -> Check: {obs.is_check}")

    print(f"\nPosition after opening: {obs.fen}")
    print(f"Legal moves for Black: {len(obs.legal_moves)}")
    print(f"Material: {obs.metadata.get('material', {})}")

    env.close()


def demonstrate_illegal_move():
    """Show how illegal moves are handled."""
    print("\n=== Handling illegal moves ===\n")

    env = ChessEnvironment()
    obs = env.reset()

    # Try an illegal move
    illegal_action = ChessAction(move="e2e5")  # Can't move pawn 3 squares
    obs, reward, done = env.step(illegal_action)

    print("Attempted illegal move: e2e5")
    print(f"Reward: {reward}")  # Should be negative
    print(f"Error: {obs.metadata.get('error', 'None')}")
    print(f"Done: {done}")  # Game continues

    env.close()


def with_evaluation_rewards():
    """Show evaluation-based intermediate rewards."""
    print("\n=== Evaluation-based rewards ===\n")

    config = RewardConfig(
        use_evaluation=True,
        evaluation_scale=0.0001,  # Scale down the centipawn values
    )

    env = ChessEnvironment(reward_config=config)
    obs = env.reset()

    # Play a few moves and observe evaluation changes
    moves = ["e2e4", "d7d5", "e4d5"]  # White wins a pawn

    for move in moves:
        action = ChessAction(move=move)
        obs, reward, done = env.step(action)
        eval_score = obs.metadata.get("evaluation", 0)
        print(f"Move: {move}, Reward: {reward:.4f}, Eval: {eval_score:.1f}")

    env.close()


if __name__ == "__main__":
    play_random_game()
    play_specific_opening()
    demonstrate_illegal_move()
    with_evaluation_rewards()
