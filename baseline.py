import os
from environment import EchoEnv, Action, grader_task1_short_message, grader_task2_medium_message, grader_task3_long_message


def run_simple_baseline():
    """Run a simple baseline without AI - just predefined actions."""
    env = EchoEnv()

    # Test messages
    test_messages = [
        "Hi",
        "This is a medium length message for testing",
        "Another message",
        "More text to accumulate characters",
        "Final message"
    ]

    # Task 1: Send any message
    print("\n=== Task 1: Send any message ===")
    env.reset()
    episode_history = []

    action = Action(message=test_messages[0])
    obs, reward, done, info = env.step(action)
    episode_history.append({"action": action, "observation": obs, "reward": reward})

    score1 = grader_task1_short_message(env, episode_history)
    print(f"Task 1 Score: {score1}")

    # Task 2: Send a medium message
    print("\n=== Task 2: Send a medium message ===")
    env.reset()
    episode_history = []

    action = Action(message=test_messages[1])
    obs, reward, done, info = env.step(action)
    episode_history.append({"action": action, "observation": obs, "reward": reward})

    score2 = grader_task2_medium_message(env, episode_history)
    print(f"Task 2 Score: {score2}")

    # Task 3: Send multiple messages
    print("\n=== Task 3: Send multiple messages ===")
    env.reset()
    episode_history = []

    for msg in test_messages:
        action = Action(message=msg)
        obs, reward, done, info = env.step(action)
        episode_history.append({"action": action, "observation": obs, "reward": reward})
        print(f"  Step {obs.step_count}: '{msg}' (reward: {reward.value:.2f})")
        if done:
            break

    score3 = grader_task3_long_message(env, episode_history)
    print(f"Task 3 Score: {score3}")

    # Summary
    print("\n=== Baseline Scores ===")
    print(f"Task 1 (Easy): {score1}")
    print(f"Task 2 (Medium): {score2}")
    print(f"Task 3 (Hard): {score3}")
    print(f"Average: {(score1 + score2 + score3) / 3:.2f}")

    return {
        "task1": score1,
        "task2": score2,
        "task3": score3,
        "average": (score1 + score2 + score3) / 3
    }


if __name__ == "__main__":
    run_simple_baseline()
