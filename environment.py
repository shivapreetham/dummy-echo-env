from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class Action(BaseModel):
    message: str = Field(..., description="Message to echo back")


class Observation(BaseModel):
    echoed_message: str = Field(..., description="The echoed message")
    step_count: int = Field(..., description="Number of steps taken")


class Reward(BaseModel):
    value: float = Field(..., description="Reward value between 0 and 1")
    reason: str = Field(..., description="Explanation for the reward")


class EchoEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 10
        self.current_message = ""

    def reset(self) -> Observation:
        """Reset the environment to initial state."""
        self.step_count = 0
        self.current_message = ""
        return Observation(
            echoed_message="Environment ready. Send me a message!",
            step_count=0
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.step_count += 1
        self.current_message = action.message

        # Simple reward: longer messages get higher rewards
        message_length = len(action.message)
        reward_value = min(message_length / 100.0, 1.0)

        observation = Observation(
            echoed_message=f"Echo: {action.message}",
            step_count=self.step_count
        )

        reward = Reward(
            value=reward_value,
            reason=f"Message length: {message_length} characters"
        )

        done = self.step_count >= self.max_steps

        info = {
            "message_length": message_length,
            "steps_remaining": self.max_steps - self.step_count
        }

        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return current state of the environment."""
        return {
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_message": self.current_message
        }


# Task graders
def grader_task1_short_message(env: EchoEnv, episode_history: list) -> float:
    """Easy task: Send any message."""
    if len(episode_history) > 0:
        last_action = episode_history[-1]["action"]
        if len(last_action.message) > 0:
            return 1.0
    return 0.0


def grader_task2_medium_message(env: EchoEnv, episode_history: list) -> float:
    """Medium task: Send a message with at least 20 characters."""
    if len(episode_history) > 0:
        last_action = episode_history[-1]["action"]
        if len(last_action.message) >= 20:
            return 1.0
        elif len(last_action.message) > 0:
            return len(last_action.message) / 20.0
    return 0.0


def grader_task3_long_message(env: EchoEnv, episode_history: list) -> float:
    """Hard task: Send multiple messages totaling 100+ characters."""
    total_chars = sum(len(step["action"].message) for step in episode_history)
    if total_chars >= 100:
        return 1.0
    return total_chars / 100.0
