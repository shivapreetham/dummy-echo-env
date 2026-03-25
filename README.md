# Simple Echo Environment

A minimal OpenEnv environment for learning the basics. This environment echoes back messages you send to it.

## Description

This is the simplest possible OpenEnv implementation to understand the core concepts:
- How to define Action, Observation, and Reward models
- How to implement step(), reset(), and state() methods
- How to create tasks with graders
- How to structure an OpenEnv project

## Action Space

```python
class Action:
    message: str  # Any text message to echo
```

## Observation Space

```python
class Observation:
    echoed_message: str  # The echoed message
    step_count: int      # Current step number
```

## Tasks

1. **Task 1 (Easy)**: Send any message
   - Score: 1.0 if any message sent, 0.0 otherwise

2. **Task 2 (Medium)**: Send a message with at least 20 characters
   - Score: 1.0 if 20+ chars, proportional if less

3. **Task 3 (Hard)**: Send multiple messages totaling 100+ characters
   - Score: 1.0 if 100+ total chars, proportional if less

## Reward Function

Reward is based on message length: `min(message_length / 100, 1.0)`

## Setup Instructions

### Local Setup

```bash
cd simple-echo-env
pip install -r requirements.txt
python baseline.py
```

### Docker Setup

```bash
cd simple-echo-env
docker build -t simple-echo-env .
docker run simple-echo-env
```

## Baseline Scores

Running the baseline script produces:
- Task 1 (Easy): 1.0
- Task 2 (Medium): 1.0
- Task 3 (Hard): 1.0
- Average: 1.0

## Usage Example

```python
from environment import EchoEnv, Action

env = EchoEnv()
obs = env.reset()
print(obs.echoed_message)  # "Environment ready. Send me a message!"

action = Action(message="Hello, world!")
obs, reward, done, info = env.step(action)
print(obs.echoed_message)  # "Echo: Hello, world!"
print(reward.value)        # 0.13 (13 chars / 100)
```

## Key Learnings

This simple environment demonstrates:
1. Pydantic models for type safety
2. Standard OpenEnv interface (step/reset/state)
3. Episode management (max 10 steps)
4. Task graders that score agent performance
5. Reward shaping for agent learning
6. Docker containerization
