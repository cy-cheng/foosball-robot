# Foosball Robot RL Project

This project uses reinforcement learning to train a foosball-playing robot. The environment is simulated using PyBullet. The agent is trained using the PPO algorithm from Stable Baselines 3.

## Project Notes

### Goal Detection

The user has requested that the goal detection logic in `foosball_env.py` should not be changed. Modifying it might break the simulation.

### Reward Function

The reward function in `foosball_env.py` was modified to encourage more strategic gameplay. The following changes were made:

*   **Removed `ball_speed_reward`**: This reward was counterproductive as it encouraged random flailing to maximize ball velocity.
*   **Removed `reward_near_own_goal` and `reward_near_opponent_goal`**: These rewards could cause the agent to get stuck in local optima.
*   **Adjusted `reward_ctrl` weight**: The weight of the control penalty was increased to encourage smoother actions.

The new reward function is:

```python
# Total reward
reward = (
    reward_dist + 
    0.1 * reward_ctrl
)
```
