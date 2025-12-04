I need you to implement a Reinforcement Learning environment for Foosball using PyBullet. I have a foosball/foosball.urdf file and a ball file (and associated meshes) that provides the visual and collision geometry.

Context: The goal is to train a single RL agent (Team 1) to play against a hard-coded heuristic bot (Team 2) using Curriculum Learning. The solution must be computationally efficient to allow for finite time training.

Please implement the following Python code:

1. The Physics Setup

    URDF Fixes: Assume the provided foosball/foosball.urdf contains links but might be missing control joints. Write a utility function or script to programmatically treat the model as having 8 controllable rods (4 per team). Each rod needs 2 degrees of freedom: prismatic (slide) and revolute (spin).

    Joint Parsing: Automatically detect joints for "Team 1" (Agent) and "Team 2" (Opponent) based on naming conventions (e.g., names containing "1" are Team 1, "2" are Team 2).

    Optimization: In the reset() function, do not reload the URDFs. Load the table and plane once in __init__. In reset(), only reset the joint states and the ball position/velocity.

2. The Two-Agent Environment Logic

    Action Space: The environment step should accept an action vector of size 8 (4 rods * 2 DOFs) for the Agent (Team 1).

    Observation Space: Return a flat vector containing: Ball Position (XYZ), Ball Velocity (XYZ), and Joint Positions/Velocities for all 16 rods.

    The Opponent (Bot): Inside the step() function, calculate actions for Team 2 using a simple logic (e.g., if the ball is to the left, slide a bit left; constant forward rotation). Do not make the opponent purely random, or the agent won't learn defense. Apply these actions to Team 2's motors directly.

3. Curriculum Learning Implementation

create logic that Modify reset() to spawn the ball differently based on the level:

    Level 1 (Striker): Spawn the ball stationary immediately in front of the Agent's offensive rod.

    Level 2 (Interceptor): Spawn the ball at midfield with velocity rolling towards the Agent's rod.

    Level 3 (Goalie): Spawn the ball on the opponent's side shooting high-speed towards the Agent's goal.

    Level 4 (Full Game): Random spawn position and velocity within the table bounds.

4. robust Scoring & Termination Logic

    Goal Detection: Instead of generic x-coordinates, use p.getAABB(table_id) to dynamically determine the goal lines.

    Reward Shaping:

        Dense: Reward velocity_towards_opponent_goal.

        Sparse: +100 for Goal, -50 for Own Goal.

    Stuck Ball Handling: Implement a counter. If the ball velocity is near zero for N steps, truncate the episode (return truncated=True) and apply a small negative reward to discourage stalling.

5. Collision Reward

    The collision reward is working correctly. The logic in `_parse_joints_and_links` correctly identifies the player links, and `_compute_reward` correctly uses `p.getContactPoints` to check for collisions with these links.

Output Requirements: Provide the complete, runnable code for foosball_env.py. Ensure the class inherits from gym.Env and handles the PyBullet connection logic (GUI vs. DIRECT) efficiently.