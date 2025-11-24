import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FoosballEnv(gym.Env):
    def __init__(self, gui=False):
        super(FoosballEnv, self).__init__()

        # Connect to the physics engine
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load foosball table
        foosball_urdf_path = os.path.join(os.getcwd(), 'foosball', 'foosball.urdf')
        self.table_id = p.loadURDF(foosball_urdf_path, useFixedBase=1)

        # Change colors for better visualization
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.2, 0.6, 0.2, 1]) # Green Table
        team1_color = [1, 0, 0, 1]  # Red
        team2_color = [0, 0, 1, 1]  # Blue

        for i in range(p.getNumJoints(self.table_id)):
            joint_info = p.getJointInfo(self.table_id, i)
            link_name = joint_info[12].decode('UTF-8')

            if '_vir' not in link_name:
                if '1' in link_name:
                    p.changeVisualShape(self.table_id, i, rgbaColor=team1_color)
                elif '2' in link_name:
                    p.changeVisualShape(self.table_id, i, rgbaColor=team2_color)

        # Get table height
        table_aabb = p.getAABB(self.table_id, -1)
        self.table_height = table_aabb[1][2]

        # Load ball
        ball_urdf_path = os.path.join(os.getcwd(), 'foosball', 'ball.urdf')
        self.ball_id = p.loadURDF(ball_urdf_path, basePosition=[0, 0, self.table_height + 0.1])
        
        # Add tiny randomized initial velocity
        random_vel = np.random.uniform(-0.05, 0.05, 3)
        p.resetBaseVelocity(self.ball_id, linearVelocity=random_vel)
        
        # Get joint info
        self.revolute_joints = []
        self.prismatic_joints = []
        num_joints = p.getNumJoints(self.table_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.table_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.revolute_joints.append(i)
            elif joint_info[2] == p.JOINT_PRISMATIC:
                self.prismatic_joints.append(i)
        
        self.table_aabb = p.getAABB(self.table_id, -1)
        self.goal_width = 0.2
        self.goal_height = 0.4 # Adjusted goal height
        self.goal_half_width = self.goal_width / 2
        self.opponent_goal_x = self.table_aabb[1][0]
        self.own_goal_x = self.table_aabb[0][0]
        self.score = [0, 0]

        # Reset prismatic joints to their initial position (middle = 0.1)
        for joint_index in self.prismatic_joints:
            p.resetJointState(self.table_id, joint_index, targetValue=0.1)

        obs = self._get_obs()
        self.prev_ball_dist_to_goal = np.linalg.norm(obs[0:2] - [self.opponent_goal_x, 0])
        
        self.last_ball_pos = obs[0:3]
        
        self.ball_in_play = False
        
        return obs, {}

    def _get_obs(self):
        ball_pos, ball_orn = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, ball_ang_vel = p.getBaseVelocity(self.ball_id)
        
        joint_states = p.getJointStates(self.table_id, range(p.getNumJoints(self.table_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        return np.concatenate([ball_pos, ball_vel, joint_positions, joint_velocities]).astype(np.float32)

    def step(self, action):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        
        if not self.ball_in_play:
            if ball_pos[2] < self.table_height + 0.01:
                self.ball_in_play = True
        
        # Set control for revolute joints (rotation)
        for i, joint_index in enumerate(self.revolute_joints):
            rotation_vel = action[i] * 5 if self.ball_in_play else 0
            p.setJointMotorControl2(self.table_id, joint_index, p.VELOCITY_CONTROL, 
                                    targetVelocity=rotation_vel, force=500)
        
        # Set control for prismatic joints (translation)
        for i, joint_index in enumerate(self.prismatic_joints):
             p.setJointMotorControl2(self.table_id, joint_index, p.VELOCITY_CONTROL, 
                                     targetVelocity=action[i+8] * 2, force=500)

        p.stepSimulation()
        time.sleep(1./240.)
        
        observation = self._get_obs()
        ball_pos = observation[0:3]
        
        # Reward function
        ball_vel = observation[3:6]
        
        # 1. Reward for moving the ball towards the opponent's goal
        dist_to_goal = np.linalg.norm(ball_pos[0:2] - [self.opponent_goal_x, 0])
        reward_dist = self.prev_ball_dist_to_goal - dist_to_goal
        self.prev_ball_dist_to_goal = dist_to_goal
        
        # 2. Penalty for ball being near own goal
        dist_to_own_goal = np.linalg.norm(ball_pos[0:2] - [self.own_goal_x, 0])
        reward_near_own_goal = -1.0 / (dist_to_own_goal + 1e-6)

        # 3. Reward for ball being near opponent's goal
        reward_near_opponent_goal = 1.0 / (dist_to_goal + 1e-6)

        # 4. Reward for ball speed
        ball_speed_reward = np.linalg.norm(ball_vel)

        # 5. Control penalty
        reward_ctrl = -np.mean(np.square(action))

        # Total reward
        reward = (
            reward_dist + 
            0.01 * reward_near_own_goal + 
            0.01 * reward_near_opponent_goal +
            0.01 * ball_speed_reward + 
            0.001 * reward_ctrl
        )

        # Termination condition
        terminated = False
        
        if ball_pos[2] < 0.3: # Ball is below the table surface
            if ball_pos[0] > 0: # Opponent's side, simplified check
                reward = 100
                self.score[0] += 1
                print(f"Goal! Score: {self.score[0]} - {self.score[1]}")
            else: # Own side
                reward = -100
                self.score[1] += 1
                print(f"Own Goal! Score: {self.score[0]} - {self.score[1]}")
            terminated = True
        
        elif not (self.table_aabb[0][1] < ball_pos[1] < self.table_aabb[1][1]):
            reward = -10 # Ball out of bounds (sides)
            terminated = True


        # Stuck ball detection
        dist_moved = np.linalg.norm(ball_pos - self.last_ball_pos)
        self.last_ball_pos = ball_pos
        
        if dist_moved < 0.0001: # Check if ball moved less than 1mm
            reward -= 0.01 # Small penalty for each step the ball is stuck
        
        truncated = False
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.physics_client)

if __name__ == '__main__':
    env = FoosballEnv(gui=True)
    observation, info = env.reset()

    for t in range(10000):
        # All rods rotate with the same angular velocity
        # Oscillate translations with a sine wave
        rotation_vel = np.ones(8) * 5  # constant high-speed rotation
        translation_vel = np.ones(8) * (2 * np.sin(t * 0.01))  # Oscillating velocity
        action = np.concatenate([rotation_vel, translation_vel])
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()
