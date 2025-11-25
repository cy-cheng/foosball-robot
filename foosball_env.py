import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os


class FoosballEnv(gym.Env):
    """
    Two-agent symmetric foosball environment for RL training.
    
    Uses symmetric observations/actions: Team 2 sees negated X coordinates.
    This allows training a single policy and mirroring it for both teams.
    
    player_id: 1 or 2 (which team this env controls; other team is controlled externally)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode='human', curriculum_level=1, debug_mode=False, player_id=1):
        super(FoosballEnv, self).__init__()

        self.render_mode = render_mode
        self.curriculum_level = curriculum_level
        self.debug_mode = debug_mode
        self.player_id = player_id
        self.goals_this_level = 0

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        else:
            self.client = p.connect(p.DIRECT)

        # Action space: 4 rods, each with 2 DOF (slide + rotate)
        # Slide: [-1, 1] normalized to joint limits
        # Rotate: [-1, 1] normalized to [-pi, pi]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # Observation space: ball_pos(3), ball_vel(3), all_joint_pos(16), all_joint_vel(16)
        obs_space_dim = 3 + 3 + 16 + 16
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32)

        self.ball_stuck_counter = 0
        self.max_stuck_steps = 1500
        self.episode_step_count = 0
        self.max_episode_steps = 10000
        
        # Load models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSubSteps=4)
        
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        
        urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)
        
        # Color table
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1])

        self._setup_camera()
        
        # Parse joints
        self.team1_slide_joints = []
        self.team1_rev_joints = []
        self.team2_slide_joints = []
        self.team2_rev_joints = []
        self.joint_name_to_id = {}
        self.joint_limits = {}
        self.goal_link_a = None
        self.goal_link_b = None

        self._parse_joints()
        
        # Create ball as simple sphere
        ball_radius = 0.025
        ball_mass = 0.025
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1])
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        
        start_pos = [0, 0, 0.55]
        self.ball_id = p.createMultiBody(
            baseMass=ball_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        p.changeDynamics(self.ball_id, -1, restitution=0.8, rollingFriction=0.001, 
                        spinningFriction=0.001, lateralFriction=0.01)
        
        # Apply dynamics to rods
        for joint_id in self.team1_slide_joints + self.team1_rev_joints + self.team2_slide_joints + self.team2_rev_joints:
            p.changeDynamics(self.table_id, joint_id, linearDamping=0.5, angularDamping=0.5, restitution=0.7)
        
        self.team1_joints = self.team1_slide_joints + self.team1_rev_joints
        self.team2_joints = self.team2_slide_joints + self.team2_rev_joints
        self.all_joints = self.team1_joints + self.team2_joints
        
        self.max_vel = 1.0
        self.max_force = 0.001
        
        # Goal line detection
        self.goal_line_x_1 = -0.59  # Team 1 goal (left side)
        self.goal_line_x_2 = 0.59   # Team 2 goal (right side)
        
        self.debug_step_counter = 0
        if self.debug_mode and self.render_mode == 'human':
            self._add_debug_sliders()

    def _parse_joints(self):
        """Parse URDF joints and organize by team (rods 1-4 are Team 1, 5-8 are Team 2)"""
        num_joints = p.getNumJoints(self.table_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(self.table_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            self.joint_name_to_id[joint_name] = i
            self.joint_limits[i] = (info[8], info[9])
            
            # Detect goal sensors
            if "goal_sensor_A" in joint_name:
                self.goal_link_a = i
            if "goal_sensor_B" in joint_name:
                self.goal_link_b = i
            
            # Only control PRISMATIC and REVOLUTE joints
            if joint_type not in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
                continue
            
            # Parse rod number from joint name (rod_1 through rod_8)
            rod_num = None
            for rod_idx in range(1, 9):
                if f"rod_{rod_idx}" in joint_name:
                    rod_num = rod_idx
                    break
            
            if rod_num is None:
                continue
            
            # Team 1: rods 1-4, Team 2: rods 5-8
            team = 1 if rod_num <= 4 else 2
            
            # Color code by team
            if team == 1:
                p.changeVisualShape(self.table_id, i, rgbaColor=[0, 0, 1, 1])
            else:
                p.changeVisualShape(self.table_id, i, rgbaColor=[1, 0, 0, 1])
            
            # Categorize by joint type
            if 'slide' in joint_name.lower():
                if team == 1:
                    self.team1_slide_joints.append(i)
                else:
                    self.team2_slide_joints.append(i)
            elif 'rotate' in joint_name.lower():
                if team == 1:
                    self.team1_rev_joints.append(i)
                else:
                    self.team2_rev_joints.append(i)

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0.5]
        )
    
    def _add_debug_sliders(self):
        """Add sliders for manual debugging"""
        self.slider_ids = {}
        
        for i, joint_id in enumerate(self.team1_rev_joints):
            name = f"T1_Rod{i+1}_Rot"
            self.slider_ids[joint_id] = p.addUserDebugParameter(name, -np.pi, np.pi, 0)
        
        for i, joint_id in enumerate(self.team1_slide_joints):
            lower, upper = self.joint_limits[joint_id]
            name = f"T1_Rod{i+1}_Slide"
            self.slider_ids[joint_id] = p.addUserDebugParameter(name, lower, upper, (lower + upper) / 2)

    def _curriculum_spawn_ball(self):
        """Spawn ball according to curriculum level"""
        if self.curriculum_level == 1:
            # Dribble: stationary in front of own offensive rod
            if self.player_id == 1:
                ball_pos = [-0.2, 0, 0.55]
            else:
                ball_pos = [0.2, 0, 0.55]
            ball_vel = [0, 0, 0]
        elif self.curriculum_level == 2:
            # Pass: ball at midfield rolling toward own rods
            ball_pos = [0, 0, 0.55]
            if self.player_id == 1:
                ball_vel = [-1, np.random.uniform(-0.5, 0.5), 0]
            else:
                ball_vel = [1, np.random.uniform(-0.5, 0.5), 0]
        elif self.curriculum_level == 3:
            # Defend: ball on opponent side shooting at goal
            if self.player_id == 1:
                ball_pos = [0.4, np.random.uniform(-0.2, 0.2), 0.55]
                ball_vel = [-5, np.random.uniform(-1, 1), 0]
            else:
                ball_pos = [-0.4, np.random.uniform(-0.2, 0.2), 0.55]
                ball_vel = [5, np.random.uniform(-1, 1), 0]
        else:  # level 4: Full Game
            ball_pos = [np.random.uniform(-0.6, 0.6), np.random.uniform(-0.3, 0.3), 0.55]
            ball_vel = [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0]
        
        p.resetBasePositionAndOrientation(self.ball_id, ball_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball_id, linearVelocity=ball_vel)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset all joints to home position
        for joint_id in self.all_joints:
            p.resetJointState(self.table_id, joint_id, targetValue=0, targetVelocity=0)
        
        # Spawn ball according to curriculum
        self._curriculum_spawn_ball()
        
        self.ball_stuck_counter = 0
        self.episode_step_count = 0
        self.goals_this_level = 0
        
        return self._get_obs(), {}

    def step(self, action, opponent_action=None):
        """
        Step the environment.
        
        action: 8-dim vector for this player's rods (4 slides + 4 rotates)
        opponent_action: 8-dim vector for opponent (if None, uses zero/constant action)
        """
        self.episode_step_count += 1
        
        if self.debug_mode and self.render_mode == 'human':
            self._debug_step(action)
        else:
            # Apply agent action
            scaled_action = self._scale_action(action, self.player_id)
            self._apply_action(scaled_action, self.player_id)
            
            # Apply opponent action (default to simple bot logic if None)
            if opponent_action is None:
                ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
                # Mirror ball position for opponent
                if self.player_id == 1:
                    mirrored_ball_pos = [-ball_pos[0], ball_pos[1], ball_pos[2]]
                else:
                    mirrored_ball_pos = [-ball_pos[0], ball_pos[1], ball_pos[2]]
                opponent_action = self._simple_bot_logic(mirrored_ball_pos)
            
            scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
            self._apply_action(scaled_opponent_action, 3 - self.player_id)
        
        p.stepSimulation()
        
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._check_termination(obs)
        
        return obs, reward, terminated, truncated, {}

    def _scale_action(self, action, player_id):
        """Scale action from [-1, 1] to actual joint ranges"""
        scaled = np.zeros(8)
        
        # Player ID determines which joints to scale
        if player_id == 1:
            slides = self.team1_slide_joints
            revs = self.team1_rev_joints
        else:
            slides = self.team2_slide_joints
            revs = self.team2_rev_joints
        
        # Scale slides
        for i, joint_id in enumerate(slides):
            lower, upper = self.joint_limits[joint_id]
            scaled[i] = lower + (action[i] + 1) / 2 * (upper - lower)
        
        # Scale revolutes
        for i, joint_id in enumerate(revs):
            scaled[i + 4] = action[i + 4] * np.pi
        
        return scaled

    def _apply_action(self, scaled_action, player_id):
        """Apply scaled action to joints"""
        if player_id == 1:
            slides = self.team1_slide_joints
            revs = self.team1_rev_joints
        else:
            slides = self.team2_slide_joints
            revs = self.team2_rev_joints
        
        for i, joint_id in enumerate(slides):
            p.setJointMotorControl2(
                self.table_id, joint_id, p.POSITION_CONTROL,
                targetPosition=scaled_action[i],
                maxVelocity=self.max_vel,
                force=self.max_force
            )
        
        for i, joint_id in enumerate(revs):
            p.setJointMotorControl2(
                self.table_id, joint_id, p.POSITION_CONTROL,
                targetPosition=scaled_action[i + 4],
                maxVelocity=self.max_vel,
                force=self.max_force
            )

    def _debug_step(self, action):
        """Debug step with manual slider control"""
        for joint_id in self.team1_rev_joints:
            if joint_id in self.slider_ids:
                target_pos = p.readUserDebugParameter(self.slider_ids[joint_id])
                p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL,
                                       targetPosition=target_pos, maxVelocity=self.max_vel, force=self.max_force)
        
        for joint_id in self.team1_slide_joints:
            if joint_id in self.slider_ids:
                target_pos = p.readUserDebugParameter(self.slider_ids[joint_id])
                p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL,
                                       targetPosition=target_pos, maxVelocity=self.max_vel, force=self.max_force)

    def _simple_bot_logic(self, ball_pos):
        """Simple opponent bot logic (mirrored observation)"""
        action = np.zeros(8)
        
        # Slide towards ball y-position (clamped)
        for i in range(4):
            action[i] = np.clip(ball_pos[1] / 0.3, -1, 1)
        
        # Constant rotation
        for i in range(4):
            action[i + 4] = 0.3
        
        return action

    def _get_obs(self):
        """
        Get observation: [ball_pos(3), ball_vel(3), joint_pos(16), joint_vel(16)]
        
        For player_id=2, mirror X coordinates for symmetry.
        """
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        
        joint_states = p.getJointStates(self.table_id, self.all_joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        
        # For player 2, mirror the observation for symmetry
        if self.player_id == 2:
            ball_pos = (-ball_pos[0], ball_pos[1], ball_pos[2])
            ball_vel = (-ball_vel[0], ball_vel[1], ball_vel[2])
        
        obs = np.concatenate([ball_pos, ball_vel, joint_pos, joint_vel]).astype(np.float32)
        return obs

    def _compute_reward(self):
        """
        Compute reward with dense shaping + sparse goals.
        
        Dense:
        - Ball velocity towards opponent goal: +0.1
        - Distance to opponent goal: -0.01
        - Rod extension bonus: +0.1 per step if extended
        
        Sparse:
        - Goal: +100
        - Conceded goal: -50
        """
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        
        reward = 0
        
        # Dense reward 1: velocity towards opponent
        if self.player_id == 1:
            goal_x = self.goal_line_x_2
            reward += ball_vel[0] * 0.1  # towards positive x
        else:
            goal_x = self.goal_line_x_1
            reward += -ball_vel[0] * 0.1  # towards negative x
        
        # Dense reward 2: distance to opponent goal (negative, incentivizes moving ball forward)
        dist_to_goal = abs(ball_pos[0] - goal_x)
        reward -= 0.01 * dist_to_goal
        
        # Dense reward 3: rod extension (encourages active play)
        team_slides = self.team1_slide_joints if self.player_id == 1 else self.team2_slide_joints
        joint_states = p.getJointStates(self.table_id, team_slides)
        avg_extension = np.mean([abs(state[0]) for state in joint_states])
        reward += avg_extension * 0.1
        
        # Sparse reward: goals
        # Goal for this player (ball crosses opponent goal line)
        if self.player_id == 1 and ball_pos[0] > self.goal_line_x_2:
            reward += 100
            self.goals_this_level += 1
        elif self.player_id == 2 and ball_pos[0] < self.goal_line_x_1:
            reward += 100
            self.goals_this_level += 1
        
        # Own goal
        if self.player_id == 1 and ball_pos[0] < self.goal_line_x_1:
            reward -= 50
        elif self.player_id == 2 and ball_pos[0] > self.goal_line_x_2:
            reward -= 50
        
        return reward

    def _check_termination(self, obs):
        """Check for episode termination or truncation"""
        ball_pos = obs[:3]
        ball_vel = obs[3:6]
        
        terminated = False
        truncated = False
        
        # Terminal: goal scored
        if self.player_id == 1:
            if ball_pos[0] > self.goal_line_x_2 or ball_pos[0] < self.goal_line_x_1:
                terminated = True
        else:
            if ball_pos[0] < self.goal_line_x_1 or ball_pos[0] > self.goal_line_x_2:
                terminated = True
        
        # Truncate: ball out of bounds
        table_aabb = p.getAABB(self.table_id)
        if not (table_aabb[0][0] - 0.1 < ball_pos[0] < table_aabb[1][0] + 0.1 and \
                table_aabb[0][1] - 0.1 < ball_pos[1] < table_aabb[1][1] + 0.1):
            truncated = True
        
        # Truncate: ball stuck
        if np.linalg.norm(ball_vel) < 0.001:
            self.ball_stuck_counter += 1
        else:
            self.ball_stuck_counter = 0
        
        if self.ball_stuck_counter > self.max_stuck_steps:
            truncated = True
        
        # Truncate: max steps
        if self.episode_step_count >= self.max_episode_steps:
            truncated = True
        
        return terminated, truncated

    def close(self):
        p.disconnect(self.client)


if __name__ == '__main__':
    env = FoosballEnv(render_mode='human', curriculum_level=1, debug_mode=False)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
