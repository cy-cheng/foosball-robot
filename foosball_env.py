import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time

from foosball_utils import get_config_value


class FoosballEnv(gym.Env):
    """
    Two-agent symmetric foosball environment for RL training.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config, render_mode='human', curriculum_level=1, debug_mode=False, player_id=1, opponent_model=None, goal_debug_mode=False):
        super(FoosballEnv, self).__init__()

        self.config = config # Store the full config
        self.env_config = config['environment']
        self.reward_config = config['reward']
        self.curriculum_config_all_stages = config['curriculum'] # Renamed to avoid confusion with current stage
        self.physics_config = config['physics']

        self.render_mode = render_mode
        self.goal_debug_mode = goal_debug_mode
        if self.goal_debug_mode:
            self.render_mode = 'human'

        self.curriculum_level = curriculum_level
        self.debug_mode = debug_mode
        self.player_id = player_id
        self.opponent_model = opponent_model
        self.goals_this_level = 0

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        else:
            self.client = p.connect(p.DIRECT)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        obs_space_dim = 3 + 3 + 16 + 16 # Ball pos (3), Ball vel (3), Joint pos (16), Joint vel (16)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32)

        self.ball_stuck_counter = 0
        self.max_stuck_steps = self.env_config['max_stuck_steps'] # From config
        self.episode_step_count = 0
        # self.max_episode_steps will be retrieved dynamically per stage in reset()
        self.previous_action = np.zeros(self.action_space.shape)
        self.previous_ball_dist = 0
        # For new stagnation penalty
        self.last_slide_positions = np.zeros(4)
        self.slide_stagnation_counter = 0
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.physics_config['gravity'], physicsClientId=self.client) # From config, set gravX, gravY to 0
        p.setPhysicsEngineParameter(numSubSteps=self.physics_config['physics_substeps'], # From config
                                    physicsClientId=self.client,
                                    fixedTimeStep=self.physics_config['simulation_timestep']) # From config
        
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0], physicsClientId=self.client)
        
        urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True, physicsClientId=self.client)
        
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1], physicsClientId=self.client)

        self._setup_camera()
        
        self.team1_slide_joints = []
        self.team1_rev_joints = []
        self.team2_slide_joints = []
        self.team2_rev_joints = []
        self.team1_player_links = []
        self.team2_player_links = []
        self.team1_players_by_rod = {}
        self.team2_players_by_rod = {}
        self.joint_name_to_id = {}
        self.joint_limits = {}
        self.goal_link_a = None
        self.goal_link_b = None

        self._parse_joints_and_links()
        
        ball_radius = self.env_config['ball_radius'] # From config
        ball_mass = self.env_config['ball_mass'] # From config
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius, physicsClientId=self.client)
        self.ball_id = p.createMultiBody(baseMass=ball_mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 0.55], physicsClientId=self.client)
        p.changeDynamics(self.ball_id, -1, 
                         restitution=self.env_config['ball_restitution'], # From config
                         rollingFriction=self.env_config['ball_rolling_friction'], # From config
                         spinningFriction=self.env_config['ball_spinning_friction'], # From config
                         lateralFriction=self.env_config['ball_lateral_friction'], # From config
                         physicsClientId=self.client)
        
        for joint_id in self.team1_slide_joints + self.team1_rev_joints + self.team2_slide_joints + self.team2_rev_joints:
            p.changeDynamics(self.table_id, joint_id, 
                             linearDamping=self.env_config['rod_linear_damping'], # From config
                             angularDamping=self.env_config['rod_angular_damping'], # From config
                             restitution=self.env_config['rod_restitution'], # From config
                             physicsClientId=self.client)
        
        self.team1_joints = self.team1_slide_joints + self.team1_rev_joints
        self.team2_joints = self.team2_slide_joints + self.team2_rev_joints
        self.all_joints = self.team1_joints + self.team2_joints
        
        self.max_vel = self.env_config['max_velocity'] # From config
        self.max_force = self.env_config['max_force'] # From config
        self.max_torque = self.env_config.get('max_torque', 1.0) # From config, with default
        
        self.goal_line_x_1 = self.env_config['goal_line_x_1'] # From config
        self.goal_line_x_2 = self.env_config['goal_line_x_2'] # From config
        
        if self.goal_debug_mode:
            self.goal_line_slider_1 = p.addUserDebugParameter("Goal Line 1", -1.0, 1.0, self.goal_line_x_1, physicsClientId=self.client)
            self.goal_line_slider_2 = p.addUserDebugParameter("Goal Line 2", -1.0, 1.0, self.goal_line_x_2, physicsClientId=self.client)

        if self.debug_mode and self.render_mode == 'human':
            self._add_debug_sliders()

    def update_opponent_model(self, state_dict):
        # In Stage 4, the env is initialized with opponent_model=None.
        # We need to create a placeholder model here so we can load the state dict.
        if self.opponent_model is None and self.curriculum_level == 4:
            from stable_baselines3 import PPO  # Local import to avoid circular dependency
            
            # Create a shell PPO model. The policy and parameters don't matter much
            # as they will be immediately overwritten by the state_dict.
            self.opponent_model = PPO("MlpPolicy", self, verbose=0)
            self.opponent_model._setup_model()

        if self.opponent_model:
            self.opponent_model.policy.load_state_dict(state_dict)

    def _parse_joints_and_links(self):
        num_joints = p.getNumJoints(self.table_id)
        team1_slide_joints_map, team1_rev_joints_map = {}, {}
        team2_slide_joints_map, team2_rev_joints_map = {}, {}
        link_name_to_index = {p.getBodyInfo(self.table_id)[0].decode('UTF-8'): -1}
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.table_id, i)
            link_name = joint_info[12].decode('UTF-8')
            link_name_to_index[link_name] = i

        for link_name, link_index in link_name_to_index.items():
            if link_name.startswith('p'): # This identifies player links like 'p1_1', 'p2_1', etc.
                try:
                    rod_num = int(link_name.split('_')[0][1:])
                    team = 1 if rod_num in [1, 2, 4, 6] else 2
                    if team == 1:
                        self.team1_player_links.append(link_index)
                        if rod_num not in self.team1_players_by_rod:
                            self.team1_players_by_rod[rod_num] = []
                        self.team1_players_by_rod[rod_num].append(link_index)
                    else:
                        self.team2_player_links.append(link_index)
                        if rod_num not in self.team2_players_by_rod:
                            self.team2_players_by_rod[rod_num] = []
                        self.team2_players_by_rod[rod_num].append(link_index)
                except (ValueError, IndexError):
                    continue # Not a player link

        for i in range(num_joints):
            info = p.getJointInfo(self.table_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            self.joint_name_to_id[joint_name] = i
            self.joint_limits[i] = (info[8], info[9])

            if "goal_sensor_A" in joint_name: self.goal_link_a = i
            if "goal_sensor_B" in joint_name: self.goal_link_b = i
            if joint_type not in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]: continue
            
            rod_num_str = ''.join(filter(str.isdigit, joint_name))
            if not rod_num_str: continue
            rod_num = int(rod_num_str)

            if f"rod_{rod_num}" not in joint_name: continue

            team = 1 if rod_num in [1, 2, 4, 6] else 2
            
            if 'slide' in joint_name.lower():
                if team == 1: team1_slide_joints_map[rod_num] = i
                else: team2_slide_joints_map[rod_num] = i
            elif 'rotate' in joint_name.lower():
                if team == 1: team1_rev_joints_map[rod_num] = i
                else: team2_rev_joints_map[rod_num] = i
        
        self.team1_slide_joints = [team1_slide_joints_map[k] for k in sorted(team1_slide_joints_map)]
        self.team1_rev_joints = [team1_rev_joints_map[k] for k in sorted(team1_rev_joints_map)]
        self.team2_slide_joints = [team2_slide_joints_map[k] for k in sorted(team2_slide_joints_map)]
        self.team2_rev_joints = [team2_rev_joints_map[k] for k in sorted(team2_rev_joints_map)]

        # Disable the default motors for revolute joints to allow for torque control
        all_rev_joints = self.team1_rev_joints + self.team2_rev_joints
        for joint_id in all_rev_joints:
            p.setJointMotorControl2(self.table_id, joint_id, p.VELOCITY_CONTROL, force=0, physicsClientId=self.client)


    def _setup_camera(self):
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0.5])

    def _add_debug_sliders(self):
        self.slider_ids = {}
        for i, joint_id in enumerate(self.team1_rev_joints):
            self.slider_ids[joint_id] = p.addUserDebugParameter(f"T1_Rod{i+1}_Rot", -np.pi, np.pi, 0)
        for i, joint_id in enumerate(self.team1_slide_joints):
            lower, upper = self.joint_limits[joint_id]
            self.slider_ids[joint_id] = p.addUserDebugParameter(f"T1_Rod{i+1}_Slide", lower, upper, (lower + upper) / 2)


    def _curriculum_spawn_ball(self):
        curriculum_stage_config = self.curriculum_config_all_stages[f'stage_{self.curriculum_level}']
        
        # Determine the source for X-spawn: either 'ball_spawn_x_range' or 'ball_spawn_x'
        # Both are now expected to be lists of [min, max] intervals
        x_spawn_config_key = 'ball_spawn_x_range'
        if self.curriculum_level == 2: # Stage 2 specifically uses 'ball_spawn_x'
            x_spawn_config_key = 'ball_spawn_x'
        
        # Retrieve the list of intervals, with a safe fallback
        ball_spawn_x_ranges_list = curriculum_stage_config.get(x_spawn_config_key, [[-0.6, 0.6]])
        
        # Ensure it's a list of lists, even if the config only provides a single [min, max] list directly
        # This check might be redundant if the config is always well-formed as a list of lists
        if ball_spawn_x_ranges_list and not isinstance(ball_spawn_x_ranges_list[0], list):
            ball_spawn_x_ranges_list = [ball_spawn_x_ranges_list] 

        # Randomly pick one [min, max] interval from the list
        selected_x_range = ball_spawn_x_ranges_list[np.random.randint(len(ball_spawn_x_ranges_list))]
        ball_x = np.random.uniform(selected_x_range[0], selected_x_range[1])
        
        # Common defaults for other spawn parameters
        ball_spawn_y_range = curriculum_stage_config.get('ball_spawn_y_range', [-0.3, 0.3])
        ball_velocity_range = curriculum_stage_config.get('ball_velocity_range', [-1.0, 1.0])
        ball_velocity_speed_range = curriculum_stage_config.get('ball_velocity_speed_range', [3.0, 4.5])

        if self.curriculum_level == 1:
            # ball_x is already determined above
            ball_y = np.random.uniform(ball_spawn_y_range[0], ball_spawn_y_range[1])
            ball_vel = [np.random.uniform(ball_velocity_range[0], ball_velocity_range[1]), np.random.uniform(-0.1, 0.1), 0]
            ball_pos = [ball_x, ball_y, 0.55]
        elif self.curriculum_level == 2:
            ball_pos = [ball_x, np.random.uniform(ball_spawn_y_range[0], ball_spawn_y_range[1]), 0.55]
            ball_vel = [ball_velocity_range[0], np.random.uniform(-0.5, 0.5), 0] # Always spawn towards Team 1's goal (negative x)
        elif self.curriculum_level == 3:
            speed = np.random.uniform(ball_velocity_speed_range[0], ball_velocity_speed_range[1])
            # ball_x is already determined above
            spawn_pos = np.array([ball_x, np.random.uniform(ball_spawn_y_range[0], ball_spawn_y_range[1]), 0.55])
            
            # Target agent's goal line
            target_goal_line_x = self.goal_line_x_1 if self.player_id == 1 else self.goal_line_x_2
            target_pos = np.array([target_goal_line_x, np.random.uniform(-0.05, 0.05), 0.55])
            
            direction = target_pos - spawn_pos
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0: direction_norm = 1
            ball_vel = (direction / direction_norm) * speed
            ball_pos = spawn_pos.tolist()
        else: # Level 4 (Full Game)
            # ball_x is already determined above
            ball_y = np.random.uniform(ball_spawn_y_range[0], ball_spawn_y_range[1])
            ball_pos = [ball_x, ball_y, 0.55]
            ball_vel = [np.random.uniform(ball_velocity_range[0], ball_velocity_range[1]), np.random.uniform(ball_velocity_range[0], ball_velocity_range[1]), 0]
        
        p.resetBasePositionAndOrientation(self.ball_id, ball_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.ball_id, linearVelocity=ball_vel, physicsClientId=self.client)

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        for joint_id in self.all_joints: p.resetJointState(self.table_id, joint_id, targetValue=0, targetVelocity=0, physicsClientId=self.client)
        self._curriculum_spawn_ball()
        if self.curriculum_level == 1:
            for _ in range(100): # More steps to ensure rods are settled
                self._set_opponent_rods_to_90_degrees()
                p.stepSimulation(physicsClientId=self.client)
        self.ball_stuck_counter, self.episode_step_count, self.goals_this_level = 0, 0, 0
        # For new stagnation penalty
        self.last_slide_positions = np.zeros(4)
        self.slide_stagnation_counter = 0
        
        # Set max_episode_steps from curriculum config for the current stage
        curriculum_stage_config = self.curriculum_config_all_stages[f'stage_{self.curriculum_level}']
        self.max_episode_steps = curriculum_stage_config.get('max_episode_steps', 2000) # Default to 2000 if not specified
        
        return self._get_obs(), {}

    def step(self, action, opponent_action=None):
        self.episode_step_count += 1
        scaled_action = self._scale_action(action, self.player_id)
        self._apply_action(scaled_action, self.player_id)

        if self.curriculum_level == 1:
            self._set_opponent_rods_to_90_degrees()
        elif self.curriculum_level == 4 and self.opponent_model:
            mirrored_obs = self._get_mirrored_obs()
            opponent_action, _ = self.opponent_model.predict(mirrored_obs, deterministic=False)
            opponent_action *= -1 # Mirror the entire action
            scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
            self._apply_action(scaled_opponent_action, 3 - self.player_id)
        elif opponent_action is None:
            bot_action = self._simple_bot_logic()
            self._apply_opponent_action(bot_action)
        else: # for testing
            scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
            self._apply_action(scaled_opponent_action, 3 - self.player_id)
            
        p.stepSimulation(physicsClientId=self.client)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated, truncated, goal_scored, goal_conceded = self._check_termination(obs)
        
        info = {
            'goal_scored': goal_scored,
            'goal_conceded': goal_conceded
        }

        self.previous_action = action
        return obs, reward, terminated, truncated, info

    def _get_mirrored_obs(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)

        # Get joint states for each team
        team1_joint_states = p.getJointStates(self.table_id, self.team1_joints, physicsClientId=self.client)
        team2_joint_states = p.getJointStates(self.table_id, self.team2_joints, physicsClientId=self.client)

        team1_pos = [state[0] for state in team1_joint_states]
        team1_vel = [state[1] for state in team1_joint_states]
        team2_pos = [state[0] for state in team2_joint_states]
        team2_vel = [state[1] for state in team2_joint_states]

        # Mirrored observation for the opponent (team 2)
        # The opponent sees itself as team 1, and the agent as team 2.
        mirrored_ball_pos = (-ball_pos[0], -ball_pos[1], ball_pos[2])
        mirrored_ball_vel = (-ball_vel[0], -ball_vel[1], ball_vel[2])
        
        # The opponent's joints are now team 1, and the agent's joints are team 2
        mirrored_joint_pos = team2_pos + team1_pos
        mirrored_joint_vel = team2_vel + team1_vel

        return np.concatenate([mirrored_ball_pos, mirrored_ball_vel, mirrored_joint_pos, mirrored_joint_vel]).astype(np.float32)

    def _set_opponent_rods_to_90_degrees(self):
        """Set opponent rods to 90 degrees."""
        revs = self.team2_rev_joints if self.player_id == 1 else self.team1_rev_joints
        for joint_id in revs:
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=np.pi/2, force=self.max_force, physicsClientId=self.client)

    def _set_all_rods_to_90_degrees(self):
        """Set all rods to 90 degrees for a clear view."""
        all_rev_joints = self.team1_rev_joints + self.team2_rev_joints
        for joint_id in all_rev_joints:
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=np.pi/2, force=self.max_force, physicsClientId=self.client)
        # Step simulation a few times to let rods settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
            if self.render_mode == 'human':
                time.sleep(1./240.)

    def _simple_bot_logic(self):
        action = np.zeros(8)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        if self.player_id == 1: mirrored_ball_pos = [-ball_pos[0], ball_pos[1], ball_pos[2]]
        else: mirrored_ball_pos = [ball_pos[0], ball_pos[1], ball_pos[2]]
        
        if self.curriculum_level == 2:
            for i in range(4):
                action[i] = np.random.uniform(-1, 1)
                action[i+4] = np.random.uniform(-np.pi, np.pi)
        elif self.curriculum_level >= 3:
            for i in range(4):
                action[i] = np.clip(-mirrored_ball_pos[1] / 0.3, -1, 1)
                action[i+4] = np.pi/2 # Keep rods vertical
        return action

    def _apply_opponent_action(self, action):
        if self.player_id == 1:
            revs, slides = self.team2_rev_joints, self.team2_slide_joints
        else:
            revs, slides = self.team1_rev_joints, self.team1_slide_joints
        for i, joint_id in enumerate(slides):
            lower, upper = self.joint_limits[joint_id]
            target_pos = lower + (action[i] + 1) / 2 * (upper - lower)
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=target_pos, force=self.max_force, physicsClientId=self.client)
        for i, joint_id in enumerate(revs):
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=action[i+4], force=self.max_force, physicsClientId=self.client)

    def _scale_action(self, action, player_id):
        scaled = np.zeros(8)
        if player_id == 1:
            slides, revs = self.team1_slide_joints, self.team1_rev_joints
        else:
            slides, revs = self.team2_slide_joints, self.team2_rev_joints
        for i, joint_id in enumerate(slides):
            lower, upper = self.joint_limits[joint_id]
            scaled[i] = lower + (action[i] + 1) / 2 * (upper - lower)
        for i, joint_id in enumerate(revs):
            scaled[i + 4] = action[i + 4] * np.pi # [MODIFIED] Scale to angle (Pi) instead of max_torque
        return scaled

    def _apply_action(self, scaled_action, player_id):
        if player_id == 1:
            slides, revs = self.team1_slide_joints, self.team1_rev_joints
        else:
            slides, revs = self.team2_slide_joints, self.team2_rev_joints
        
        # Slide control (Position Control)
        for i, joint_id in enumerate(slides):
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=scaled_action[i], maxVelocity=self.max_vel, force=self.max_force, physicsClientId=self.client)
        
        # Rotation control (MODIFIED: Switch to Position Control)
        for i, joint_id in enumerate(revs):
            p.setJointMotorControl2(
                self.table_id, 
                joint_id, 
                p.POSITION_CONTROL,  # Changed from TORQUE_CONTROL
                targetPosition=scaled_action[i + 4], # Now interpreted as an angle
                maxVelocity=15.0, # High velocity for snappy kicks
                force=self.max_torque, 
                physicsClientId=self.client
            )

    def _debug_step(self, action):
        for joint_id in self.team1_rev_joints:
            if joint_id in self.slider_ids: p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=p.readUserDebugParameter(self.slider_ids[joint_id], physicsClientId=self.client), maxVelocity=self.max_vel, force=self.max_force, physicsClientId=self.client)
        for joint_id in self.team1_slide_joints:
            if joint_id in self.slider_ids: p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=p.readUserDebugParameter(self.slider_ids[joint_id], physicsClientId=self.client), maxVelocity=self.max_vel, force=self.max_force, physicsClientId=self.client)

    def _get_obs(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
        joint_states = p.getJointStates(self.table_id, self.all_joints, physicsClientId=self.client)
        if joint_states is None: return np.zeros(self.observation_space.shape)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        if self.player_id == 2:
            ball_pos = (-ball_pos[0], -ball_pos[1], ball_pos[2])
            ball_vel = (-ball_vel[0], -ball_vel[1], ball_vel[2])
        return np.concatenate([ball_pos, ball_vel, joint_pos, joint_vel]).astype(np.float32)

    def get_mirrored_obs_from_unnormalized_obs(self, obs):
        ball_pos, ball_vel, joint_pos, joint_vel = obs[:3], obs[3:6], obs[6:22], obs[22:38]
        
        # Mirror ball state
        mirrored_ball_pos = np.array([-ball_pos[0], -ball_pos[1], ball_pos[2]])
        mirrored_ball_vel = np.array([-ball_vel[0], -ball_vel[1], ball_vel[2]])
        
        # Mirror joint states
        team1_pos, team2_pos = joint_pos[:8], joint_pos[8:]
        mirrored_joint_pos = np.concatenate([team2_pos, team1_pos])
        
        team1_vel, team2_vel = joint_vel[:8], joint_vel[8:]
        mirrored_joint_vel = np.concatenate([team2_vel, team1_vel])
        
        return np.concatenate([mirrored_ball_pos, mirrored_ball_vel, mirrored_joint_pos, mirrored_joint_vel]).astype(np.float32)


    def _compute_reward(self, action):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
        reward = 0

        # --- Sparse Rewards for Goals ---
        goal_reward = self.reward_config['goal_reward']
        concede_penalty = self.reward_config['concede_penalty']

        if (self.player_id == 1 and ball_pos[0] > self.goal_line_x_2) or \
           (self.player_id == 2 and ball_pos[0] < self.goal_line_x_1):
            reward += goal_reward  # Goal for agent
        if (self.player_id == 1 and ball_pos[0] < self.goal_line_x_1) or \
           (self.player_id == 2 and ball_pos[0] > self.goal_line_x_2):
            reward -= concede_penalty  # Own goal

        # --- Dense Rewards and Penalties ---

        # 1a. Dense reward for minimizing distance to ball
        agent_player_links = self.team1_player_links if self.player_id == 1 else self.team2_player_links
        min_dist_to_ball = float('inf')
        
        player_ball_distance_scale = self.reward_config['player_ball_distance_scale']
        
        for link_idx in agent_player_links:
            link_state = p.getLinkState(self.table_id, link_idx, physicsClientId=self.client)
            link_pos = np.array(link_state[0]) # This is the link's CoM
            dist = np.linalg.norm(np.array(ball_pos) - link_pos)
            if dist < min_dist_to_ball:
                min_dist_to_ball = dist
        
        # Reward is higher when the player is closer to the ball.
        # The value is scaled to be significant but not overpower the goal reward.
        distance_reward = player_ball_distance_scale * (1 - np.tanh(min_dist_to_ball))
        reward += distance_reward

        # 1. Reward for ball velocity towards opponent's goal
        ball_velocity_scale = self.reward_config['ball_velocity_scale']
        reward += (ball_vel[0] if self.player_id == 1 else -ball_vel[0]) * ball_velocity_scale

        # 2. Reward for making contact with the ball
        contact_with_agent = False
        contact_reward_value = self.reward_config['contact_reward']
        for link_idx in agent_player_links:
            if p.getContactPoints(bodyA=self.table_id, bodyB=self.ball_id, linkIndexA=link_idx, physicsClientId=self.client):
                contact_with_agent = True
                break
        if contact_with_agent:
            reward += contact_reward_value

        # 3. Refined penalties for agent stagnation
        agent_slides = self.team1_slide_joints if self.player_id == 1 else self.team2_slide_joints
        agent_spins = self.team1_rev_joints if self.player_id == 1 else self.team2_rev_joints

        # 3a. Positional (slide) stagnation penalty
        current_slide_pos = np.array([state[0] for state in p.getJointStates(self.table_id, agent_slides, physicsClientId=self.client)])
        if np.linalg.norm(current_slide_pos - self.last_slide_positions) < 0.001:
            self.slide_stagnation_counter += 1
        else:
            self.slide_stagnation_counter = 0
        if self.slide_stagnation_counter > self.reward_config.get('stagnation_slide_steps', 150): # Get from config or default
            reward -= self.reward_config.get('stagnation_slide_penalty', 0.05) # Get from config or default
        self.last_slide_positions = current_slide_pos

        # 3b. Power usage penalty
        power_penalty_scale = self.reward_config.get('power_penalty_scale', 0.001)
        current_spin_vels = np.array([state[1] for state in p.getJointStates(self.table_id, agent_spins, physicsClientId=self.client)])
        # The torque is the action itself, scaled
        torques = action[4:] * self.max_torque
        power_usage = np.sum(np.abs(torques * current_spin_vels))
        reward -= power_penalty_scale * power_usage

        # 4. Penalty for the ball being stuck
        if np.linalg.norm(ball_vel) < 0.01:
            self.ball_stuck_counter += 1
        else:
            self.ball_stuck_counter = 0

        if self.ball_stuck_counter > self.reward_config.get('stuck_penalty_threshold', 500):
            reward -= self.reward_config.get('stuck_penalty_per_step', 0.01) * self.ball_stuck_counter

        # 5. Inactivity penalty (low ball velocity)
        inactivity_threshold = self.reward_config.get('inactivity_threshold', 0.001)
        if np.linalg.norm(ball_vel) < inactivity_threshold:
            reward -= self.reward_config.get('inactivity_penalty', 0.01)

        # 6. Rod angle penalty
        rod_angle_penalty_scale = self.reward_config.get('rod_angle_penalty_scale', 0.005)
        # Assuming agent_spins are the revolute joints we want to penalize for angle
        current_rev_pos = np.array([state[0] for state in p.getJointStates(self.table_id, agent_spins, physicsClientId=self.client)])
        reward -= rod_angle_penalty_scale * np.sum(np.abs(current_rev_pos))

        # 7. Midfield reward
        midfield_reward_weight = self.reward_config.get('midfield_reward_weight', 0.0)
        reward += midfield_reward_weight * (1 - np.tanh(abs(ball_pos[1])))

        return reward

    def _check_termination(self, obs):
        ball_pos, ball_vel = obs[:3], obs[3:6]
        terminated, truncated = False, False
        goal_scored, goal_conceded = 0, 0

        if self.player_id == 1:
            if ball_pos[0] > self.goal_line_x_2:
                terminated = True
                goal_scored = 1
            elif ball_pos[0] < self.goal_line_x_1:
                terminated = True
                goal_conceded = 1
        else: # Player 2
            if ball_pos[0] < self.goal_line_x_1:
                terminated = True
                goal_scored = 1
            elif ball_pos[0] > self.goal_line_x_2:
                terminated = True
                goal_conceded = 1

        table_aabb = p.getAABB(self.table_id, physicsClientId=self.client)
        if not (table_aabb[0][0] - 0.1 < ball_pos[0] < table_aabb[1][0] + 0.1 and \
                table_aabb[0][1] - 0.1 < ball_pos[1] < table_aabb[1][1] + 0.1):
            truncated = True

        if np.linalg.norm(ball_vel) < 0.001:
            self.ball_stuck_counter += 1
        else:
            self.ball_stuck_counter = 0
        
        if self.ball_stuck_counter > self.max_stuck_steps:
            truncated = True
        
        if self.episode_step_count > self.max_episode_steps:
            truncated = True
            
        return terminated, truncated, goal_scored, goal_conceded

    def close(self):
        p.disconnect(self.client)

    def run_goal_debug_loop(self):
        """
        An interactive debug loop for visualizing goal lines and manually controlling the ball,
        now with real-time contact status reporting.
        """

        if not self.goal_debug_mode:
            print("Goal debug mode is not enabled. Please instantiate Env with goal_debug_mode=True.")
            return

        print("\n" + "="*80 + "\nINTERACTIVE DEBUG MODE\n" + "="*80)
        print(" - Use ARROW KEYS to move the ball.")
        print(" - Use the sliders to adjust the goal lines.")
        print(" - Contact status with agent/opponent rods will be printed on change.")
        print(" - Press ESC or close the window to exit.")

        table_aabb = p.getAABB(self.table_id, physicsClientId=self.client)
        y_min, y_max = table_aabb[0][1], table_aabb[1][1]
        z_pos = 0.55  # Approximate height of the playing surface

        line1_id, line2_id = None, None
        move_speed = 0.01
        last_contact_status = "None"

        try:
            while True:
                # Keyboard events for ball control
                keys = p.getKeyboardEvents(physicsClientId=self.client)
                ball_pos, ball_orn = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
                new_pos = list(ball_pos)

                if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: new_pos[0] -= move_speed
                if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: new_pos[0] += move_speed
                if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: new_pos[1] += move_speed
                if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: new_pos[1] -= move_speed
                p.resetBasePositionAndOrientation(self.ball_id, new_pos, ball_orn, physicsClientId=self.client)

                # --- Contact Detection Logic ---
                current_contact_status = "None"
                agent_player_links = self.team1_player_links if self.player_id == 1 else self.team2_player_links
                opponent_player_links = self.team2_player_links if self.player_id == 1 else self.team1_player_links


                # Check for contact with agent rods
                for link_idx in agent_player_links:
                    if p.getContactPoints(bodyA=self.table_id, bodyB=self.ball_id, linkIndexA=link_idx, physicsClientId=self.client):
                        current_contact_status = f"Contact with AGENT (Player {self.player_id})"
                        break

                # If no agent contact, check for opponent contact
                if current_contact_status == "None":
                    for link_idx in opponent_player_links:
                        if p.getContactPoints(bodyA=self.table_id, bodyB=self.ball_id, linkIndexA=link_idx, physicsClientId=self.client):
                            current_contact_status = f"Contact with OPPONENT (Player {3 - self.player_id})"
                            break

                if current_contact_status != last_contact_status:
                    print(f"\n[Contact Status Changed] -> {current_contact_status}")
                    last_contact_status = current_contact_status

                # Read sliders and update goal lines
                self.goal_line_x_1 = p.readUserDebugParameter(self.goal_line_slider_1, physicsClientId=self.client)
                self.goal_line_x_2 = p.readUserDebugParameter(self.goal_line_slider_2, physicsClientId=self.client)

                # Draw new debug lines
                if line1_id is not None: p.removeUserDebugItem(line1_id, physicsClientId=self.client)
                if line2_id is not None: p.removeUserDebugItem(line2_id, physicsClientId=self.client)
                line1_id = p.addUserDebugLine([self.goal_line_x_1, y_min, z_pos], [self.goal_line_x_1, y_max, z_pos], [1, 0, 0], 2, physicsClientId=self.client)
                line2_id = p.addUserDebugLine([self.goal_line_x_2, y_min, z_pos], [self.goal_line_x_2, y_max, z_pos], [0, 0, 1], 2, physicsClientId=self.client)

                p.stepSimulation(physicsClientId=self.client)
                time.sleep(1./240.)

        except p.error as e:
            pass # This can happen if the user closes the window
        finally:
            print("\nExiting interactive debug mode.")
            self.close()
