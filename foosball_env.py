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
    Updated with Action Repeat (8x), First Touch Reward, and Relative Observations.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config, render_mode='human', curriculum_level=1, debug_mode=False, player_id=1, opponent_model=None, goal_debug_mode=False):
        super(FoosballEnv, self).__init__()

        self.config = config 
        self.env_config = config['environment']
        self.reward_config = config['reward']
        self.curriculum_config_all_stages = config['curriculum'] 
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
        
        # --- KEY CHANGE: ACTION REPEAT ---
        # 240Hz Simulation / 8 = 30Hz Control Policy
        self.action_repeat = 8

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        else:
            self.client = p.connect(p.DIRECT)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        # --- KEY CHANGE: OBSERVATION SPACE ---
        # Original: 3 (Ball Pos) + 3 (Ball Vel) + 16 (Joint Pos) + 16 (Joint Vel) = 38
        # New: + 4 (Relative Y Distances for my 4 rods) = 42
        obs_space_dim = 3 + 3 + 16 + 16 + 4 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32)

        self.ball_stuck_counter = 0
        self.max_stuck_steps = self.env_config['max_stuck_steps'] 
        self.episode_step_count = 0
        self.previous_action = np.zeros(self.action_space.shape)
        self.previous_ball_dist = 0
        self.last_slide_positions = np.zeros(4)
        self.slide_stagnation_counter = 0
        
        # Reward State
        self.has_touched_ball = False
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.physics_config['gravity'], physicsClientId=self.client) 
        p.setPhysicsEngineParameter(numSubSteps=self.physics_config['physics_substeps'], 
                                    physicsClientId=self.client,
                                    fixedTimeStep=self.physics_config['simulation_timestep']) 
        
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
        
        ball_radius = self.env_config['ball_radius'] 
        ball_mass = self.env_config['ball_mass'] 
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius, physicsClientId=self.client)
        self.ball_id = p.createMultiBody(baseMass=ball_mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 0.55], physicsClientId=self.client)
        p.changeDynamics(self.ball_id, -1, 
                         restitution=self.env_config['ball_restitution'], 
                         rollingFriction=self.env_config['ball_rolling_friction'], 
                         spinningFriction=self.env_config['ball_spinning_friction'], 
                         lateralFriction=self.env_config['ball_lateral_friction'], 
                         physicsClientId=self.client)
        
        for joint_id in self.team1_slide_joints + self.team1_rev_joints + self.team2_slide_joints + self.team2_rev_joints:
            p.changeDynamics(self.table_id, joint_id, 
                             linearDamping=self.env_config['rod_linear_damping'], 
                             angularDamping=self.env_config['rod_angular_damping'], 
                             restitution=self.env_config['rod_restitution'], 
                             physicsClientId=self.client)
        
        self.team1_joints = self.team1_slide_joints + self.team1_rev_joints
        self.team2_joints = self.team2_slide_joints + self.team2_rev_joints
        self.all_joints = self.team1_joints + self.team2_joints
        
        self.max_vel = self.env_config['max_velocity'] 
        self.max_force = self.env_config['max_force'] 
        self.max_torque = self.env_config.get('max_torque', 1.0) 
        
        self.goal_line_x_1 = self.env_config['goal_line_x_1'] 
        self.goal_line_x_2 = self.env_config['goal_line_x_2'] 
        
        if self.goal_debug_mode:
            self.goal_line_slider_1 = p.addUserDebugParameter("Goal Line 1", -1.0, 1.0, self.goal_line_x_1, physicsClientId=self.client)
            self.goal_line_slider_2 = p.addUserDebugParameter("Goal Line 2", -1.0, 1.0, self.goal_line_x_2, physicsClientId=self.client)

        if self.debug_mode and self.render_mode == 'human':
            self._add_debug_sliders()

    def update_opponent_model(self, state_dict):
        if self.opponent_model is None and self.curriculum_level == 4:
            from stable_baselines3 import PPO  
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
            if link_name.startswith('p'): 
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
                    continue 

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

        # Disable the default motors for revolute joints
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
        
        x_spawn_config_key = 'ball_spawn_x_range'
        if self.curriculum_level == 2: 
            x_spawn_config_key = 'ball_spawn_x'
        
        ball_spawn_x_ranges_list = curriculum_stage_config.get(x_spawn_config_key, [[-0.6, 0.6]])
        
        if ball_spawn_x_ranges_list and not isinstance(ball_spawn_x_ranges_list[0], list):
            ball_spawn_x_ranges_list = [ball_spawn_x_ranges_list] 

        selected_x_range = ball_spawn_x_ranges_list[np.random.randint(len(ball_spawn_x_ranges_list))]
        ball_x = np.random.uniform(selected_x_range[0], selected_x_range[1])
        
        ball_spawn_y_range = curriculum_stage_config.get('ball_spawn_y_range', [-0.3, 0.3])
        ball_velocity_range = curriculum_stage_config.get('ball_velocity_range', [-1.0, 1.0])
        ball_velocity_speed_range = curriculum_stage_config.get('ball_velocity_speed_range', [3.0, 4.5])

        if self.curriculum_level == 1:
            ball_y = np.random.uniform(ball_spawn_y_range[0], ball_spawn_y_range[1])
            ball_vel = [np.random.uniform(ball_velocity_range[0], ball_velocity_range[1]), np.random.uniform(-0.1, 0.1), 0]
            ball_pos = [ball_x, ball_y, 0.55]
        elif self.curriculum_level == 2:
            ball_pos = [ball_x, np.random.uniform(ball_spawn_y_range[0], ball_spawn_y_range[1]), 0.55]
            ball_vel = [ball_velocity_range[0], np.random.uniform(-0.5, 0.5), 0] 
        elif self.curriculum_level == 3:
            speed = np.random.uniform(ball_velocity_speed_range[0], ball_velocity_speed_range[1])
            spawn_pos = np.array([ball_x, np.random.uniform(ball_spawn_y_range[0], ball_spawn_y_range[1]), 0.55])
            
            target_goal_line_x = self.goal_line_x_1 if self.player_id == 1 else self.goal_line_x_2
            target_pos = np.array([target_goal_line_x, np.random.uniform(-0.05, 0.05), 0.55])
            
            direction = target_pos - spawn_pos
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0: direction_norm = 1
            ball_vel = (direction / direction_norm) * speed
            ball_pos = spawn_pos.tolist()
        else: # Level 4 (Full Game)
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
            for _ in range(100): 
                self._set_opponent_rods_to_90_degrees()
                p.stepSimulation(physicsClientId=self.client)
        
        self.ball_stuck_counter, self.episode_step_count, self.goals_this_level = 0, 0, 0
        self.last_slide_positions = np.zeros(4)
        self.slide_stagnation_counter = 0
        
        # --- KEY CHANGE: FIRST TOUCH RESET ---
        self.has_touched_ball = False
        
        curriculum_stage_config = self.curriculum_config_all_stages[f'stage_{self.curriculum_level}']
        self.max_episode_steps = curriculum_stage_config.get('max_episode_steps', 2000) 
        
        return self._get_obs(), {}

    def step(self, action, opponent_action=None):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {
            'goal_scored': 0,
            'goal_conceded': 0
        }
        
        # Scale action once
        scaled_action = self._scale_action(action, self.player_id)
        
        # --- KEY CHANGE: ACTION REPEAT LOOP (8x) ---
        for _ in range(self.action_repeat):
            self.episode_step_count += 1
            
            # 1. Apply Actions
            self._apply_action(scaled_action, self.player_id)

            if self.curriculum_level == 1:
                self._set_opponent_rods_to_90_degrees()
            elif self.curriculum_level == 4 and self.opponent_model:
                # Handle self-play opponent
                # Ideally calculate once per decision step, but safely checking/acting here
                if opponent_action is None:
                     mirrored_obs = self._get_mirrored_obs()
                     opp_act_raw, _ = self.opponent_model.predict(mirrored_obs, deterministic=False)
                     scaled_opponent_action = self._scale_action(opp_act_raw * -1, 3 - self.player_id)
                     self._apply_action(scaled_opponent_action, 3 - self.player_id)
                else:
                    scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
                    self._apply_action(scaled_opponent_action, 3 - self.player_id)
                    
            elif opponent_action is None:
                bot_action = self._simple_bot_logic()
                self._apply_opponent_action(bot_action)
            else: 
                scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
                self._apply_action(scaled_opponent_action, 3 - self.player_id)
                
            p.stepSimulation(physicsClientId=self.client)

            # 2. Accumulate Reward (Sum over physics steps)
            step_reward = self._compute_reward(action)
            total_reward += step_reward

            # 3. Check Termination (Instant)
            current_obs = self._get_obs()
            term, trunc, goal_scored, goal_conceded = self._check_termination(current_obs)
            
            if goal_scored: info['goal_scored'] = 1
            if goal_conceded: info['goal_conceded'] = 1
            
            if term or trunc:
                terminated = term
                truncated = trunc
                break # Exit physics loop immediately if done
        
        obs = self._get_obs() 
        self.previous_action = action
        return obs, total_reward, terminated, truncated, info

    def _get_mirrored_obs(self):
        # Generates observation from opponent's perspective (for self-play)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
        team1_joint_states = p.getJointStates(self.table_id, self.team1_joints, physicsClientId=self.client)
        team2_joint_states = p.getJointStates(self.table_id, self.team2_joints, physicsClientId=self.client)
        team1_pos = [state[0] for state in team1_joint_states]
        team1_vel = [state[1] for state in team1_joint_states]
        team2_pos = [state[0] for state in team2_joint_states]
        team2_vel = [state[1] for state in team2_joint_states]
        
        mirrored_ball_pos = (-ball_pos[0], -ball_pos[1], ball_pos[2])
        mirrored_ball_vel = (-ball_vel[0], -ball_vel[1], ball_vel[2])
        mirrored_joint_pos = team2_pos + team1_pos
        mirrored_joint_vel = team2_vel + team1_vel
        
        # Calculate Relative Y for opponent (Team 2 joints are at indices 0-3 in mirrored list)
        rel_y = []
        for i in range(4):
            # Opponent slide pos is mirrored_joint_pos[i]
            # Opponent ball y is mirrored_ball_pos[1]
            rel_y.append(mirrored_ball_pos[1] - mirrored_joint_pos[i])
            
        return np.concatenate([
            mirrored_ball_pos, mirrored_ball_vel, 
            mirrored_joint_pos, mirrored_joint_vel,
            rel_y
        ]).astype(np.float32)

    def _set_opponent_rods_to_90_degrees(self):
        revs = self.team2_rev_joints if self.player_id == 1 else self.team1_rev_joints
        for joint_id in revs:
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=np.pi/2, force=self.max_force, physicsClientId=self.client)

    def _set_all_rods_to_90_degrees(self):
        all_rev_joints = self.team1_rev_joints + self.team2_rev_joints
        for joint_id in all_rev_joints:
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=np.pi/2, force=self.max_force, physicsClientId=self.client)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
            if self.render_mode == 'human':
                time.sleep(1./240.)

    def _simple_bot_logic(self):
        action = np.zeros(8)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        if self.player_id == 1: mirrored_ball_pos = [-ball_pos[0], ball_pos[1], ball_pos[2]]
        else: mirrored_ball_pos = [ball_pos[0], ball_pos[1], ball_pos[2]]
        
        if self.curriculum_level >= 1:
            for i in range(4):
                action[i] = np.clip(-mirrored_ball_pos[1] / 0.3, -1, 1)
                action[i+4] = np.pi/2 
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
            scaled[i + 4] = action[i + 4] * np.pi 
        return scaled

    def _apply_action(self, scaled_action, player_id):
        if player_id == 1:
            slides, revs = self.team1_slide_joints, self.team1_rev_joints
        else:
            slides, revs = self.team2_slide_joints, self.team2_rev_joints
        
        for i, joint_id in enumerate(slides):
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=scaled_action[i], maxVelocity=self.max_vel, force=self.max_force, physicsClientId=self.client)
        
        for i, joint_id in enumerate(revs):
            p.setJointMotorControl2(
                self.table_id, 
                joint_id, 
                p.POSITION_CONTROL, 
                targetPosition=scaled_action[i + 4], 
                maxVelocity=15.0, 
                force=self.max_torque, 
                physicsClientId=self.client
            )

    def _debug_step(self, action):
        pass

    def _get_obs(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
        joint_states = p.getJointStates(self.table_id, self.all_joints, physicsClientId=self.client)
        
        if joint_states is None: 
            return np.zeros(42, dtype=np.float32)
            
        joint_pos = np.array([state[0] for state in joint_states])
        joint_vel = np.array([state[1] for state in joint_states])

        # --- KEY CHANGE: RELATIVE OBSERVATION ---
        
        if self.player_id == 1:
            # Team 1 is indices 0-7. Slides are 0-3.
            my_slides = joint_pos[0:4]
            # Ball Y is ball_pos[1]
            rel_y = ball_pos[1] - my_slides
        else:
            # Player 2 perspective flip
            ball_pos = (-ball_pos[0], -ball_pos[1], ball_pos[2])
            ball_vel = (-ball_vel[0], -ball_vel[1], ball_vel[2])
            
            # Swap Joint Order so Agent is always first
            t1_pos, t2_pos = joint_pos[:8], joint_pos[8:]
            t1_vel, t2_vel = joint_vel[:8], joint_vel[8:]
            
            joint_pos = np.concatenate([t2_pos, t1_pos])
            joint_vel = np.concatenate([t2_vel, t1_vel])
            
            # Agent (now index 0-3) vs Agent Ball Y
            rel_y = ball_pos[1] - joint_pos[0:4]

        # Add 4 relative values to the end
        final_obs = np.concatenate([ball_pos, ball_vel, joint_pos, joint_vel, rel_y]).astype(np.float32)
        return final_obs

    def _compute_reward(self, action):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
        reward = 0

        goal_reward = self.reward_config['goal_reward']
        concede_penalty = self.reward_config['concede_penalty']

        if (self.player_id == 1 and ball_pos[0] > self.goal_line_x_2) or \
           (self.player_id == 2 and ball_pos[0] < self.goal_line_x_1):
            reward += goal_reward 
        if (self.player_id == 1 and ball_pos[0] < self.goal_line_x_1) or \
           (self.player_id == 2 and ball_pos[0] > self.goal_line_x_2):
            reward -= concede_penalty 

        # --- KEY CHANGE: FIRST TOUCH & IMPACT REWARDS ---
        
        # 1. Check for contact
        current_contact = False
        agent_player_links = self.team1_player_links if self.player_id == 1 else self.team2_player_links
        for link_idx in agent_player_links:
            if p.getContactPoints(self.table_id, self.ball_id, linkIndexA=link_idx, physicsClientId=self.client):
                current_contact = True
                break

        # 2. First Touch Bonus (Exploration)
        if current_contact and not self.has_touched_ball:
            reward += 5.0  # Big "Checkpoint" reward
            self.has_touched_ball = True
        
        # 3. Impact/Kick Reward (Velocity based)
        ball_speed = np.linalg.norm(ball_vel)
        if current_contact and ball_speed > 0.5:
            reward += 0.1 * ball_speed 

        # --- Standard Dense Rewards ---
        
        player_ball_distance_scale = self.reward_config['player_ball_distance_scale']
        min_dist_to_ball = float('inf')
        for link_idx in agent_player_links:
            link_state = p.getLinkState(self.table_id, link_idx, physicsClientId=self.client)
            link_pos = np.array(link_state[0]) 
            dist = np.linalg.norm(np.array(ball_pos) - link_pos)
            if dist < min_dist_to_ball:
                min_dist_to_ball = dist
        
        reward += player_ball_distance_scale * (1 - np.tanh(min_dist_to_ball))

        ball_velocity_scale = self.reward_config['ball_velocity_scale']
        reward += (ball_vel[0] if self.player_id == 1 else -ball_vel[0]) * ball_velocity_scale

        return reward

    def _check_termination(self, obs):
        ball_pos = obs[:3]
        ball_vel = obs[3:6] 
        terminated, truncated = False, False
        goal_scored, goal_conceded = 0, 0

        if self.player_id == 1:
            if ball_pos[0] > self.goal_line_x_2:
                terminated = True
                goal_scored = 1
            elif ball_pos[0] < self.goal_line_x_1:
                terminated = True
                goal_conceded = 1
        else: 
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
