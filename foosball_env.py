import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from gymnasium import spaces

class FoosballEnv(gym.Env):
    """
    Multi-Agent Foosball Environment (Config-Driven).
    Fully controlled by external YAML configuration passed via update_stage_config().
    """
    metadata = {'render.modes': ['human', 'direct']}

    def __init__(self, config, render_mode='human', curriculum_level=1, debug_mode=False, player_id=1, fixed_active_agent=None):
        super(FoosballEnv, self).__init__()
        
        self.full_config = config
        self.render_mode = render_mode
        self.player_id = player_id 
        self.action_repeat = 8     
        self.fixed_active_agent = fixed_active_agent 
        
        # Default to Stage 1 config if not set
        self.current_stage = curriculum_level
        self.stage_params = self.full_config['curriculum'][f'stage_{curriculum_level}']
        self.rewards = self.stage_params['rewards']
        self.env_cfg = self.stage_params['env_config']

        self.max_episode_steps = 90
        self.episode_step_count = 0
        
        self.num_agents = 4 
        self.agent_roles = ["GK", "DEF", "MID", "STR"]
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.full_config.get('physics', {}).get('gravity', -9.8), physicsClientId=self.client)
        
        # Load Assets
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.client)
        urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(urdf_path, [0, 0, 0.5], useFixedBase=True, physicsClientId=self.client)
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1], physicsClientId=self.client)

        self._setup_camera()
        self._parse_joints()
        self._create_ball()
        
        self.active_agent_idx = 3 
        self.has_touched_ball = [False] * 4
        self.slide_stagnation_counter = np.zeros(4)
        self.last_slide_pos = np.zeros(4)
        
        self.agent_targets = np.zeros((4, 2)) 
        self.opponent_targets = np.zeros((4, 2))
        
        # Goal Lines (from global config as they are physical properties)
        self.goal_line_x_1 = -0.6
        self.goal_line_x_2 = 0.6

    def update_stage_config(self, stage_num):
        """Called by trainer to switch stages dynamically"""
        key = f"stage_{stage_num}"
        if key in self.full_config['curriculum']:
            self.current_stage = stage_num
            self.stage_params = self.full_config['curriculum'][key]
            self.rewards = self.stage_params['rewards']
            self.env_cfg = self.stage_params['env_config']
            return True
        return False

    def _create_ball(self):
        # Using default ball physics
        ball_radius = 0.025
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius, physicsClientId=self.client)
        self.ball_id = p.createMultiBody(baseMass=0.02, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 10], physicsClientId=self.client)
        p.changeDynamics(self.ball_id, -1, restitution=0.7, rollingFriction=0.01, lateralFriction=0.5, physicsClientId=self.client)

    def _parse_joints(self):
        if self.player_id == 1:
            my_rods, op_rods = [1, 2, 4, 6], [8, 7, 5, 3]
        else:
            my_rods, op_rods = [8, 7, 5, 3], [1, 2, 4, 6]
        
        num_joints = p.getNumJoints(self.table_id, physicsClientId=self.client)
        
        # Initialize dictionaries BEFORE usage
        self.joint_limits = {} 
        self.agent_joints = []
        self.opponent_joints = []
        
        def find_rod_joints(rod_list):
            joints_list = []
            for rod_num in rod_list:
                slide_id, rot_id = -1, -1
                for i in range(num_joints):
                    info = p.getJointInfo(self.table_id, i, physicsClientId=self.client)
                    name = info[1].decode('utf-8')
                    if f"rod_{rod_num}" in name and "slide" in name:
                        slide_id = i
                        self.joint_limits[slide_id] = (info[8], info[9]) 
                        p.setJointMotorControl2(self.table_id, i, p.VELOCITY_CONTROL, force=0, physicsClientId=self.client)
                    if f"rod_{rod_num}" in name and "rotate" in name:
                        rot_id = i
                        p.setJointMotorControl2(self.table_id, i, p.VELOCITY_CONTROL, force=0, physicsClientId=self.client)
                if slide_id != -1 and rot_id != -1:
                    joints_list.append((slide_id, rot_id))
            return joints_list

        self.agent_joints = find_rod_joints(my_rods)
        self.opponent_joints = find_rod_joints(op_rods)
        
        # Cache opponent limits
        for pair in self.opponent_joints:
             for j_id in pair:
                 if j_id != -1:
                     info = p.getJointInfo(self.table_id, j_id, physicsClientId=self.client)
                     self.joint_limits[j_id] = (info[8], info[9])

    def reset(self, seed=None, options=None):
        if seed: np.random.seed(seed)
        self.episode_step_count = 0 
        p.resetBaseVelocity(self.ball_id, [0,0,0], [0,0,0], physicsClientId=self.client)
        for i in range(p.getNumJoints(self.table_id, physicsClientId=self.client)):
            p.resetJointState(self.table_id, i, 0, 0, physicsClientId=self.client)

        self.active_agent_idx = self.fixed_active_agent if self.fixed_active_agent is not None else np.random.randint(0, 4)
        active_rod_slide_pos = 0.0

        # 1. Randomize Agents
        for i in range(4):
            slide_joint = self.agent_joints[i][0]
            lower, upper = self.joint_limits[slide_joint]
            eff_lower, eff_upper = max(lower, -0.3), min(upper, 0.3)
            random_slide = np.random.uniform(eff_lower, eff_upper)
            p.resetJointState(self.table_id, slide_joint, targetValue=random_slide, physicsClientId=self.client)
            self.agent_targets[i] = [random_slide, 0.0]
            self.last_slide_pos[i] = random_slide
            if i == self.active_agent_idx: active_rod_slide_pos = random_slide

        # 2. Randomize Opponents
        for i in range(4):
            slide_joint = self.opponent_joints[i][0]
            lower, upper = self.joint_limits[slide_joint]
            eff_lower, eff_upper = max(lower, -0.3), min(upper, 0.3)
            random_slide = np.random.uniform(eff_lower, eff_upper)
            p.resetJointState(self.table_id, slide_joint, targetValue=random_slide, physicsClientId=self.client)
            self.opponent_targets[i] = [random_slide, 0.0]

        # 3. Spawn Ball (Using Stage Config)
        slide_joint = self.agent_joints[self.active_agent_idx][0]
        rod_state = p.getLinkState(self.table_id, slide_joint, physicsClientId=self.client)
        rod_x = rod_state[0][0]
        
        if self.env_cfg['ball_moving']:
            # Stage 3 Logic: Dynamic
            dist_offset = 0.15 if self.player_id == 1 else -0.15
            ball_x = rod_x + dist_offset
            ball_y = np.random.uniform(-0.3, 0.3)
            p.resetBasePositionAndOrientation(self.ball_id, [ball_x, ball_y, 0.55], [0,0,0,1], physicsClientId=self.client)
            
            vel_x = -1.5 if self.player_id == 1 else 1.5 
            vel_y = (active_rod_slide_pos - ball_y) * 2.0 
            p.resetBaseVelocity(self.ball_id, [vel_x, vel_y, 0], [0,0,0], physicsClientId=self.client)
        else:
            # Stage 1 & 2 Logic: Static Close
            offset = self.env_cfg.get('ball_spawn_dist_x', 0.02)
            spawn_offset_x = offset if self.player_id == 1 else -offset
            spawn_offset_x += np.random.uniform(-0.002, 0.002)
            ball_x = rod_x + spawn_offset_x
            ball_y = active_rod_slide_pos + np.random.uniform(-0.02, 0.02)
            ball_y = np.clip(ball_y, -0.3, 0.3)
            p.resetBasePositionAndOrientation(self.ball_id, [ball_x, ball_y, 0.55], [0,0,0,1], physicsClientId=self.client)

        self.has_touched_ball = [False] * 4
        self.slide_stagnation_counter = np.zeros(4)
        
        return self._get_obs(), {}

    def step(self, actions):
        self.episode_step_count += 1 
        rewards = np.zeros(4)
        info = {'goal_scored': 0}
        
        i = self.active_agent_idx
        slide_id = self.agent_joints[i][0]
        lower, upper = self.joint_limits[slide_id]
        
        act_slide = actions[i][0]
        target_slide = lower + (act_slide + 1) / 2 * (upper - lower)
        target_rot = actions[i][1] * np.pi
        self.agent_targets[i] = [target_slide, target_rot]
        
        # --- OPPONENT CONTROL (Config Driven) ---
        if self.env_cfg['opponent_moving']:
             if self.episode_step_count % 10 == 0:
                speed = self.env_cfg.get('opponent_speed', 0.1)
                for j in range(4):
                    if j < len(self.opponent_joints):
                        s_id = self.opponent_joints[j][0]
                        lower, upper = self.joint_limits[s_id]
                        eff_lower, eff_upper = max(lower, -0.3), min(upper, 0.3)
                        current_t = self.opponent_targets[j][0]
                        noise = np.random.uniform(-speed, speed)
                        new_t = np.clip(current_t + noise, eff_lower, eff_upper)
                        self.opponent_targets[j] = [new_t, 0.0]

        # --- PHYSICS LOOP ---
        for _ in range(self.action_repeat):
            for j in range(4):
                # Agent
                s_id, r_id = self.agent_joints[j]
                t_s, t_r = self.agent_targets[j]
                p.setJointMotorControl2(self.table_id, s_id, p.POSITION_CONTROL, targetPosition=t_s, force=100, maxVelocity=1.0, physicsClientId=self.client)
                p.setJointMotorControl2(self.table_id, r_id, p.POSITION_CONTROL, targetPosition=t_r, force=100, maxVelocity=15.0, physicsClientId=self.client)
                
                # Opponent
                s_id, r_id = self.opponent_joints[j]
                t_s, t_r = self.opponent_targets[j]
                p.setJointMotorControl2(self.table_id, s_id, p.POSITION_CONTROL, targetPosition=t_s, force=100, maxVelocity=1.0, physicsClientId=self.client)
                p.setJointMotorControl2(self.table_id, r_id, p.POSITION_CONTROL, targetPosition=t_r, force=100, maxVelocity=15.0, physicsClientId=self.client)

            p.stepSimulation(physicsClientId=self.client)
            rewards += self._compute_physics_rewards()
            if self.render_mode == 'human':
                time.sleep(1.0/240.0)

        rewards += self._compute_decision_rewards()
        obs = self._get_obs()
        terminated, truncated = self._check_termination()
        
        if terminated:
            # Goal Logic using config
            ball_pos = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0]
            goal_val = self.rewards.get('goal_reward', 10.0)
            score = False
            
            if self.player_id == 1:
                if ball_pos[0] > self.goal_line_x_2: 
                    rewards[self.active_agent_idx] += goal_val
                    score = True
                elif ball_pos[0] < self.goal_line_x_1:
                    rewards[self.active_agent_idx] -= goal_val
            elif self.player_id == 2:
                if ball_pos[0] < self.goal_line_x_1:
                    rewards[self.active_agent_idx] += goal_val
                    score = True
                elif ball_pos[0] > self.goal_line_x_2:
                    rewards[self.active_agent_idx] -= goal_val
            
            if score: info['goal_scored'] = 1

        return obs, rewards, terminated, truncated, info

    def _compute_physics_rewards(self):
        step_rewards = np.zeros(4)
        ball_vel = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)[0]
        ball_pos = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0]
        i = self.active_agent_idx
        
        # 1. Velocity Reward
        goal_dir = 1 if self.player_id == 1 else -1
        vel_towards_goal = ball_vel[0] * goal_dir
        if vel_towards_goal > 0:
            step_rewards[i] += vel_towards_goal * self.rewards.get('ball_velocity_scale', 3.0)

        # 2. Spin Reward
        slide_id, rot_id = self.agent_joints[i]
        rod_pos = p.getLinkState(self.table_id, slide_id, physicsClientId=self.client)[0]
        dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(rod_pos[:2]))
        
        rot_vel = p.getJointState(self.table_id, rot_id, physicsClientId=self.client)[1]
        spin_val = self.rewards.get('spin_reward', 0.0)
        
        if dist < 0.1 and spin_val > 0:
            step_rewards[i] += abs(rot_vel) * spin_val
        
        return step_rewards

    def _compute_decision_rewards(self):
        rewards = np.zeros(4)
        i = self.active_agent_idx
        slide_id = self.agent_joints[i][0]
        ball_pos = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0]
        
        # 1. Alignment
        align_val = self.rewards.get('alignment_reward', 0.1)
        if align_val > 0:
            current_slide = p.getJointState(self.table_id, slide_id, physicsClientId=self.client)[0]
            y_diff = abs(current_slide - ball_pos[1])
            rewards[i] += align_val * (1.0 - np.tanh(y_diff * 5.0))
        
        # 2. Stagnation
        stag_pen = self.rewards.get('stagnation_penalty', 0.1)
        if stag_pen > 0:
            current_slide = p.getJointState(self.table_id, slide_id, physicsClientId=self.client)[0]
            if abs(current_slide - self.last_slide_pos[i]) < 0.001:
                self.slide_stagnation_counter[i] += 1
            else:
                self.slide_stagnation_counter[i] = 0
            self.last_slide_pos[i] = current_slide
            
            if self.slide_stagnation_counter[i] > 30:
                rewards[i] -= stag_pen
        
        return rewards

    def _check_termination(self):
        ball_pos = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0]
        terminated = False
        truncated = False
        if ball_pos[0] > self.goal_line_x_2 or ball_pos[0] < self.goal_line_x_1:
            terminated = True
        if self.episode_step_count >= self.max_episode_steps:
            truncated = True
        return terminated, truncated

    def _get_obs(self):
        obs_list = []
        ball_pos = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0]
        ball_vel = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)[0]
        for i in range(4):
            slide_id, rot_id = self.agent_joints[i]
            slide_pos = p.getJointState(self.table_id, slide_id, physicsClientId=self.client)[0]
            rot_pos = p.getJointState(self.table_id, rot_id, physicsClientId=self.client)[0]
            rel_y = ball_pos[1] - slide_pos
            agent_obs = np.array([ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1], slide_pos, rot_pos, rel_y], dtype=np.float32)
            if self.player_id == 2:
                agent_obs[0] *= -1 
                agent_obs[2] *= -1 
            obs_list.append(agent_obs)
        return np.stack(obs_list)

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0.5])
    
    def close(self):
        p.disconnect(self.client)
