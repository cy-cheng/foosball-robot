import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from gymnasium import spaces

class FoosballEnv(gym.Env):
    """
    Multi-Agent Foosball Environment.
    Updated: Adjusted rewards to force STRIKING instead of HUGGING.
    """
    metadata = {'render.modes': ['human', 'direct']}

    def __init__(self, config, render_mode='human', curriculum_level=1, debug_mode=False, player_id=1, fixed_active_agent=None):
        super(FoosballEnv, self).__init__()
        
        self.config = config
        self.env_config = config['environment']
        self.reward_config = config['reward']
        
        stage_key = f'stage_{curriculum_level}'
        self.curriculum_config = config['curriculum'].get(stage_key, config['curriculum']['stage_1'])
        
        self.render_mode = render_mode
        self.curriculum_level = curriculum_level
        self.player_id = player_id 
        self.action_repeat = 8     
        
        self.fixed_active_agent = fixed_active_agent 
        
        # 3.0 seconds limit
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
        p.setGravity(0, 0, self.config['physics']['gravity'], physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.config['physics']['simulation_timestep'], 
            numSubSteps=self.config['physics']['physics_substeps'], 
            physicsClientId=self.client
        )
        
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
        
        # Persistence Holders
        self.agent_targets = np.zeros((4, 2)) 
        self.opponent_targets = np.zeros((4, 2))
        
        self.goal_line_x_1 = self.env_config['goal_line_x_1']
        self.goal_line_x_2 = self.env_config['goal_line_x_2']

    def _create_ball(self):
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=self.env_config['ball_radius'], rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.env_config['ball_radius'], physicsClientId=self.client)
        self.ball_id = p.createMultiBody(baseMass=self.env_config['ball_mass'], baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 10], physicsClientId=self.client)
        p.changeDynamics(self.ball_id, -1, 
                         restitution=self.env_config['ball_restitution'], 
                         rollingFriction=self.env_config['ball_rolling_friction'], 
                         lateralFriction=self.env_config['ball_lateral_friction'], 
                         physicsClientId=self.client)

    def _parse_joints(self):
        if self.player_id == 1:
            my_rods = [1, 2, 4, 6]
            op_rods = [8, 7, 5, 3]
        else:
            my_rods = [8, 7, 5, 3]
            op_rods = [1, 2, 4, 6]
            
        self.agent_joints = [] 
        self.opponent_joints = []
        self.joint_limits = {}
        
        num_joints = p.getNumJoints(self.table_id, physicsClientId=self.client)
        
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

    def reset(self, seed=None, options=None):
        if seed: np.random.seed(seed)
        
        self.episode_step_count = 0 
        
        p.resetBaseVelocity(self.ball_id, [0,0,0], [0,0,0], physicsClientId=self.client)
        for i in range(p.getNumJoints(self.table_id, physicsClientId=self.client)):
            p.resetJointState(self.table_id, i, 0, 0, physicsClientId=self.client)

        if self.fixed_active_agent is not None:
            self.active_agent_idx = self.fixed_active_agent
        else:
            self.active_agent_idx = np.random.randint(0, 4)
        
        active_rod_slide_pos = 0.0

        # --- 1. RANDOMIZE AGENTS ---
        for i in range(4):
            slide_joint = self.agent_joints[i][0]
            lower, upper = self.joint_limits[slide_joint]
            
            eff_lower = max(lower, -0.3)
            eff_upper = min(upper, 0.3)
            
            random_slide = np.random.uniform(eff_lower, eff_upper)
            random_rot = 0.0 
            
            p.resetJointState(self.table_id, slide_joint, targetValue=random_slide, physicsClientId=self.client)
            self.agent_targets[i] = [random_slide, random_rot]
            
            if i == self.active_agent_idx:
                active_rod_slide_pos = random_slide

        # --- 2. RANDOMIZE OPPONENTS ---
        for i in range(4):
            slide_joint = self.opponent_joints[i][0]
            lower, upper = self.joint_limits[slide_joint]
            
            eff_lower = max(lower, -0.3)
            eff_upper = min(upper, 0.3)
            
            random_slide = np.random.uniform(eff_lower, eff_upper)
            random_rot = 0.0 
            
            p.resetJointState(self.table_id, slide_joint, targetValue=random_slide, physicsClientId=self.client)
            self.opponent_targets[i] = [random_slide, random_rot]

        # --- 3. SPAWN BALL ---
        slide_joint = self.agent_joints[self.active_agent_idx][0]
        rod_state = p.getLinkState(self.table_id, slide_joint, physicsClientId=self.client)
        rod_x = rod_state[0][0]
        
        # 2cm offset (Almost touching)
        spawn_offset_x = 0.02 if self.player_id == 1 else -0.02
        spawn_offset_x += np.random.uniform(-0.002, 0.002)
        
        ball_x = rod_x + spawn_offset_x
        ball_y = active_rod_slide_pos + np.random.uniform(-0.02, 0.02)
        ball_y = np.clip(ball_y, -0.3, 0.3)
        
        p.resetBasePositionAndOrientation(self.ball_id, [ball_x, ball_y, 0.55], [0,0,0,1], physicsClientId=self.client)
        
        self.has_touched_ball = [False] * 4
        self.slide_stagnation_counter = np.zeros(4)
        self.last_slide_pos = np.zeros(4)
        
        for i in range(4):
            sid = self.agent_joints[i][0]
            self.last_slide_pos[i] = p.getJointState(self.table_id, sid, physicsClientId=self.client)[0]

        return self._get_obs(), {}

    def step(self, actions):
        self.episode_step_count += 1 
        rewards = np.zeros(4)
        
        # Update Targets (Active Only)
        i = self.active_agent_idx
        slide_id = self.agent_joints[i][0]
        lower, upper = self.joint_limits[slide_id]
        
        act_slide = actions[i][0]
        target_slide = lower + (act_slide + 1) / 2 * (upper - lower)
        target_rot = actions[i][1] * np.pi
        
        self.agent_targets[i] = [target_slide, target_rot]
        
        # --- PHYSICS LOOP ---
        for _ in range(self.action_repeat):
            # Apply Targets
            for j in range(4):
                # Agent
                s_id, r_id = self.agent_joints[j]
                t_s, t_r = self.agent_targets[j]
                p.setJointMotorControl2(self.table_id, s_id, p.POSITION_CONTROL, targetPosition=t_s, force=self.env_config['max_force'], maxVelocity=self.env_config['max_velocity'], physicsClientId=self.client)
                p.setJointMotorControl2(self.table_id, r_id, p.POSITION_CONTROL, targetPosition=t_r, force=self.env_config['max_torque'], maxVelocity=15.0, physicsClientId=self.client)
                
                # Opponent
                s_id, r_id = self.opponent_joints[j]
                t_s, t_r = self.opponent_targets[j]
                p.setJointMotorControl2(self.table_id, s_id, p.POSITION_CONTROL, targetPosition=t_s, force=self.env_config['max_force'], maxVelocity=self.env_config['max_velocity'], physicsClientId=self.client)
                p.setJointMotorControl2(self.table_id, r_id, p.POSITION_CONTROL, targetPosition=t_r, force=self.env_config['max_torque'], maxVelocity=15.0, physicsClientId=self.client)

            p.stepSimulation(physicsClientId=self.client)
            rewards += self._compute_physics_rewards()
            
            if self.render_mode == 'human':
                time.sleep(1.0/240.0)

        decision_rewards = self._compute_decision_rewards()
        rewards += decision_rewards

        obs = self._get_obs()
        terminated, truncated = self._check_termination()
        
        # --- GOAL REWARDS (Restored) ---
        if terminated:
            ball_pos = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0]
            # Player 1 Scoring (Right side > X2)
            if self.player_id == 1:
                if ball_pos[0] > self.goal_line_x_2: 
                    rewards[self.active_agent_idx] += self.reward_config['goal_reward']
                elif ball_pos[0] < self.goal_line_x_1:
                    rewards[self.active_agent_idx] -= self.reward_config['goal_reward']
            # Player 2 Scoring (Left side < X1)
            elif self.player_id == 2:
                if ball_pos[0] < self.goal_line_x_1:
                    rewards[self.active_agent_idx] += self.reward_config['goal_reward']
                elif ball_pos[0] > self.goal_line_x_2:
                    rewards[self.active_agent_idx] -= self.reward_config['goal_reward']

        return obs, rewards, terminated, truncated, {}

    def _compute_physics_rewards(self):
        step_rewards = np.zeros(4)
        ball_vel = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)[0]
        
        i = self.active_agent_idx
        
        # 1. Velocity Reward (Tripled!)
        goal_direction = 1 if self.player_id == 1 else -1
        vel_towards_goal = ball_vel[0] * goal_direction
        
        # BOOST: 3x stronger incentive to hit HARD
        if vel_towards_goal > 0:
            step_rewards[i] += vel_towards_goal * (self.reward_config['ball_velocity_scale'] * 3.0)

        # 2. Spin Reward (New)
        # Hitting well requires spinning. Reward high angular velocity of the rod.
        # This helps the agent discover the "Kick" mechanic.
        _, rot_id = self.agent_joints[i]
        rot_vel = p.getJointState(self.table_id, rot_id, physicsClientId=self.client)[1]
        
        # Only reward spin if ball is moving forward (effective shot)
        if vel_towards_goal > 0.1:
            step_rewards[i] += abs(rot_vel) * 0.001

        return step_rewards

    def _compute_decision_rewards(self):
        rewards = np.zeros(4)
        ball_pos = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0]
        
        i = self.active_agent_idx
        slide_id = self.agent_joints[i][0]
        rod_pos = p.getLinkState(self.table_id, slide_id, physicsClientId=self.client)[0]
        
        dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(rod_pos[:2]))
        
        # --- REMOVED DISTANCE REWARD FOR STAGE 1 ---
        # We spawn ball at 0.02. We don't need to guide them.
        # Rewards[i] += ... (Deleted to prevent hugging)
        
        # Keep First Touch Bonus (Eureka Moment)
        if dist < 0.05 and not self.has_touched_ball[i]:
            rewards[i] += 5.0 
            self.has_touched_ball[i] = True
            
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
            
            agent_obs = np.array([
                ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1],
                slide_pos, rot_pos, rel_y
            ], dtype=np.float32)
            
            if self.player_id == 2:
                agent_obs[0] *= -1 
                agent_obs[2] *= -1 
            
            obs_list.append(agent_obs)
            
        return np.stack(obs_list)

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0.5])
    
    def close(self):
        p.disconnect(self.client)
