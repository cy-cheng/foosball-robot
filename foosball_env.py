import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time


class FoosballEnv(gym.Env):
    """
    Two-agent symmetric foosball environment for RL training.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode='human', curriculum_level=1, debug_mode=False, player_id=1, opponent_model=None, goal_debug_mode=False):
        super(FoosballEnv, self).__init__()

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
        obs_space_dim = 3 + 3 + 16 + 16
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32)

        self.ball_stuck_counter = 0
        self.max_stuck_steps = 1500
        self.episode_step_count = 0
        self.max_episode_steps = 10000
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSubSteps=4)
        
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        
        urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)
        
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1])

        self._setup_camera()
        
        self.team1_slide_joints = []
        self.team1_rev_joints = []
        self.team2_slide_joints = []
        self.team2_rev_joints = []
        self.joint_name_to_id = {}
        self.joint_limits = {}
        self.goal_link_a = None
        self.goal_link_b = None

        self._parse_joints()
        
        ball_radius = 0.025
        ball_mass = 0.025
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1])
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        self.ball_id = p.createMultiBody(baseMass=ball_mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 0.55])
        p.changeDynamics(self.ball_id, -1, restitution=0.8, rollingFriction=0.001, spinningFriction=0.001, lateralFriction=0.01)
        
        for joint_id in self.team1_slide_joints + self.team1_rev_joints + self.team2_slide_joints + self.team2_rev_joints:
            p.changeDynamics(self.table_id, joint_id, linearDamping=1, angularDamping=1, restitution=0.7)
        
        self.team1_joints = self.team1_slide_joints + self.team1_rev_joints
        self.team2_joints = self.team2_slide_joints + self.team2_rev_joints
        self.all_joints = self.team1_joints + self.team2_joints
        
        self.max_vel = 1.5
        self.max_force = 1.0
        
        self.goal_line_x_1 = -0.75
        self.goal_line_x_2 = 0.75
        self.stuck_ball_penalty = -0.01
        
        if self.goal_debug_mode:
            self.goal_line_slider_1 = p.addUserDebugParameter("Goal Line 1", -1.0, 1.0, self.goal_line_x_1)
            self.goal_line_slider_2 = p.addUserDebugParameter("Goal Line 2", -1.0, 1.0, self.goal_line_x_2)

        if self.debug_mode and self.render_mode == 'human':
            self._add_debug_sliders()

    def update_opponent_model(self, state_dict):
        if self.opponent_model:
            self.opponent_model.policy.load_state_dict(state_dict)

    def _parse_joints(self):
        num_joints = p.getNumJoints(self.table_id)
        team1_slide_joints_map = {}
        team1_rev_joints_map = {}
        team2_slide_joints_map = {}
        team2_rev_joints_map = {}
        for i in range(num_joints):
            info = p.getJointInfo(self.table_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            self.joint_name_to_id[joint_name] = i
            self.joint_limits[i] = (info[8], info[9])
            if "goal_sensor_A" in joint_name: self.goal_link_a = i
            if "goal_sensor_B" in joint_name: self.goal_link_b = i
            if joint_type not in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]: continue
            rod_num = None
            for rod_idx in range(1, 9):
                if f"rod_{rod_idx}" in joint_name:
                    rod_num = rod_idx
                    break
            if rod_num is None: continue
            team = 1 if rod_num in [1, 2, 4, 6] else 2
            if team == 1: p.changeVisualShape(self.table_id, i, rgbaColor=[1, 0, 0, 1])
            else: p.changeVisualShape(self.table_id, i, rgbaColor=[0, 0, 1, 1])
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
        if self.curriculum_level == 1:
            if self.player_id == 1:
                ball_x, ball_y = np.random.uniform(-0.4, 0.0), np.random.uniform(-0.3, 0.3)
                ball_vel = [np.random.uniform(-0.2, -0.1), np.random.uniform(-0.1, 0.1), 0]
            else:
                ball_x, ball_y = np.random.uniform(0.0, 0.4), np.random.uniform(-0.3, 0.3)
                ball_vel = [np.random.uniform(0.1, 0.2), np.random.uniform(-0.1, 0.1), 0]
            ball_pos = [ball_x, ball_y, 0.55]
        elif self.curriculum_level == 2:
            ball_pos = [0, 0, 0.55]
            if self.player_id == 1: ball_vel = [-1, np.random.uniform(-0.5, 0.5), 0]
            else: ball_vel = [1, np.random.uniform(-0.5, 0.5), 0]
        elif self.curriculum_level == 3:
            if self.player_id == 1:
                ball_pos, ball_vel = [-0.4, np.random.uniform(-0.2, 0.2), 0.55], [-5, np.random.uniform(-1, 1), 0]
            else:
                ball_pos, ball_vel = [0.4, np.random.uniform(-0.2, 0.2), 0.55], [5, np.random.uniform(-1, 1), 0]
        else:
            ball_pos, ball_vel = [np.random.uniform(-0.6, 0.6), np.random.uniform(-0.3, 0.3), 0.55], [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0]
        p.resetBasePositionAndOrientation(self.ball_id, ball_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball_id, linearVelocity=ball_vel)

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        for joint_id in self.all_joints: p.resetJointState(self.table_id, joint_id, targetValue=0, targetVelocity=0)
        self._curriculum_spawn_ball()
        if self.curriculum_level == 1:
            for _ in range(20):
                self._set_opponent_rods_to_90_degrees()
                p.stepSimulation()
        self.ball_stuck_counter, self.episode_step_count, self.goals_this_level = 0, 0, 0
        return self._get_obs(), {}

    def step(self, action, opponent_action=None):
        self.episode_step_count += 1
        scaled_action = self._scale_action(action, self.player_id)
        self._apply_action(scaled_action, self.player_id)

        if self.curriculum_level == 1:
            self._set_opponent_rods_to_90_degrees()
        elif self.curriculum_level == 4 and self.opponent_model:
            mirrored_obs = self._get_mirrored_obs()
            opponent_action, _ = self.opponent_model.predict(mirrored_obs, deterministic=True)
            scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
            self._apply_action(scaled_opponent_action, 3 - self.player_id)
        elif opponent_action is None:
            bot_action = self._simple_bot_logic()
            self._apply_opponent_action(bot_action)
        else: # for testing
            scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
            self._apply_action(scaled_opponent_action, 3 - self.player_id)
            
        p.stepSimulation()
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._check_termination(obs)
        return obs, reward, terminated, truncated, {}

    def _get_mirrored_obs(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        joint_states = p.getJointStates(self.table_id, self.all_joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        mirrored_ball_pos = (-ball_pos[0], ball_pos[1], ball_pos[2])
        mirrored_ball_vel = (-ball_vel[0], ball_vel[1], ball_vel[2])
        return np.concatenate([mirrored_ball_pos, mirrored_ball_vel, joint_pos, joint_vel]).astype(np.float32)

    def _set_opponent_rods_to_90_degrees(self):
        """Set opponent rods to 90 degrees."""
        revs = self.team2_rev_joints if self.player_id == 1 else self.team1_rev_joints
        for joint_id in revs:
            p.resetJointState(self.table_id, joint_id, targetValue=np.pi/2)
            p.setJointMotorControl2(self.table_id, joint_id, p.VELOCITY_CONTROL, force=0)

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
                action[i] = np.clip(mirrored_ball_pos[1] / 0.3, -1, 1)
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
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=target_pos, force=self.max_force)
        for i, joint_id in enumerate(revs):
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=action[i+4], force=self.max_force)

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
            scaled[i + 4] = action[i + 4] * 10
        return scaled

    def _apply_action(self, scaled_action, player_id):
        if player_id == 1:
            slides, revs = self.team1_slide_joints, self.team1_rev_joints
        else:
            slides, revs = self.team2_slide_joints, self.team2_rev_joints
        for i, joint_id in enumerate(slides):
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=scaled_action[i], maxVelocity=self.max_vel, force=self.max_force)
        for i, joint_id in enumerate(revs):
            #print(f"Applying velocity {scaled_action[i + 4]} to joint {joint_id}")
            p.setJointMotorControl2(self.table_id, joint_id, p.VELOCITY_CONTROL, targetVelocity=scaled_action[i + 4], force=self.max_force)

    def _debug_step(self, action):
        for joint_id in self.team1_rev_joints:
            if joint_id in self.slider_ids: p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=p.readUserDebugParameter(self.slider_ids[joint_id]), maxVelocity=self.max_vel, force=self.max_force)
        for joint_id in self.team1_slide_joints:
            if joint_id in self.slider_ids: p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=p.readUserDebugParameter(self.slider_ids[joint_id]), maxVelocity=self.max_vel, force=self.max_force)

    def _get_obs(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        joint_states = p.getJointStates(self.table_id, self.all_joints)
        if joint_states is None: return np.zeros(self.observation_space.shape)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        if self.player_id == 2:
            ball_pos = (-ball_pos[0], ball_pos[1], ball_pos[2])
            ball_vel = (-ball_vel[0], ball_vel[1], ball_vel[2])
        return np.concatenate([ball_pos, ball_vel, joint_pos, joint_vel]).astype(np.float32)

    def _compute_reward(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        reward = 0
        goal_x = self.goal_line_x_2 if self.player_id == 1 else self.goal_line_x_1
        reward += (ball_vel[0] if self.player_id == 1 else -ball_vel[0]) * 0.1
        reward -= 0.01 * abs(ball_pos[0] - goal_x)
        team_slides = self.team1_slide_joints if self.player_id == 1 else self.team2_slide_joints
        joint_states = p.getJointStates(self.table_id, team_slides)
        if joint_states: reward += np.mean([abs(state[0]) for state in joint_states]) * 0.1
        team_revs = self.team1_rev_joints if self.player_id == 1 else self.team2_rev_joints
        joint_states_rev = p.getJointStates(self.table_id, team_revs)
        if joint_states_rev:
            active_rods = sum(1 for state in joint_states_rev if abs(state[1]) > 0.1)
            if active_rods > 2: reward -= 0.002 * (active_rods - 2)
        if (self.player_id == 1 and ball_pos[0] > self.goal_line_x_2) or (self.player_id == 2 and ball_pos[0] < self.goal_line_x_1):
            reward += 100
            self.goals_this_level += 1
        if (self.player_id == 1 and ball_pos[0] < self.goal_line_x_1) or (self.player_id == 2 and ball_pos[0] > self.goal_line_x_2):
            reward -= 50
        
        if np.linalg.norm(ball_vel) < 0.001:
            reward += self.stuck_ball_penalty
        
        return reward

    def _check_termination(self, obs):
        ball_pos, ball_vel = obs[:3], obs[3:6]
        terminated, truncated = False, False
        if (self.player_id == 1 and (ball_pos[0] > self.goal_line_x_2 or ball_pos[0] < self.goal_line_x_1)) or \
           (self.player_id == 2 and (ball_pos[0] < self.goal_line_x_1 or ball_pos[0] > self.goal_line_x_2)):
            terminated = True
        table_aabb = p.getAABB(self.table_id)
        if not (table_aabb[0][0] - 0.1 < ball_pos[0] < table_aabb[1][0] + 0.1 and table_aabb[0][1] - 0.1 < ball_pos[1] < table_aabb[1][1] + 0.1):
            truncated = True
        if np.linalg.norm(ball_vel) < 0.001: self.ball_stuck_counter += 1
        else: self.ball_stuck_counter = 0
        if self.ball_stuck_counter > self.max_stuck_steps: truncated = True
        if self.episode_step_count >= self.max_episode_steps: truncated = True
        return terminated, truncated

    def close(self):
        p.disconnect(self.client)

    def run_goal_debug_loop(self):
        """
        A debug loop for visualizing goal lines and manually controlling the ball.
        """
        if not self.goal_debug_mode:
            print("Goal debug mode is not enabled. Please instantiate Env with goal_debug_mode=True.")
            return

        print("\n" + "="*80 + "\nGOAL LINE DEBUG MODE\n" + "="*80)
        print(" - Use ARROW KEYS to move the ball.")
        print(" - Use the sliders to adjust the goal lines.")
        print(" - Ball coordinates are printed in the console.")
        print(" - Press ESC or close the window to exit.")
        
        table_aabb = p.getAABB(self.table_id)
        y_min, y_max = table_aabb[0][1], table_aabb[1][1]
        z_pos = 0.55  # Approximate height of the playing surface

        line1_id = None
        line2_id = None
        
        move_speed = 0.01

        try:
            while True:
                # Keyboard events for ball control
                keys = p.getKeyboardEvents()
                
                ball_pos, ball_orn = p.getBasePositionAndOrientation(self.ball_id)
                new_pos = list(ball_pos)

                if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                    new_pos[0] -= move_speed
                if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                    new_pos[0] += move_speed
                if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                    new_pos[1] += move_speed
                if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                    new_pos[1] -= move_speed
                
                p.resetBasePositionAndOrientation(self.ball_id, new_pos, ball_orn)

                # Read sliders and update goal lines
                self.goal_line_x_1 = p.readUserDebugParameter(self.goal_line_slider_1)
                self.goal_line_x_2 = p.readUserDebugParameter(self.goal_line_slider_2)
                
                # Draw new debug lines
                if line1_id is not None:
                    p.removeUserDebugItem(line1_id)
                if line2_id is not None:
                    p.removeUserDebugItem(line2_id)
                
                line1_id = p.addUserDebugLine([self.goal_line_x_1, y_min, z_pos], [self.goal_line_x_1, y_max, z_pos], [1, 0, 0], 2)
                line2_id = p.addUserDebugLine([self.goal_line_x_2, y_min, z_pos], [self.goal_line_x_2, y_max, z_pos], [0, 0, 1], 2)

                # Print ball coordinates
                current_ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
                goal_status = ""
                # Assuming player 1's perspective for debug
                if current_ball_pos[0] > self.goal_line_x_2:
                    goal_status = "GOAL for Player 1!"
                elif current_ball_pos[0] < self.goal_line_x_1:
                    goal_status = "OWN GOAL (Player 2 scores)"

                print(f"\rBall Position: x={current_ball_pos[0]:.3f}, y={current_ball_pos[1]:.3f}, z={current_ball_pos[2]:.3f} | {goal_status}                ", end="")

                p.stepSimulation()
                time.sleep(1./240.)

        except p.error as e:
            # This can happen if the user closes the window
            pass
        finally:
            print("\nExiting goal debug mode.")
            self.close()


def test_individual_rod_control():
    print("\n" + "="*80 + "\nTEST: INDIVIDUAL ROD CONTROL (Team 1 - RED)\n" + "="*80)
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    num_rods = 4
    for i in range(num_rods):
        print(f"\nRod {i+1} (rotate index {i+4}): Spinning CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = 1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team1_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 1 rod positions: {joint_pos}")
            time.sleep(0.01)
        print(f"Rod {i+1} (rotate index {i+4}): Spinning COUNTER-CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = -1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team1_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 1 rod positions: {joint_pos}")
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    for i in range(num_rods):
        print(f"\nRod {i+1} (slide index {i}): Sliding IN")
        action = np.zeros(8)
        action[i] = 1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        print(f"Rod {i+1} (slide index {i}): Sliding OUT")
        action = np.zeros(8)
        action[i] = -1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    print("\n✅ Individual rod control test complete")
    env.close()

def test_blue_team_rod_control():
    """Test moving each rod of the blue team individually."""
    print("\n" + "="*80 + "\nTEST: INDIVIDUAL ROD CONTROL (Team 2 - BLUE)\n" + "="*80)
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=2)
    obs, _ = env.reset()
    num_rods = 4
    for i in range(num_rods):
        print(f"\nRod {i+1} (rotate index {i+4}): Spinning CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = 1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team2_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 2 rod positions: {joint_pos}")
            time.sleep(0.01)
        print(f"Rod {i+1} (rotate index {i+4}): Spinning COUNTER-CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = -1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team2_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 2 rod positions: {joint_pos}")
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    for i in range(num_rods):
        print(f"\nRod {i+1} (slide index {i}): Sliding IN")
        action = np.zeros(8)
        action[i] = 1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        print(f"Rod {i+1} (slide index {i}): Sliding OUT")
        action = np.zeros(8)
        action[i] = -1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    print("\n✅ Individual rod control test complete for Blue Team")
    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test or debug the FoosballEnv.")
    parser.add_argument("--debug-goal", action="store_true", help="Run the goal line debug mode.")
    args = parser.parse_args()

    if args.debug_goal:
        env = FoosballEnv(goal_debug_mode=True)
        env.run_goal_debug_loop()
    else:
        test_individual_rod_control()
        test_blue_team_rod_control()
