import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import unittest

URDF_PATH = "foosball.urdf"
BALL_RADIUS = 0.025  # 標準手足球約 34mm 直徑
BALL_MASS = 0.025    # 約 25g
GRAVITY = -9.8

class FoosballTester:
    def __init__(self, mode=p.GUI):

        # 1. 初始化 PyBullet
        # 在 M1 Mac 上，有時使用 GUI 會不穩定，加入 options 嘗試改善
        self.client = p.connect(mode) 

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        p.setRealTimeSimulation(0) # 我們手動 step simulation 以獲得精確控制

        # 設定攝影機視角
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

        # 2. 載入資產
        self.plane_id = p.loadURDF("plane.urdf")
        
        print(f"正在載入 {URDF_PATH} ...")
        try:
            # useFixedBase=True 確保桌子不會亂動
            self.table_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0.5], useFixedBase=True)
        except Exception as e:
            print(f"錯誤：找不到 {URDF_PATH}。請確認檔案在同一個資料夾內。")
            raise e

        # 3. 解析關節 (Joints)
        self.joints = {}     # 儲存 joint_name -> joint_index
        self.sliders = {}    # 儲存 joint_name -> slider_id
        self.rod_indices = [] # 儲存所有球桿相關的 joint index
        
        self.parse_joints()
        self.create_gui()
        
        # 緩衝時間：等待 GUI 元素完全建立，避免 M1 Mac 上讀取過快導致的錯誤
        time.sleep(0.5)

        # 4. 生成足球
        self.ball_id = None
        self.reset_ball()

        # 狀態標記
        self.is_diagnostic_running = False
        self.score_a = 0
        self.score_b = 0

    def safe_read_param(self, param_id, default_value=0):
        """安全讀取 GUI 參數，避免 PyBullet 錯誤導致崩潰"""
        try:
            return p.readUserDebugParameter(param_id)
        except Exception:
            return default_value

    def parse_joints(self):
        """讀取 URDF 中的所有 Joint"""
        num_joints = p.getNumJoints(self.table_id)
        print(f"模型共有 {num_joints} 個關節。")
        
        # Initialize goal links
        self.goal_link_a = None
        self.goal_link_b = None
        
        for i in range(num_joints):
            info = p.getJointInfo(self.table_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # 記錄起來
            self.joints[joint_name] = i
            
            # 找出球門感測器的 Link Index (用於碰撞偵測)
            if "goal_sensor_A" in joint_name:
                self.goal_link_a = i
                print(f"偵測到 A 隊球門感測器 Link ID: {i}")
            if "goal_sensor_B" in joint_name:
                self.goal_link_b = i
                print(f"偵測到 B 隊球門感測器 Link ID: {i}")
            
            # 找出需要控制的關節 (Prismatic=Slide, Revolute=Rotate)
            # PyBullet 會將 URDF 的 continuous joint 載入為 JOINT_REVOLUTE
            if joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
                self.rod_indices.append(i)

    def create_gui(self):
        """建立除錯面板"""
        # 一般控制滑桿
        for joint_index in self.rod_indices:
            info = p.getJointInfo(self.table_id, joint_index)
            joint_name = info[1].decode('utf-8')
            lower = info[8]
            upper = info[9]
            
            # 處理 Continuous Joint (旋轉無限制)
            if upper < lower:
                lower = -3.14
                upper = 3.14
            
            # 建立滑桿
            slider = p.addUserDebugParameter(joint_name, lower, upper, 0)
            self.sliders[joint_name] = (slider, joint_index)

        # 功能按鈕
        self.btn_reset_ball = p.addUserDebugParameter("Reset Ball (R)", 1, 0, 0)
        self.btn_diagnostic = p.addUserDebugParameter("Run Diagnostics", 1, 0, 0)
        self.btn_reset_score = p.addUserDebugParameter("Reset Score", 1, 0, 0)
        
        # 檢查按鈕 ID 是否有效
        if self.btn_diagnostic < 0:
            print("警告：無法建立診斷按鈕 GUI")

        # 文字顯示
        self.debug_text_id = p.addUserDebugText("Score A: 0  B: 0", [0, 0, 1.5], textColorRGB=[1, 1, 1], textSize=1.5)

    def reset_ball(self):
        """重新生成/重置球的位置"""
        if self.ball_id is not None:
            p.removeBody(self.ball_id)
        
        # 建立球的視覺與碰撞形狀
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 1, 1, 1])
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
        
        # 隨機稍微偏一點點，避免正中央死區
        start_pos = [0, np.random.uniform(-0.01, 0.01), 0.55] # 0.5 (table height) + 0.05
        
        self.ball_id = p.createMultiBody(
            baseMass=BALL_MASS,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        # 增加一點彈性與滾動摩擦
        p.changeDynamics(self.ball_id, -1, restitution=0.8, rollingFriction=0.001, spinningFriction=0.001, lateralFriction=0.01)

    def check_goal(self):
        """檢查球是否碰到球門感測器"""
        if self.ball_id is None: return

        # 檢查是否碰到 A 球門
        contacts_a = p.getContactPoints(bodyA=self.ball_id, bodyB=self.table_id, linkIndexB=self.goal_link_a)
        if contacts_a:
            print("!!! GOAL FOR TEAM B !!!")
            self.score_b += 1
            self.update_score_text()
            self.reset_ball()
            time.sleep(0.5) # 暫停一下

        # 檢查是否碰到 B 球門
        contacts_b = p.getContactPoints(bodyA=self.ball_id, bodyB=self.table_id, linkIndexB=self.goal_link_b)
        if contacts_b:
            print("!!! GOAL FOR TEAM A !!!")
            self.score_a += 1
            self.update_score_text()
            self.reset_ball()
            time.sleep(0.5)

    def update_score_text(self):
        p.removeUserDebugItem(self.debug_text_id)
        text = f"Score A: {self.score_a}  B: {self.score_b}"
        self.debug_text_id = p.addUserDebugText(text, [0, 0, 1.2], textColorRGB=[1, 1, 1], textSize=1.5)

    def run_diagnostics(self):
        """自動化測試：移動所有關節以檢測限制"""
        print("\n=== 開始自動化診斷 ===")
        self.is_diagnostic_running = True
        
        # 1. 測試滑動 (Slide)
        print("測試滑動極限...")
        steps = 100
        for i in range(steps):
            val = math.sin(i * 0.1) * 0.15 # 擺盪幅度
            for name, (slider, idx) in self.sliders.items():
                if "slide" in name:
                    p.setJointMotorControl2(self.table_id, idx, p.POSITION_CONTROL, targetPosition=val)
            p.stepSimulation()
            time.sleep(0.01)
            
        # 2. 測試旋轉 (Rotate)
        print("測試旋轉機制...")
        for i in range(steps):
            angle = i * 0.2
            for name, (slider, idx) in self.sliders.items():
                if "rotate" in name:
                    p.setJointMotorControl2(self.table_id, idx, p.POSITION_CONTROL, targetPosition=angle)
            p.stepSimulation()
            time.sleep(0.01)

        # 歸零
        for name, (slider, idx) in self.sliders.items():
            p.setJointMotorControl2(self.table_id, idx, p.POSITION_CONTROL, targetPosition=0)
            
        print("=== 診斷結束 ===\n")
        self.is_diagnostic_running = False

    def run(self):
        print("開始模擬迴圈。按 'R' 重置球，或使用右側按鈕。")
        
        # 追蹤按鈕狀態 (避免重複觸發)
        btn_counts = {
            "reset_ball": self.safe_read_param(self.btn_reset_ball),
            "diagnostic": self.safe_read_param(self.btn_diagnostic),
            "reset_score": self.safe_read_param(self.btn_reset_score)
        }

        while True:
            # 1. 處理鍵盤事件
            keys = p.getKeyboardEvents()
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                self.reset_ball()

            # 2. 處理 GUI 按鈕事件 - 使用 safe_read_param
            curr_reset_ball = self.safe_read_param(self.btn_reset_ball, btn_counts["reset_ball"])
            if curr_reset_ball > btn_counts["reset_ball"]:
                self.reset_ball()
                btn_counts["reset_ball"] = curr_reset_ball
            
            curr_diagnostic = self.safe_read_param(self.btn_diagnostic, btn_counts["diagnostic"])
            if curr_diagnostic > btn_counts["diagnostic"]:
                self.run_diagnostics()
                btn_counts["diagnostic"] = curr_diagnostic

            curr_reset_score = self.safe_read_param(self.btn_reset_score, btn_counts["reset_score"])
            if curr_reset_score > btn_counts["reset_score"]:
                self.score_a = 0
                self.score_b = 0
                self.update_score_text()
                btn_counts["reset_score"] = curr_reset_score

            # 3. 更新馬達控制 (如果不在診斷模式下)
            if not self.is_diagnostic_running:
                for name, (slider, idx) in self.sliders.items():
                    target_pos = self.safe_read_param(slider)
                    # 使用 Position Control 來驅動關節
                    p.setJointMotorControl2(
                        self.table_id, 
                        idx, 
                        p.POSITION_CONTROL, 
                        targetPosition=target_pos, 
                        force=100  # 馬達力量
                    )

            # 4. 物理模擬步進
            p.stepSimulation()
            
            # 5. 邏輯檢查
            self.check_goal()

            # 控制 FPS
            time.sleep(1./240.)

class TestFoosballScoring(unittest.TestCase):
    def setUp(self):
        self.tester = FoosballTester(mode=p.GUI)
        self.tester.score_a = 0
        self.tester.score_b = 0
        
        # Clear the path by rotating all rods 180 degrees
        for name, (slider, idx) in self.tester.sliders.items():
            if "rotate" in name:
                p.setJointMotorControl2(self.tester.table_id, idx, p.POSITION_CONTROL, targetPosition=math.pi)
        
        # Step simulation to let rods move
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Reset ball to center after rods are out of the way
        self.tester.reset_ball()

    def tearDown(self):
        p.disconnect(self.tester.client)

    def test_goal_for_b(self):
        # Move ball from center towards goal A (Team B scores)
        initial_pos = [0, 0, 0.55]
        goal_link_state = p.getLinkState(self.tester.table_id, self.tester.goal_link_a)
        goal_pos = goal_link_state[0]
        
        # Apply velocity towards the goal
        direction = [goal_pos[0] - initial_pos[0], goal_pos[1] - initial_pos[1], 0]
        dist = (direction[0]**2 + direction[1]**2)**0.5
        if dist > 0:
            velocity = [d / dist * 3.0 for d in direction]  # 3.0 m/s towards goal
        else:
            velocity = [0, 0, 0]
        
        p.resetBaseVelocity(self.tester.ball_id, linearVelocity=velocity)
        
        # Simulate until goal is detected or timeout
        for _ in range(500):  # ~2 seconds at 240Hz
            p.stepSimulation()
            self.tester.check_goal()
            if self.tester.score_b > 0:
                break
            time.sleep(1./240.)
        
        self.assertEqual(self.tester.score_b, 1)
        self.assertEqual(self.tester.score_a, 0)

    def test_goal_for_a(self):
        # Move ball from center towards goal B (Team A scores)
        initial_pos = [0, 0, 0.55]
        goal_link_state = p.getLinkState(self.tester.table_id, self.tester.goal_link_b)
        goal_pos = goal_link_state[0]
        
        # Apply velocity towards the goal
        direction = [goal_pos[0] - initial_pos[0], goal_pos[1] - initial_pos[1], 0]
        dist = (direction[0]**2 + direction[1]**2)**0.5
        if dist > 0:
            velocity = [d / dist * 3.0 for d in direction]  # 3.0 m/s towards goal
        else:
            velocity = [0, 0, 0]
        
        p.resetBaseVelocity(self.tester.ball_id, linearVelocity=velocity)
        
        # Simulate until goal is detected or timeout
        for _ in range(500):  # ~2 seconds at 240Hz
            p.stepSimulation()
            self.tester.check_goal()
            if self.tester.score_a > 0:
                break
            time.sleep(1./240.)
        
        self.assertEqual(self.tester.score_a, 1)
        self.assertEqual(self.tester.score_b, 0)


if __name__ == "__main__":

    tester = FoosballTester()
    tester.run()
