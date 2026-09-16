import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu, JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger


def solve_leg_ik(x: float, y: float, a=142.5, b=315.0, c=324.0):
    E1 = -2 * a * x
    E4 = 2 * a * (-x + c)
    F1 = -2 * a * y
    F4 = -2 * a * y
    G1 = (a**2) - (b**2) + (x**2) + (y**2)
    G4 = (c**2) + (a**2) - (b**2) + (x**2) + (y**2) - 2 * c * x

    disc1 = np.maximum(0.0, E1**2 + F1**2 - G1**2)
    disc4 = np.maximum(0.0, E4**2 + F4**2 - G4**2)
    denom1 = G1-E1
    denom4 = G4-E4
#    denom1 = np.where(np.abs(G1 - E1) < 1e-6, 1e-6, G1 - E1)
#    denom4 = np.where(np.abs(G4 - E4) < 1e-6, 1e-6, G4 - E4)

    t1 = 2 * np.arctan2((-F1 + np.sqrt(disc1)) , denom1)
    t4 = 2 * np.arctan2((-F4 - np.sqrt(disc4)) , denom4)

    return t1, t4


class ZinoRosBridgeNode(Node):
    def __init__(self):
        super().__init__('zino_gym_bridge')
        self.latest_imu = None
        self.latest_joints = None
        qos = QoSProfile(
             history=HistoryPolicy.KEEP_LAST,
             depth=1,
             reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.cmd_pub = self.create_publisher(Float64MultiArray, '/robot_cmd', 10)
        self.create_subscription(Imu, '/imu', self._imu_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        

    def _imu_cb(self, msg):
        self.latest_imu = msg

    def _joint_cb(self, msg):
        self.latest_joints = msg


class ZinoEnv(gym.Env):


    def __init__(self, **kwargs):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()
   
        self.node = ZinoRosBridgeNode()
        self.reset_cli = self.node.create_client(Trigger, '/zino/reset')
        while not self.reset_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for reset service...')
        # Action & Observation spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-20.94, -20.94, 120, 120, 0.4, 0.4], dtype=np.float32),
            high=np.array([20.94, 20.94, 350, 350, 0.6, 0.6], dtype=np.float32),
            dtype=np.float32
        )

        self.prev_V = 0.0
        self.last_action = np.zeros(self.action_space.shape[0])
        self.dt = 0.02  # 50 Hz control loop step period

    def _spin_ros(self, timeout_sec=1.0):
        """Blocks until a NEW IMU and Joint message arrives from MuJoCo (No wall-clock sleep)."""
        self.node.latest_imu = None
        self.node.latest_joints = None
        
        start_time = time.time()
        while self.node.latest_imu is None or self.node.latest_joints is None:
            rclpy.spin_once(self.node, timeout_sec=0.001)
            if time.time() - start_time > timeout_sec:
                self.node.get_logger().error("Timeout waiting for simulation step!")
                break

    def _get_obs(self):
        if self.node.latest_imu is None or self.node.latest_joints is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        q_msg = self.node.latest_imu.orientation
        w, x, y, z = q_msg.w, q_msg.x, q_msg.y, q_msg.z

        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        joints = self.node.latest_joints
        l_wheel_vel, r_wheel_vel = 0.0, 0.0
        l_wheel_current, r_wheel_current = 0.0, 0.0

        if "leftwheel_joint" in joints.name and "rightwheel_joint" in joints.name:
            l_idx = joints.name.index("leftwheel_joint")
            r_idx = joints.name.index("rightwheel_joint")
            l_wheel_vel = joints.velocity[l_idx] if len(joints.velocity) > l_idx else 0.0
            r_wheel_vel = joints.velocity[r_idx] if len(joints.velocity) > r_idx else 0.0
            l_wheel_current = 0.45 + (joints.effort[l_idx]) / 0.768 if len(joints.effort) > l_idx else 0.0
            r_wheel_current = 0.45 + (joints.effort[r_idx]) / 0.768 if len(joints.effort) > r_idx else 0.0

        ULpos, LLpos, URpos, LRpos = 0.0, 0.0, 0.0, 0.0
        ULcurr, LLcurr, URcurr, LRcurr = 0.0, 0.0, 0.0, 0.0

        if all(j in joints.name for j in ["ULhip_joint", "LLhip_joint", "URhip_joint", "LRhip_joint"]):
            UL_idx = joints.name.index("ULhip_joint")
            LL_idx = joints.name.index("LLhip_joint")
            UR_idx = joints.name.index("URhip_joint")
            LR_idx = joints.name.index("LRhip_joint")

            ULpos = joints.position[UL_idx] if len(joints.position) > UL_idx else 0.0
            LLpos = joints.position[LL_idx] if len(joints.position) > LL_idx else 0.0
            URpos = joints.position[UR_idx] if len(joints.position) > UR_idx else 0.0
            LRpos = joints.position[LR_idx] if len(joints.position) > LR_idx else 0.0

            ULcurr = 0.006 + (joints.effort[UL_idx]) / 4.03 if len(joints.effort) > UL_idx else 0.0
            LLcurr = 0.006 + (joints.effort[LL_idx]) / 4.03 if len(joints.effort) > LL_idx else 0.0
            URcurr = 0.006 + (joints.effort[UR_idx]) / 4.03 if len(joints.effort) > UR_idx else 0.0
            LRcurr = 0.006 + (joints.effort[LR_idx]) / 4.03 if len(joints.effort) > LR_idx else 0.0

        a, b, c = 142.5, 315.0, 324.0

        tL1 = LLpos + (np.pi / 2)
        tL4 = -ULpos + (np.pi / 2)
        tR1 = URpos + (np.pi / 2)
        tR4 = -LRpos + (np.pi / 2)

        EL = 2 * b * (c + a * (np.cos(tL4) - np.cos(tL1)))
        ER = 2 * b * (c + a * (np.cos(tR4) - np.cos(tR1)))

        FL = 2 * a * b * (np.sin(tL4) - np.sin(tL1))
        FR = 2 * a * b * (np.sin(tR4) - np.sin(tR1))

        GL = (c**2) + 2 * (a**2) + (2 * c * a * np.cos(tL4)) - (2 * c * a * np.cos(tL1)) - (2 * (a**2) * np.cos(tL4 - tL1))
        GR = (c**2) + 2 * (a**2) + (2 * c * a * np.cos(tR4)) - (2 * c * a * np.cos(tR1)) - (2 * (a**2) * np.cos(tR4 - tR1))
        denomL = GL -EL
        denomR = GR- ER
#        denomL = np.where(abs(GL - EL) < 1e-6, 1e-6, GL - EL)
#        denomR = np.where(abs(GR - ER) < 1e-6, 1e-6, GR - ER)

        xL = c + (a * np.cos(tL4)) + (b * np.cos(2 * np.arctan(abs((-FL + np.sqrt(np.maximum(0.0, (EL**2) + (FL**2) - (GL**2)))) / denomL))))
        xR = c + (a * np.cos(tR4)) + (b * np.cos(2 * np.arctan(abs((-FR + np.sqrt(np.maximum(0.0, (ER**2) + (FR**2) - (GR**2)))) / denomR))))

        yL = (a * np.sin(tL4)) + (b * np.sin(2 * np.arctan(abs((-FL + np.sqrt(np.maximum(0.0, (EL**2) + (FL**2) - (GL**2)))) / denomL))))
        yR = (a * np.sin(tR4)) + (b * np.sin(2 * np.arctan(abs((-FR + np.sqrt(np.maximum(0.0, (ER**2) + (FR**2) - (GR**2)))) / denomR))))

        return np.array([
            pitch, roll, yaw,
            l_wheel_vel, r_wheel_vel, xL, yL, xR, yR,
            ULcurr, LLcurr, URcurr, LRcurr, l_wheel_current, r_wheel_current
        ], dtype=np.float32)

    def _Normalizer(self, xp, y):
        R = 457.5 - 25.0
        xmin = 324.0 - np.sqrt(np.maximum(0.0, (R**2) - (y**2)))
        xmax = np.sqrt(np.maximum(0.0, (R**2) - (y**2)))

        return xmin + xp*(xmax-xmin)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        l_wheel_cmd, r_wheel_cmd = action[0], action[1]
        yL, yR = action[2], action[3]
        Xl, Xr = action[4], action[5]

        xL = self._Normalizer(Xl, yL)
        xR = self._Normalizer(Xr, yR)

        t1_L, t4_L = solve_leg_ik(x=xL, y=yL)
        t1_R, t4_R = solve_leg_ik(x=xR, y=yR)

        TL1 = -t1_L + (3.14 / 2)
        TL4 = t4_L - 1.57
        TR1 = -t1_R + (3.14 / 2)
        TR4 = t4_R - 1.57

        # 1. Publish action command over ROS 2
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [float(l_wheel_cmd), float(r_wheel_cmd), float(TL4), float(TL1), float(TR1), float(TR4)]
        self.node.latest_imu = None
        self.node.latest_joints = None
        self.node.cmd_pub.publish(cmd_msg)

        # 2. Allow physics launcher time to execute steps and stream updated telemetry
        self._spin_ros()

        # 3. Read observation from telemetry
        obs = self._get_obs()
        pitch, roll, yaw, l_wheel_vel, r_wheel_vel, xL, yL, xR, yR, ULcurr, LLcurr, URcurr, LRcurr, l_wheel_current, r_wheel_current = obs

        I = 0.2*abs(ULcurr) + 0.2*abs(LLcurr) + 0.2*abs(URcurr) + 0.2*abs(LRcurr) + 0.0 * abs(l_wheel_current) + 0.0 * abs(r_wheel_current)
        V = 0.01 * (roll**2 + pitch**2 + yaw**2 + 0.001*l_wheel_vel**2 + 0.001*r_wheel_vel**2)
        dV = (V - self.prev_V) / self.dt

        r_pitch = -10 ** (pitch ** 2)
        r_roll = -10 ** (roll ** 2)
        r_yaw = -10 ** (yaw ** 2)
        r_current = -1 * (I**2)
        r_alive = 250.0
        r_stability = -1 * max(0.0, dV)
        self.prev_V = V
        reward = (r_pitch + r_roll + r_yaw + r_current + r_stability + r_alive)

        terminated = bool(abs(pitch) > 0.3 or abs(roll) > 0.1 or abs(yaw) > 0.3 or (((xL-162)**2)+(yL**2))**0.5>450 or (((xL-162)**2)+(yL**2))**0.5>450  or  not np.isfinite(obs).all())
        if terminated:
            reward -= 5000.0

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.node.latest_imu = None
        self.node.latest_joints = None
        if hasattr(self, "obs_buffer"):
            self.obs_buffer.clear() 

        req = Trigger.Request()
        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        # Reset standing command
        target_y = 350
        R = 457.5 - 25.0
        xmin = 324.0 - np.sqrt(R**2 - target_y**2)
        xmax = np.sqrt(R**2 - target_y**2)
        xp_default = (0.0 - xmin) / (xmax - xmin)

        xL_def = self._Normalizer(xp_default, target_y)
        xR_def = self._Normalizer(xp_default, target_y)

        t1_L, t4_L = solve_leg_ik(x=xL_def, y=target_y)
        t1_R, t4_R = solve_leg_ik(x=xR_def, y=target_y)

        TL1 = t1_L + (np.pi / 2)
        TL4 = -t4_L - (np.pi / 2)
        TR1 = -t1_R - (np.pi / 2)
        TR4 = t4_R + (np.pi / 2)

        cmd_msg = Float64MultiArray()
        cmd_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.node.cmd_pub.publish(cmd_msg)

        start_time = time.time()
        while self.node.latest_imu is None or self.node.latest_joints is None:
            rclpy.spin_once(self.node, timeout_sec=0.005)
            if time.time() - start_time > 2.0:
                self.node.get_logger().error("Reset timeout waiting for fresh state!")
                break

        self.last_action = np.zeros(self.action_space.shape[0])
        obs = self._get_obs()

        pitch, roll, yaw, l_wheel_vel, r_wheel_vel, xL, yL, xR, yR, ULcurr, LLcurr, URcurr, LRcurr, l_wheel_current, r_wheel_current = obs
        I_init = abs(ULcurr) + abs(LLcurr) + abs(URcurr) + abs(LRcurr) + abs(l_wheel_current) + abs(r_wheel_current)
        self.prev_V = 0.01 * (roll**2 + pitch**2 + yaw**2 + l_wheel_vel**2 + r_wheel_vel**2)

        return obs, {}

    def close(self):
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()
