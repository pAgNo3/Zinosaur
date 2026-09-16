import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class ZinoEnv(gym.Env):
    """Gymnasium Environment for Zino Wheeled-Bipedal Robot."""
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, xml_file_path="src/DINO_description/urdf/zino.xml", phase="wheels_only", render_mode="human", **kwargs):
        super().__init__()
        self.phase = phase
        self.xml_path = os.path.abspath(xml_file_path)
        self.render_mode = render_mode
        self.viewer = None
        self.frame_skip = 10
        self.prev_V = 0.0
        
        # Observation Space: [pitch, pitch_rate, left_wheel_vel, right_wheel_vel, v_x, roll]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Action Space: Target wheel velocities in rad/s (-20.94 to +20.94)
        if self.phase == "wheels_only":
            self.action_space = spaces.Box(
                low=-20.94, high=20.94, shape=(1,), dtype=np.float32
            )

        # Initialize MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Cache sensor addresses
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "camera_imu_gyro")
        self.quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "camera_imu_quat")
        self.l_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wheel_vel")
        self.r_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wheel_vel")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        
        self.last_action = np.zeros(self.action_space.shape[0])
        self.dt = self.model.opt.timestep * self.frame_skip

    def _get_obs(self):
        # 1. Quat [w, x, y, z] from IMU sensor
        q_adr = self.model.sensor_adr[self.quat_id]
        q = self.data.sensordata[q_adr : q_adr + 4]
        
        # Pitch angle calculation
        sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        
        # Roll angle calculation
        sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # 2. Gyro Pitch Rate (Y-axis angular velocity)
        g_adr = self.model.sensor_adr[self.gyro_id]
        pitch_rate = self.data.sensordata[g_adr + 1]

        # 3. Wheel velocities
        l_adr = self.model.sensor_adr[self.l_vel_id]
        r_adr = self.model.sensor_adr[self.r_vel_id]
        l_wheel_vel = self.data.sensordata[l_adr]
        r_wheel_vel = self.data.sensordata[r_adr]
        
        # 4. Body forward linear velocity
        v_x = float(self.data.qvel[0])

        return np.array([pitch, pitch_rate, l_wheel_vel, r_wheel_vel, v_x, roll], dtype=np.float32)

    def step(self, action):
        info = {}

        # --- A. Apply Control ---
        if self.phase == "wheels_only":
            self.data.ctrl[0] = action[0]  # Left wheel velocity motor
            self.data.ctrl[1] = action[0]  # Right wheel velocity motor
            self.data.ctrl[2:6] = 0.0      # Lock hip servos

        # --- B. Disturbances (Applied BEFORE physics step) ---
        if np.random.rand() < 0.005:  # 3% chance per step
            kick = np.random.choice([-1.0, 1.0]) * np.random.uniform(30.0, 70.0)
            self.data.xfrc_applied[self.torso_body_id] = [kick, 0.0, 0.0, 0.0, 0.0, 0.0]
            info["kick_applied"] = True
            info["kick_force"] = float(kick)
        else:
            self.data.xfrc_applied[self.torso_body_id] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            info["kick_applied"] = False
            info["kick_force"] = 0.0

        # --- C. Step Physics Forward ---
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # --- D. Get Observations ---
        obs = self._get_obs()
        pitch, pitch_rate, l_wheel_vel, r_wheel_vel, v_x, roll = obs

        # --- E. Compute Reward ---
#        r_pitch = -((10.0 * pitch) ** 2)
#        r_rate = -0.01 * (pitch_rate ** 2)
#        r_action = -0.015 * (l_wheel_vel**2)
#        r_alive = 10.0
#        
#        reward = r_pitch + r_rate + r_alive + r_action
#        self.last_action = action.copy()

        # --- F. Fall Termination Condition (> 45 deg tilt) ---
#        terminated = bool(abs(pitch) > 0.785 or abs(roll) > 0.1 or not np.isfinite(obs).all())


        V = 0.01 * (roll**2 + pitch**2 + 0.001*l_wheel_vel**2 + 0.001*r_wheel_vel**2)
        dV = (V - self.prev_V) / self.dt

        r_pitch = -10 ** (pitch ** 2)
        r_roll = -10 ** (roll ** 2)
#        r_yaw = -10 ** (yaw ** 2)
#        r_current = -1 * (I**2)
        r_alive = 25.0
        r_stability = -1 * max(0.0, dV)
        self.prev_V = V
        reward = (r_pitch + r_roll + r_stability + r_alive)
        
        terminated = bool(abs(pitch) > 0.3 or abs(roll) > 0.1 or  not np.isfinite(obs).all())
        if terminated:
            reward -= 5000.0
        
        
        # FIX: Return 'info' instead of empty dict {}
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Reset MuJoCo physics state
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.last_action = np.zeros(self.action_space.shape[0])

        # 2. Set initial pose above ground plane
        self.data.qpos[2] = 0.0 # Safe standing height
        self.data.qpos[4] += np.random.uniform(-0.01, 0.01)  # Tiny pitch perturbation

        # 3. Forward kinematics to settle initial sensors
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
