import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

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
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-20.94, -20.94, 120, 120, 0.4, 0.4], dtype=np.float32),
            high=np.array([20.94, 20.94, 350, 350, 0.6, 0.6], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        raw_actuators = {
            "l_wheel": "left_wheel_motor",
            "r_wheel": "right_wheel_motor",
            "UL_hip": "UL_hip_servo",
            "LL_hip": "LL_hip_servo",
            "UR_hip": "UR_hip_servo",
            "LR_hip": "LR_hip_servo",
        }
        
        self.actuator_ids = {}
        for key, name in raw_actuators.items():
          aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
          if aid == -1:
            self.get_logger().error(f"ACTUATOR NOT FOUND IN XML: '{name}'")
          self.actuator_ids[key] = aid
        
        self.target_joints = [
            "leftwheel_joint",
            "rightwheel_joint",
            "ULhip_joint",
            "LLhip_joint",
            "URhip_joint",
            "LRhip_joint",
        ]
        self.joint_actuator_map = {
            "leftwheel_joint": "l_wheel",
            "rightwheel_joint": "r_wheel",
            "ULhip_joint": "UL_hip",
            "LLhip_joint": "LL_hip",
            "URhip_joint": "UR_hip",
            "LRhip_joint": "LR_hip",
        }

        self.joint_indices = {}
        for j_name in self.target_joints:
          j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
          if j_id == -1:
            self.get_logger().error(f"JOINT NOT FOUND IN XML: '{j_name}'")
            continue
          self.joint_indices[j_name] = {
              "qpos_idx": self.model.jnt_qposadr[j_id],
              "qvel_idx": self.model.jnt_dofadr[j_id],
              "act_id": self.actuator_ids[self.joint_actuator_map[j_name]],
          }
        
        # Cache sensor addresses
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "camera_imu_gyro")
        self.quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "camera_im000u_quat")
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
        
        #yaw calculation
        siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        
        positions, velocities, efforts = [], [], []
        for j_name in self.target_joints:
          idxs = self.joint_indices.get(j_name)
          if idxs and idxs["act_id"] != -1:
            positions.append(float(self.data.qpos[idxs["qpos_idx"]]))
            velocities.append(float(self.data.qvel[idxs["qvel_idx"]]))
            efforts.append(float(self.data.actuator_force[idxs["act_id"]]))
          else:
            positions.append(0.0)
            velocities.append(0.0)
            efforts.append(0.0)
        l_wheel_ids = self.joint_indices.get("leftwheel_joint")
        r_wheel_ids = self.joint_indices.get("rightwheel_joint")
        UL_ids = self.joint_indices.get("ULhip_joint")
        LL_ids = self.joint_indices.get("LLhip_joint")
        UR_ids = self.joint_indices.get("URhip_joint")
        LR_ids = self.joint_indices.get("LRhip_joint")
        
        l_wheel_vel = float(self.data.qvel[l_wheel_ids["qvel_idx"]])
        r_wheel_vel = float(self.data.qvel[r_wheel_ids["qvel_idx"]])
        
        ULpos = float(self.data.qpos[UL_ids["qpos_idx"]])
        LLpos = float(self.data.qpos[LL_ids["qpos_idx"]])
        URpos = float(self.data.qpos[UR_ids["qpos_idx"]])
        LRpos = float(self.data.qpos[LR_ids["qpos_idx"]])
        
        ULcurr = 0.006 + (float(self.data.actuator_force[UL_ids["act_id"]]))/4.03
        LLcurr = 0.006 + (float(self.data.actuator_force[LL_ids["act_id"]]))/4.03
        URcurr = 0.006 + (float(self.data.actuator_force[UR_ids["act_id"]]))/4.03
        LRcurr = 0.006 + (float(self.data.actuator_force[LR_ids["act_id"]]))/4.03
        l_wheel_curr = 0.45 + (float(self.data.actuator_force[l_wheel_ids["act_id"]]))/0.768
        r_wheel_curr = 0.45 + (float(self.data.actuator_force[r_wheel_ids["act_id"]]))/0.768
        
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
            ULcurr, LLcurr, URcurr, LRcurr, l_wheel_curr, r_wheel_curr
        ], dtype=np.float32)
    
    def _Normalizer(self, xp, y):
        R = 457.5 - 25.0
        xmin = 324.0 - np.sqrt(np.maximum(0.0, (R**2) - (y**2)))
        xmax = np.sqrt(np.maximum(0.0, (R**2) - (y**2)))
        return xmin + xp*(xmax-xmin)
        
    def step(self, action):
        info = {}

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
        
        output = [l_wheel_cmd, r_wheel_cmd, TL4, TL1, TR1, TR4]
        
        for i, key in enumerate(
            ["l_wheel", "r_wheel", "UL_hip", "LL_hip", "UR_hip", "LR_hip"]
        ):
          aid = self.actuator_ids.get(key, -1)
          if aid != -1:
            self.data.ctrl[aid] = float(output[i])

        # --- B. Disturbances (Applied BEFORE physics step) ---
#        if np.random.rand() < 0.005:  # 3% chance per step
#            kick = np.random.choice([-1.0, 1.0]) * np.random.uniform(30.0, 70.0)
#            self.data.xfrc_applied[self.torso_body_id] = [kick, 0.0, 0.0, 0.0, 0.0, 0.0]
#            info["kick_applied"] = True
#            info["kick_force"] = float(kick)
#        else:
#            self.data.xfrc_applied[self.torso_body_id] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#            info["kick_applied"] = False
#            info["kick_force"] = 0.0

        # --- C. Step Physics Forward ---
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # --- D. Get Observations ---
        obs = self._get_obs()
        pitch, roll, yaw, l_wheel_vel, r_wheel_vel, xL, yL, xR, yR, ULcurr, LLcurr, URcurr, LRcurr, l_wheel_curr, r_wheel_curr = obs

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

        I = 0.2*abs(ULcurr) + 0.2*abs(LLcurr) + 0.2*abs(URcurr) + 0.2*abs(LRcurr) + 0.0 * abs(l_wheel_curr) + 0.0 * abs(r_wheel_curr)
        V = 0.01 * (roll**2 + pitch**2 + 0.001*l_wheel_vel**2 + 0.001*r_wheel_vel**2)
        dV = (V - self.prev_V) / self.dt

        r_pitch = -10 ** (pitch ** 2)
        r_roll = -10 ** (roll ** 2)
        r_yaw = -10 ** (yaw ** 2)
        r_current = -0.75 * (I**2)
        r_alive = 25.0
        r_stability = -1 * max(0.0, dV)
        self.prev_V = V
        reward = (r_pitch + r_roll + r_yaw + r_current + r_stability + r_alive)
        
        terminated = bool(abs(pitch) > 0.3 or abs(roll) > 0.1 or abs(yaw) > 0.3 or (((xL-162)**2)+(yL**2))**0.5>450 or (((xL-162)**2)+(yL**2))**0.5>450  or  not np.isfinite(obs).all())
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
