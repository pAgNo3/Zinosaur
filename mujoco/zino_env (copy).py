import time
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from zino_env import ZinoEnv

# 1. Initialize environment and load SAC weights
env = ZinoEnv(xml_file_path="zino.xml", phase="wheels_only")
model = SAC.load("zino_sac_wheels_only")

print("Launching 3D MuJoCo Viewer...")

# 2. Launch native MuJoCo passive viewer window
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    obs, _ = env.reset()
    
    while viewer.is_running():
        step_start = time.time()

        # Get deterministic prediction from SAC policy
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

        # Sync viewer state
        viewer.sync()

        # Maintain 50 Hz real-time rendering rate
        time_until_next_frame = (1.0 / 50.0) - (time.time() - step_start)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)
