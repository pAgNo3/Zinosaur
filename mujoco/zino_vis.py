import time
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from zino_env import ZinoEnv

# 1. Initialize environment and load SAC weights
env = ZinoEnv(xml_file_path="zino.xml", phase="wheels_only")
model = SAC.load("BalancePRO")

print("Launching 3D MuJoCo Viewer...")

step_count = 0

# 2. Launch native MuJoCo passive viewer window
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    obs, info = env.reset()
    
    while viewer.is_running():
        step_start = time.time()
        step_count += 1

        # Get deterministic prediction from SAC policy
        action, _ = model.predict(obs, deterministic=True)
        
        # Capture 'info' dict properly
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if kick occurred on THIS step
        if info.get("kick_applied"):
            print(f"[Step {step_count}] Kick applied | Force: {info['kick_force']:.2f} N")

        if terminated or truncated:
            obs, info = env.reset()
            step_count = 0

        # Sync viewer state
        viewer.sync()

        # Maintain 50 Hz real-time rendering rate
        time_until_next_frame = (1.0 / 50.0) - (time.time() - step_start)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)
