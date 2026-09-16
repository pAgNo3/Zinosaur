#!/usr/bin/env python3
import time
from stable_baselines3 import SAC
from zino_env import ZinoEnv

def main():
    print("Initializing ROS 2 Gym environment...")
    # Note: xml_file_path is no longer needed here as simlauncher loads the XML
    env = ZinoEnv()

    print("Loading SAC policy weights...")
    model = SAC.load("zino_sac_wheels_only")

    print("Starting SAC policy evaluation loop...")
    obs, info = env.reset()
    step_count = 0

    try:
        while True:
            step_start = time.time()
            step_count += 1

            # Get deterministic action prediction from SAC policy
            action, _ = model.predict(obs, deterministic=False)
            
            # Step environment via ROS 2 bridge
            obs, reward, terminated, truncated, info = env.step(action)

            # Log kick perturbations if enabled in launcher/env info
            if info.get("kick_applied"):
                print(f"[Step {step_count}] Kick applied | Force: {info['kick_force']:.2f} N")

            if terminated or truncated:
                print(f"Episode finished after {step_count} steps. Resetting...")
                obs, info = env.reset()
                step_count = 0

            # Target 50 Hz control loop sync
            elapsed = time.time() - step_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

    except KeyboardInterrupt:
        print("\nStopping evaluation loop...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
