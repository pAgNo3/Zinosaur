#!/usr/bin/env python3
import time
from stable_baselines3 import SAC
from zinoMUJOCO_env import ZinoEnv


def main():
    print("Initializing Gym environment with GUI visualization...")
    # 1. Pass render_mode="human" to enable viewer instantiation
    env = ZinoEnv(render_mode="human")

    print("Loading SAC policy weights...")
    model = SAC.load("zino_sac_wheels_only")

    print("Starting SAC policy evaluation loop...")
    obs, info = env.reset()
    step_count = 0

    try:
        while True:
            step_start = time.time()
            step_count += 1

            # 2. Use deterministic=True for policy evaluation (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # 3. Explicitly render the frame to the viewer window
            env.render()

            # Log kick perturbations
            if info.get("kick_applied"):
                print(
                    f"[Step {step_count}] Kick applied | Force: {info['kick_force']:.2f} N"
                )

            if terminated or truncated:
                print(
                    f"Episode finished after {step_count} steps. Resetting..."
                )
                obs, info = env.reset()
                step_count = 0

            # Target 50 Hz control loop sync (0.02s step)
            elapsed = time.time() - step_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

    except KeyboardInterrupt:
        print("\nStopping evaluation loop...")
    finally:
        env.close()


if __name__ == "__main__":
    main()
