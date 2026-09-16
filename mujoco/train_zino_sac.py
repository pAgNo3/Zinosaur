from stable_baselines3 import SAC
from zino_env import ZinoEnv

# 1. Create headless Zino environment
env = ZinoEnv(xml_file_path="zino.xml", phase="wheels_only")

# 2. Configure SAC Agent
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=50000,
    batch_size=256,
    ent_coef="auto",  # Automatic entropy tuning
    verbose=1
)

# 3. Train SAC for 50,000 steps (~1-2 minutes)
print("Training SAC on Zino...")
model.learn(total_timesteps=64000)

# 4. Save trained weights
model.save("zino_sac_wheels_only")
print("Saved trained policy to zino_sac_wheels_only.zip!")
