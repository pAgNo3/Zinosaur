from stable_baselines3 import SAC
from zinoMUJOCO_env import ZinoEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

# 1. Create headless Zino environment
raw_env = DummyVecEnv([lambda: ZinoEnv(xml_file_path="src/DINO_description/urdf/zino.xml", render_mode = None)])

norm_env = VecNormalize(
    raw_env, 
    norm_obs=False, 
    norm_reward=False, 
    clip_obs=10000000.0, 
    clip_reward=10000000.0
)

# 3. Add VecMonitor to track episode lengths and rewards
env = VecMonitor(norm_env)
# 2. Configure SAC Agent
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-3,
    gamma=0.99,
    learning_starts=1000.0,
    buffer_size=100000,
    batch_size=1000,
    ent_coef="15",  # Automatic entropy tuning
    verbose=1
)

# 3. Train SAC for 50,000 steps (~1-2 minutes)
print("Training SAC on Zino...")
model.learn(total_timesteps= 500000)

# 4. Save trained weights
model.save("zino_sac_wheels_only")
print("Saved trained policy to zino_sac_wheels_only.zip!")
