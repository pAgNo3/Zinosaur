import time
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import numpy as np

# Import your custom Zino environment
from zino_env import ZinoEnv

# 1. UPDATED NETWORK ARCHITECTURE (32 -> 64 hidden layers matching training script)
class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()
        
        hidden_space1 = 32
        hidden_space2 = 64

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )
        
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )
        
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor):
        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs


# 2. INSTANTIATE ZINO ENVIRONMENT
env = ZinoEnv(xml_filename="zino.xml", phase="wheels_only")

obs_space_dims = env.observation_space.shape[0]   # 4
action_space_dims = env.action_space.shape[0] # 2 (Left & Right wheel)

# 3. INSTANTIATE NETWORK & LOAD TRAINED WEIGHTS
trained_net = Policy_Network(obs_space_dims, action_space_dims)

# Update to match the exact .pth file saved by your training script
weights_path = "zino_reinforce_seed_2.pth"  # Or "zino_reinforce_seed_1.pth"
trained_net.load_state_dict(torch.load(weights_path))
trained_net.eval()

print(f"Successfully loaded {weights_path}! Launching 3D viewer...")

# 4. RUN VISUAL SIMULATION WITH MUJOCO PASSIVE VIEWER
obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for step in range(3000):
        if not viewer.is_running():
            break

        step_start = time.time()
        
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Predict deterministic wheel target velocities
            action_mean, _ = trained_net(state_tensor)
            action = action_mean.squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        # Update render frame
        viewer.sync()

        if terminated or truncated:
            obs, info = env.reset()

        # Keep rendering locked at 50 Hz
        time_until_next_frame = (1.0 / 50.0) - (time.time() - step_start)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)

env.close()
