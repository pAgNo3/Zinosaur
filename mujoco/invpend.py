from __future__ import annotations

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym

# Import your custom environment
from zino_env import ZinoEnv

plt.rcParams["figure.figsize"] = (10, 5)

class Policy_Network(nn.Module):
    """Parametrized Policy Network for Zino."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 32  # Expanded slightly for 2D wheel control
        hidden_space2 = 64

        # Shared Feature Extractor Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        # Ensure positive standard deviation using softplus activation
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm adapted for Multi-Dimensional Action Spaces."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.learning_rate = 3e-4  # Slightly adjusted learning rate
        self.gamma = 0.99          # Discount factor
        self.eps = 1e-6            # Small number for stability

        self.probs = []            # Stores probability values of sampled actions
        self.rewards = []          # Stores corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_means, action_stddevs = self.net(state_tensor)

        # Create a Normal distribution for action sampling
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()

        # CRITICAL FIX FOR 2D ACTIONS: Sum log_probs across both wheel action dimensions
        log_prob = distrib.log_prob(action).sum()

        action_np = action.detach().numpy()
        self.probs.append(log_prob)

        return action_np

    def update(self):
        """Updates policy network weights using Monte Carlo returns."""
        running_g = 0
        gs = []

        # Discounted return calculation (backwards)
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs, dtype=torch.float32)

        # Normalize returns for training stability
        if len(deltas) > 1:
            deltas = (deltas - deltas.mean()) / (deltas.std() + 1e-8)

        log_probs = torch.stack(self.probs)

        # Policy Gradient Loss calculation
        loss = -torch.sum(log_probs * deltas)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset episode tracking buffers
        self.probs = []
        self.rewards = []


# --- 1. INSTANTIATE ZINO ENVIRONMENT ---
base_env = ZinoEnv(xml_filename="zino.xml", phase="wheels_only")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(base_env)

# Observation-space dims (4: pitch, pitch_rate, left_wheel_vel, right_wheel_vel)
obs_space_dims = wrapped_env.observation_space.shape[0]

# Action-space dims (2: left_wheel, right_wheel)
action_space_dims = wrapped_env.action_space.shape[0]

total_num_episodes = 10000  # Number of episodes per seed
rewards_over_seeds = []

# --- 2. TRAINING LOOP ---
for seed in [2, 3, 5, 8]:  # Fibonacci seeds
    print(f"\n--- Starting Training for Seed {seed} ---")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent for each seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step environment with 2D wheel action
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
#            wrapped_env.render()
            agent.rewards.append(reward)
            

            done = terminated or truncated

        # Log total reward for completed episode
        episode_reward = float(wrapped_env.return_queue[-1]) if len(wrapped_env.return_queue) > 0 else 0
        reward_over_episodes.append(episode_reward)

        # Update neural network parameters
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue)) if len(wrapped_env.return_queue) > 0 else 0
            print(f"Episode: {episode:4d} | 50-Ep Avg Reward: {avg_reward}")

    # Save weights cleanly named per seed
    save_filename = f"zino_reinforce_seed_{seed}.pth"
    torch.save(agent.net.state_dict(), save_filename)
    print(f"Weights saved to '{save_filename}'!")

    rewards_over_seeds.append(reward_over_episodes)

wrapped_env.close()

# --- 3. PLOT LEARNING CURVES ---
df1 = pd.DataFrame(rewards_over_seeds).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE Training Curve on Zino Wheeled-Biped"
)
plt.show()
