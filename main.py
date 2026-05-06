# main.py - Training loop. Ties together the network, buffer, and reward.
# Runs the DQN training process and logs progress.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import flappy_bird_gymnasium
import gymnasium as gym

from model import DQN
from replay_buffer import ReplayBuffer
from reward import shaped_reward

# --- Hyperparameters ---
EPISODES        = 1000    # total training episodes
BATCH_SIZE      = 64      # how many experiences to sample per update
BUFFER_CAPACITY = 10000   # max experiences stored
GAMMA           = 0.99    # discount factor — how much future rewards matter
LR              = 1e-4    # learning rate
EPSILON_START   = 1.0     # start fully random
EPSILON_MIN     = 0.01    # never go fully greedy
EPSILON_DECAY   = 0.995   # how fast to reduce randomness
TARGET_UPDATE   = 50      # copy main → target network every N episodes

# --- Setup ---
env = gym.make("FlappyBird-v0", use_lidar=False) # add , render_mode="human" if you want to see(slower)

main_net   = DQN()
target_net = DQN()
target_net.load_state_dict(main_net.state_dict())  # start both networks identical
target_net.eval()  # target network is never trained directly, only copied to

optimizer = optim.Adam(main_net.parameters(), lr=LR)
loss_fn   = nn.MSELoss()
buffer    = ReplayBuffer(BUFFER_CAPACITY)
epsilon   = EPSILON_START

# --- Training loop ---
for episode in range(EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # explore
        else:
            with torch.no_grad():
                q_values = main_net(torch.tensor(obs, dtype=torch.float32))
                action = q_values.argmax().item()  # exploit

        next_obs, base_reward, done, truncated, info = env.step(action)
        done = done or truncated

        # Apply shaped reward (Zalish's function)
        reward = shaped_reward(obs, base_reward, done)
        total_reward += reward

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        # Only train once the buffer has enough samples
        if len(buffer) < BATCH_SIZE:
            continue

        # Sample a random batch and unpack it
        batch = buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.tensor(np.array(states),      dtype=torch.float32)
        actions     = torch.tensor(actions,               dtype=torch.long)
        rewards     = torch.tensor(rewards,               dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones       = torch.tensor(dones,                 dtype=torch.float32)

        # Current Q-values for the actions we actually took
        q_values = main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values — use target network, no gradient needed
        with torch.no_grad():
            max_next_q = target_net(next_states).max(1).values
            targets = rewards + GAMMA * max_next_q * (1 - dones)
            # (1 - dones) zeros out the future reward if the episode ended

        loss = loss_fn(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Decay epsilon after each episode
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Copy main network weights to target network every N episodes
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(main_net.state_dict())

    # Log progress
    print(f"Episode {episode+1:4d} | Reward: {total_reward:7.2f} | Epsilon: {epsilon:.3f}")

env.close()
