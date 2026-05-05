# Test file to see you can run it.
import flappy_bird_gymnasium
import gymnasium as gym

env = gym.make("FlappyBird-v0", use_lidar=False, render_mode="human")
obs, _ = env.reset()

print("Observation shape:", obs.shape)
print("First observation:", obs)

for step in range(200):
    action = env.action_space.sample()  # random actions
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        print(f"Episode ended at step {step}, resetting...")
        obs, _ = env.reset()

env.close()
print("Environment works!")