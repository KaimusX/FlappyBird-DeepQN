# reward.py - Shapes the default reward to speed up learning.
# Default reward is sparse (survive/crash only). We add a distance-based
# signal to guide the bird toward pipe gaps during early training.
# Zalish

def shaped_reward(obs, base_reward, done):
    if done:
        return -10.0

    bird_y = obs[9]
    pipe_gap_y = obs[1]

    # Penalize distance from pipe gap center — encourages centered flight
    distance_penalty = -abs(bird_y - pipe_gap_y) * 0.1

    # Small survival bonus each timestep to encourage staying alive
    survival_bonus = 0.1

    return base_reward + survival_bonus + distance_penalty