# obs indices (use_lidar=False, normalize_obs=True):
#   obs[0]  last pipe horizontal dist
#   obs[1]  last pipe top-pipe bottom y
#   obs[2]  last pipe bottom-pipe top y
#   obs[3]  next pipe horizontal dist  <-- pipe we must fly through
#   obs[4]  next pipe top-pipe bottom y
#   obs[5]  next pipe bottom-pipe top y
#   obs[6]  next-next pipe horizontal dist
#   obs[7]  next-next pipe top-pipe bottom y
#   obs[8]  next-next pipe bottom-pipe top y
#   obs[9]  bird y  (0=top, 1=bottom)
#   obs[10] bird vertical velocity  (negative=falling)
#   obs[11] bird rotation

import math, os

# Set REWARD_MODE env var to run a mode without editing this file: REWARD_MODE=4 python3 main.py
REWARD_MODE = int(os.environ.get('REWARD_MODE', 0))


def shaped_reward(obs, base_reward, done):
    if REWARD_MODE == 0:
        # Naive baseline: uses obs[1] (last pipe top edge, not gap center) — intentionally kept as-is
        if done:
            return -10.0
        bird_y           = obs[9]
        pipe_gap_y       = obs[1]
        distance_penalty = -abs(bird_y - pipe_gap_y) * 0.1
        survival_bonus   = 0.1
        return base_reward + survival_bonus + distance_penalty

    if done:
        return -10.0

    bird_y     = obs[9]
    gap_top    = obs[4]   # bottom of top pipe (next pipe)
    gap_bot    = obs[5]   # top of bottom pipe (next pipe)
    gap_center = (gap_top + gap_bot) / 2.0
    dist       = bird_y - gap_center   # positive = bird below center
    abs_dist   = abs(dist)

    if REWARD_MODE == 1:
        penalty  = -abs_dist * 0.5
        survival = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 2:
        penalty  = -abs_dist * 2.0
        survival = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 3:
        penalty  = -(dist ** 2) * 5.0
        survival = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 4:
        gaussian = math.exp(-8.0 * dist ** 2)
        return base_reward + gaussian * 0.5

    elif REWARD_MODE == 5:
        horiz_dist = obs[3]           # 0 = at pipe, 1 = far
        urgency    = 1.0 - horiz_dist
        penalty    = -abs_dist * urgency * 3.0
        survival   = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 6:
        alignment = -abs_dist * 1.0

        vel             = obs[10]   # negative=falling, positive=rising
        going_right_way = (dist > 0) == (vel < 0)
        vel_bonus       = 0.15 if going_right_way else -0.05

        pipe_bonus = 1.0 if base_reward >= 1.0 else 0.0
        survival   = 0.05

        return base_reward + survival + alignment + vel_bonus + pipe_bonus

    else:
        raise ValueError(f"Unknown REWARD_MODE: {REWARD_MODE}")
