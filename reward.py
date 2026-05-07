# reward.py - Shaped reward function for Flappy Bird DQN.
# Zalish — switch REWARD_MODE to try different formulas.
#
# Observation index reference (use_lidar=False, normalize_obs=True):
#   obs[0]  last pipe horizontal dist (normalized 0-1)
#   obs[1]  last pipe top-pipe bottom y
#   obs[2]  last pipe bottom-pipe top y
#   obs[3]  NEXT pipe horizontal dist  <-- the pipe we must fly through
#   obs[4]  NEXT pipe top-pipe bottom y
#   obs[5]  NEXT pipe bottom-pipe top y
#   obs[6]  next-next pipe horizontal dist
#   obs[7]  next-next pipe top-pipe bottom y
#   obs[8]  next-next pipe bottom-pipe top y
#   obs[9]  bird y (0 = top, 1 = bottom, normalized)
#   obs[10] bird vertical velocity (normalized, negative = falling)
#   obs[11] bird rotation (normalized)
#
# Base reward from the env: alive = +0.1 per step, pipe passed = +1.0, dead = -1.0
#
# ── How to run an experiment and compare with plot.py ────────────────────────
# 1. Set REWARD_MODE below to the mode you want to test.
# 2. Redirect output to a txt file:
#       python3 main.py > mode0.txt
# 3. Change REWARD_MODE, run again with a different file:
#       python3 main.py > mode1.txt
# 4. Plot both together:
#       python3 plot.py "Mode 0 (Original)" mode0.txt "Mode 1 (Fixed)" mode1.txt
#    Add as many label/file pairs as you want.
# ─────────────────────────────────────────────────────────────────────────────

import math, os

# Read REWARD_MODE from environment variable if set, otherwise default to 0.
# Run a specific mode without editing this file:
#   REWARD_MODE=4 python3 main.py > mode4.txt
REWARD_MODE = int(os.environ.get('REWARD_MODE', 0))


def shaped_reward(obs, base_reward, done):
    if REWARD_MODE == 0:
        # ── Mode 0: Naive Last-Pipe Baseline ─────────────────────────────────
        # Original formula written by Zalish. Uses obs[1] as the gap reference
        # (last pipe's top edge — not the true gap center) and a light scale of
        # 0.1. Kept exactly as written so every other mode can be compared to it.
        if done:
            return -10.0
        bird_y          = obs[9]
        pipe_gap_y      = obs[1]
        distance_penalty = -abs(bird_y - pipe_gap_y) * 0.1
        survival_bonus  = 0.1
        return base_reward + survival_bonus + distance_penalty

    if done:
        return -10.0

    bird_y    = obs[9]
    gap_top   = obs[4]   # bottom of the top pipe (next pipe)
    gap_bot   = obs[5]   # top of the bottom pipe (next pipe)
    gap_center = (gap_top + gap_bot) / 2.0
    dist      = bird_y - gap_center          # signed: + = bird below center
    abs_dist  = abs(dist)

    if REWARD_MODE == 1:
        # ── Mode 1: Fixed baseline ───────────────────────────────────────────
        # Same idea as the original but using the CORRECT next-pipe gap center
        # instead of obs[1] (which was the last pipe's top edge — a bug).
        # Scale 0.5 is stronger than the old 0.1 to give a clearer gradient.
        penalty  = -abs_dist * 0.5
        survival = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 2:
        # ── Mode 2: Strong linear penalty ────────────────────────────────────
        # Crank up the distance scale to 2.0. The agent feels a big pull
        # toward the gap. Risk: if the scale is too high it can overshadow the
        # pipe-passage reward (+1) and the agent learns to hover but not advance.
        penalty  = -abs_dist * 2.0
        survival = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 3:
        # ── Mode 3: Quadratic (harsh when far, gentle when close) ────────────
        # Penalizes large deviations much harder than small ones.
        # Good if the agent keeps drifting to extremes.
        penalty  = -(dist ** 2) * 5.0
        survival = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 4:
        # ── Mode 4: Gaussian bell curve ──────────────────────────────────────
        # Reward peaks at +0.5 when perfectly centered, decays to ~0 when far.
        # No explicit penalty — the agent just gets less reward when off-center.
        # Smoother gradient than modes 1-3; may need more episodes to converge.
        gaussian = math.exp(-8.0 * dist ** 2)
        return base_reward + gaussian * 0.5

    elif REWARD_MODE == 5:
        # ── Mode 5: Proximity-weighted (urgency near pipe) ───────────────────
        # The penalty grows stronger as the bird approaches the next pipe.
        # When far away (obs[3] ≈ 1) the urgency is low.
        # When at the pipe (obs[3] ≈ 0) the urgency spikes.
        horiz_dist = obs[3]                    # 0 = at pipe, 1 = far
        urgency    = 1.0 - horiz_dist          # 0 when far, 1 when at pipe
        penalty    = -abs_dist * urgency * 3.0
        survival   = 0.1
        return base_reward + survival + penalty

    elif REWARD_MODE == 6:
        # ── Mode 6: Multi-component (alignment + velocity + pipe bonus) ──────
        # Three separate signals stacked together:
        #   1. alignment:  linear distance penalty toward gap center
        #   2. vel_bonus:  small reward when velocity is in the right direction
        #   3. pipe_bonus: extra +1 on top of base reward for clearing a pipe
        # Most informative signal; most tuning knobs. Start here after modes 1-4.
        alignment  = -abs_dist * 1.0

        vel        = obs[10]                        # negative = falling, positive = rising
        going_right_way = (dist > 0) == (vel < 0)
        vel_bonus  = 0.15 if going_right_way else -0.05

        pipe_bonus = 1.0 if base_reward >= 1.0 else 0.0  # doubled reward on pipe pass
        survival   = 0.05

        return base_reward + survival + alignment + vel_bonus + pipe_bonus

    else:
        raise ValueError(f"Unknown REWARD_MODE: {REWARD_MODE}")


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD EXPERIMENT PLAN — read this before changing REWARD_MODE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Goal: find the formula that teaches the agent to reliably pass pipes fastest.
# Run each mode for 500–1000 episodes and compare:
#   - How many episodes until the agent first clears a pipe?
#   - What is the avg reward at episode 500?
#   - Does the agent get "stuck" (hover without advancing)?
#
# ┌──────┬────────────────────────────────┬───────────────────────────────────┐
# │ Mode │ What it tests                  │ What to watch for                 │
# ├──────┼────────────────────────────────┼───────────────────────────────────┤
# │  0   │ Naive Last-Pipe Baseline        │ Baseline reference. Every other   │
# │      │ obs[1], scale 0.1              │ mode should be compared to this.  │
# ├──────┼────────────────────────────────┼───────────────────────────────────┤
# │  1   │ Fixed baseline                 │ Sanity check — should beat mode 0 │
# │      │ scale 0.5 linear               │ since obs indices are now correct  │
# ├──────┼────────────────────────────────┼───────────────────────────────────┤
# │  2   │ Stronger linear scale (2.0)    │ Does stronger = faster learning?  │
# │      │                                │ Watch for hovering-but-not-moving │
# ├──────┼────────────────────────────────┼───────────────────────────────────┤
# │  3   │ Quadratic penalty              │ Fewer extreme crashes? If yes,    │
# │      │                                │ quadratic is worth keeping.       │
# ├──────┼────────────────────────────────┼───────────────────────────────────┤
# │  4   │ Gaussian (no explicit penalty) │ Smoother learning curve? Usually  │
# │      │                                │ needs more episodes but is stable. │
# ├──────┼────────────────────────────────┼───────────────────────────────────┤
# │  5   │ Proximity-weighted urgency     │ Does the agent line up earlier    │
# │      │                                │ and then hold position near pipe? │
# ├──────┼────────────────────────────────┼───────────────────────────────────┤
# │  6   │ Multi-component (best guess)   │ Try this last — most signals but  │
# │      │                                │ also most interaction effects.    │
# └──────┴────────────────────────────────┴───────────────────────────────────┘
#
# Suggested run order: 0 → 1 → 2 → 4 → 6
#
#   python3 main.py > mode0.txt   # baseline
#   python3 main.py > mode1.txt   # fix obs index — change REWARD_MODE first
#   python3 main.py > mode2.txt
#   python3 main.py > mode4.txt
#   python3 main.py > mode6.txt
#
#   python3 plot.py "Original" mode0.txt "Fixed" mode1.txt "Strong" mode2.txt
#
# ═══════════════════════════════════════════════════════════════════════════════
