# Flappy Bird DQN

AI Decision Making Final Project — Reinforcement Learning

## Setup
pip install flappy-bird-gymnasium torch gymnasium

## Run
python3 test_env.py   # verify environment works first
python3 main.py       # start training

## Files
- main.py          — training loop (Luis)
- model.py         — neural network (Diego)
- replay_buffer.py — experience replay (Luis)
- reward.py        — shaped reward function (Zalish)
- utils.py         — logging and plotting (Luis)

## Hyperparameters
See the top of main.py to adjust episodes, batch size, epsilon, etc.
