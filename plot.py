import re, numpy as np, matplotlib.pyplot as plt, sys

# Usage: python3 plot.py label1 file1.txt label2 file2.txt ...
# Example: python3 plot.py "Baseline" baseline.txt "Target1" target1.txt

def parse(path):
    episodes, rewards = [], []
    with open(path) as f:
        for line in f:
            m = re.match(r'Episode\s+(\d+) \| Reward:\s+([-\d.]+)', line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
    return np.array(episodes), np.array(rewards)

COLORS = ['#89b4fa', '#a6e3a1', '#f38ba8', '#fab387', '#cba6f7']
WINDOW = 20

args = sys.argv[1:]  # label file label file ...
pairs = [(args[i], args[i+1]) for i in range(0, len(args), 2)]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#1e1e2e')

for i, (label, path) in enumerate(pairs):
    color = COLORS[i % len(COLORS)]
    e, r = parse(path)
    ax.scatter(e, r, alpha=0.12, s=6, color=color)
    rolled = np.convolve(r, np.ones(WINDOW)/WINDOW, mode='valid')
    ax.plot(e[WINDOW-1:], rolled, color=color, linewidth=2.5, label=label)

ax.axhline(0, color='#6c7086', linewidth=1, linestyle='--')
ax.set_xlabel('Episode', color='#cdd6f4', fontsize=12)
ax.set_ylabel('Reward', color='#cdd6f4', fontsize=12)
ax.tick_params(colors='#cdd6f4')
for spine in ax.spines.values(): spine.set_edgecolor('#313244')
ax.legend(facecolor='#313244', labelcolor='#cdd6f4')

title = ' vs '.join(l for l, _ in pairs)
ax.set_title(title, color='#cdd6f4', fontsize=14, pad=12)

outname = '_vs_'.join(l.replace(' ', '_') for l, _ in pairs) + '.png'
plt.tight_layout()
plt.savefig(outname, dpi=150, bbox_inches='tight')
print(f"Saved: {outname}")