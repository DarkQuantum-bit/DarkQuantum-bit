"""Generate an illustrative ML training GIF with the neural network represented as a graph."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.datasets import make_moons

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'assets' / 'ml-training.gif'
PREVIEW = ROOT / 'assets' / 'ml-training-preview.png'

seed = 13
rng = np.random.default_rng(seed)
X, y = make_moons(n_samples=360, noise=0.18, random_state=seed)
X = (X - X.mean(0)) / X.std(0)
y = y[:, None].astype(float)

# MLP: 2 -> 8 -> 4 -> 1
params = {
    'W1': rng.normal(0, 0.50, (2, 8)), 'b1': np.zeros((1, 8)),
    'W2': rng.normal(0, 0.42, (8, 4)), 'b2': np.zeros((1, 4)),
    'W3': rng.normal(0, 0.38, (4, 1)), 'b3': np.zeros((1, 1)),
}
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def forward(inp):
    z1 = inp @ params['W1'] + params['b1']
    a1 = np.tanh(z1)
    z2 = a1 @ params['W2'] + params['b2']
    a2 = np.tanh(z2)
    z3 = a2 @ params['W3'] + params['b3']
    p = sigmoid(z3)
    return {'a1': a1, 'a2': a2, 'p': p}


def edge_style(weight, max_abs):
    mag = abs(weight) / (max_abs + 1e-9)
    if weight >= 0:
        color = (0.10, 0.86, 0.92, 0.12 + 0.80 * mag)
    else:
        color = (0.37, 0.53, 0.59, 0.08 + 0.42 * mag)
    lw = 0.35 + 2.2 * mag
    return color, lw


def draw_graph(ax, acts, epoch, loss):
    ax.set_facecolor('#0A1E27')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#20434D')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Neural network graph', color='#E8FDFF', fontsize=11, pad=10)

    layer_x = [0.08, 0.33, 0.62, 0.89]
    layer_sizes = [2, 8, 4, 1]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    positions = []
    for x, n in zip(layer_x, layer_sizes):
        ys = np.linspace(0.85, 0.15, n)
        positions.append([(x, yi) for yi in ys])

    Ws = [params['W1'], params['W2'], params['W3']]
    max_abs = max(np.max(np.abs(w)) for w in Ws)
    for l, W in enumerate(Ws):
        for i, (x1, y1) in enumerate(positions[l]):
            for j, (x2, y2) in enumerate(positions[l+1]):
                c, lw = edge_style(W[i, j], max_abs)
                ax.plot([x1, x2], [y1, y2], color=c, lw=lw, solid_capstyle='round', zorder=1)

    node_values = [
        np.mean(np.abs(X), axis=0),
        np.mean(np.abs(acts['a1']), axis=0),
        np.mean(np.abs(acts['a2']), axis=0),
        np.array([np.mean(acts['p'])]),
    ]

    for lx, name, vals, pts in zip(layer_x, layer_names, node_values, positions):
        ax.text(lx, 0.95, name, ha='center', va='center', color='#8FDDE5', fontsize=8)
        vmax = max(np.max(vals), 1e-6)
        for val, (x, y) in zip(vals, pts):
            alpha = 0.35 + 0.65 * (float(val) / vmax)
            size = 170 if name.startswith('Hidden') else 210
            fc = (0.06, 0.14, 0.18, 1.0)
            ec = (0.24, 0.90, 0.95, alpha)
            ax.scatter([x], [y], s=size, color=fc, edgecolors=[ec], linewidths=2.0, zorder=3)
            ax.scatter([x], [y], s=size*0.33, color=(0.35, 0.94, 0.97, alpha), edgecolors='none', zorder=4)

    ax.text(0.03, 0.02, f'epoch {epoch:03d}   loss {loss:.3f}', color='#B6E5EA', fontsize=9)
    ax.text(0.53, 0.02, 'edge intensity ≈ weight magnitude', color='#7DAEB5', fontsize=8, ha='center')


def draw_boundary(ax, acts, epoch):
    ax.set_facecolor('#0A1E27')
    for sp in ax.spines.values():
        sp.set_color('#20434D')
    ax.tick_params(colors='#95B6BE', labelsize=8)
    ax.set_title('Decision boundary', color='#E8FDFF', fontsize=11, pad=10)

    xs = np.linspace(-2.7, 2.7, 160)
    ys = np.linspace(-2.2, 2.2, 128)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    p = forward(grid)['p'].reshape(xx.shape)
    cmap = LinearSegmentedColormap.from_list('cyanish', [
        '#0A2732', '#123B4A', '#195564', '#217684', '#30A6AF', '#68D5DB', '#C6FBFD'
    ])
    ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 22), cmap=cmap, alpha=0.92)
    ax.contour(xx, yy, p, levels=[0.5], colors=['#F2FFFF'], linewidths=1.6)
    ax.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], s=14, c='#081A27', edgecolors='#4ADFEB', linewidths=0.55)
    ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=14, c='#E8FEFF', edgecolors='#99E6EE', linewidths=0.45)
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel('Feature 1', color='#AFCED4', fontsize=8)
    ax.set_ylabel('Feature 2', color='#AFCED4', fontsize=8)
    ax.text(0.98, 0.03, 'two-moons', transform=ax.transAxes, ha='right', color='#B8E6EA', fontsize=8)


def draw_loss(ax, losses, epoch, final_steps):
    ax.set_facecolor('#0A1E27')
    for sp in ax.spines.values():
        sp.set_color('#20434D')
    ax.tick_params(colors='#95B6BE', labelsize=8)
    ax.set_title('Training loss', color='#E8FDFF', fontsize=10, pad=8)
    ax.plot(range(len(losses)), losses, c='#47DDE8', lw=2)
    ax.scatter([len(losses)-1], [losses[-1]], c='#F0FEFF', s=24, zorder=3)
    ax.set_xlim(0, final_steps)
    ymax = max(0.75, max(losses) * 1.05)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Step', color='#AFCED4', fontsize=8)
    ax.set_ylabel('BCE', color='#AFCED4', fontsize=8)
    ax.grid(color='#274C57', alpha=0.28)
    ax.text(0.97, 0.86, f'{losses[-1]:.3f}', transform=ax.transAxes, ha='right', color='#A7EDF2', fontsize=10)

plt.rcParams.update({'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(11.2, 4.4), dpi=130)
fig.patch.set_facecolor('#07151E')
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.05], height_ratios=[1.0, 0.58],
                      left=0.045, right=0.985, bottom=0.12, top=0.95, wspace=0.18, hspace=0.24)
ax_graph = fig.add_subplot(gs[:, 0])
ax_bound = fig.add_subplot(gs[0, 1])
ax_loss = fig.add_subplot(gs[1, 1])

frames = []
losses = []
steps = 200
snapshots = [0, 1, 2, 4, 6, 9, 13, 18, 25, 35, 48, 65, 85, 110, 140, 170, 200]

for t in range(steps + 1):
    outs = forward(X)
    p = outs['p']
    eps = 1e-8
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    losses.append(loss)

    if t in snapshots:
        ax_graph.clear(); ax_bound.clear(); ax_loss.clear()
        draw_graph(ax_graph, outs, t, loss)
        draw_boundary(ax_bound, outs, t)
        draw_loss(ax_loss, losses, t, steps)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(frame.copy()).convert('P', palette=Image.Palette.ADAPTIVE, colors=180))
        if t == snapshots[0]:
            Image.fromarray(frame.copy()).save(PREVIEW)

    if t == steps:
        break

    a1, a2 = outs['a1'], outs['a2']
    dz3 = (p - y) / len(X)
    gW3 = a2.T @ dz3
    gb3 = dz3.sum(axis=0, keepdims=True)
    da2 = dz3 @ params['W3'].T
    dz2 = da2 * (1 - a2 * a2)
    gW2 = a1.T @ dz2
    gb2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ params['W2'].T
    dz1 = da1 * (1 - a1 * a1)
    gW1 = X.T @ dz1
    gb1 = dz1.sum(axis=0, keepdims=True)
    grads = {'W1': gW1, 'b1': gb1, 'W2': gW2, 'b2': gb2, 'W3': gW3, 'b3': gb3}

    lr = 0.025
    for k in params:
        m[k] = 0.9 * m[k] + 0.1 * grads[k]
        v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
        mh = m[k] / (1 - 0.9 ** (t + 1))
        vh = v[k] / (1 - 0.999 ** (t + 1))
        params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=frames[1:],
    duration=[900] + [260] * (len(frames) - 2) + [1600],
    loop=0,
    optimize=False,
    disposal=2,
)
print(f'{OUTPUT}: {OUTPUT.stat().st_size/1024:.1f} KiB, loss {losses[0]:.3f} -> {losses[-1]:.3f}')
