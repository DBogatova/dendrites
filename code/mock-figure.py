import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ===== Mock data (clustered, adjustable if you want to drop in real points) =====
np.random.seed(0)
nmda_x = np.random.normal(2.1, 0.28, 18)
nmda_y = np.random.normal(20.0, 4.8, 18)
ca_x   = np.random.normal(4.0, 0.25, 18)
ca_y   = np.random.normal(40.0, 3.8, 18)

# ===== Helper: tight enclosing ellipse from mean/cov with scale ensuring all points are inside =====
def enclosing_ellipse(xs, ys, edgecolor, ax, margin=0.05, lw=2.5):
    X = np.vstack([xs, ys]).T
    mu = X.mean(axis=0)
    C = np.cov(X, rowvar=False)
    # Numerical stability
    C += np.eye(2) * 1e-6
    invC = np.linalg.inv(C)

    # Mahalanobis distances
    d2 = np.array([((v - mu) @ invC @ (v - mu)) for v in X])
    s = np.sqrt(d2.max()) * (1 + margin)  # scale so that max point is just inside with a margin

    # Ellipse params from eigendecomp of covariance
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

    width  = 2 * s * np.sqrt(evals[0])
    height = 2 * s * np.sqrt(evals[1])

    ell = mpatches.Ellipse(xy=mu, width=width, height=height, angle=angle,
                           fill=False, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(ell)


# ===== Helper: draw a more natural-looking dendrite with taper + side branches =====
def draw_dendrite(ax, x0, y0=0.0, height=1.2, trunk_color='red', branch_color='gold'):
    # Trunk: slight lateral meander
    n = 18
    ys = np.linspace(y0, y0 + height, n)
    drift = np.cumsum(np.random.normal(0, 0.01, n))
    xs = x0 + drift
    base_lw = 6.0
    for i in range(n - 1):
        # linear taper
        lw = base_lw * (1 - 0.65 * (i / (n - 2)))
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=trunk_color, lw=lw, solid_capstyle='round')

    # Branch origins near the top
    n_br = np.random.randint(2, 4)
    top_y = y0 + height * 0.78
    idx_top = np.searchsorted(ys, top_y)
    x_top, y_top = xs[idx_top], ys[idx_top]

    angles = np.linspace(-0.9, 0.9, n_br) + np.random.normal(0, 0.12, n_br)
    lengths = np.linspace(0.35, 0.55, n_br) + np.random.normal(0, 0.05, n_br)

    for a, L in zip(angles, lengths):
        # Primary branch
        bx = np.linspace(x_top, x_top + L * np.sin(a), 8)
        by = np.linspace(y_top, y_top + L * np.cos(a), 8)
        for i in range(len(bx) - 1):
            lw = 3.0 * (1 - 0.7 * (i / (len(bx) - 2)))
            ax.plot([bx[i], bx[i+1]], [by[i], by[i+1]], color=branch_color, lw=lw, solid_capstyle='round')
        # Small twig at the branch tip
        tipx, tipy = bx[-1], by[-1]
        twig_angle = a + np.random.choice([-0.6, 0.6]) + np.random.normal(0, 0.1)
        twig_len = L * 0.35
        tx = np.linspace(tipx, tipx + twig_len * np.sin(twig_angle), 5)
        ty = np.linspace(tipy, tipy + twig_len * np.cos(twig_angle), 5)
        for i in range(len(tx) - 1):
            lw = 2.1 * (1 - 0.7 * (i / (len(tx) - 2)))
            ax.plot([tx[i], tx[i+1]], [ty[i], ty[i+1]], color=branch_color, lw=lw, solid_capstyle='round')


# ===== Figure =====
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Left: scatter + enclosing ellipses that fully encompass points
ax[0].scatter(nmda_x, nmda_y, color='gold', edgecolor='black', s=38, zorder=3, label='NMDA Spikes')
ax[0].scatter(ca_x,   ca_y,   color='red',  edgecolor='black', s=38, zorder=3, label='Ca$^{2+}$ Transients')

enclosing_ellipse(nmda_x, nmda_y, edgecolor='gold', ax=ax[0], margin=0.10, lw=2.8)
enclosing_ellipse(ca_x,   ca_y,   edgecolor='red',  ax=ax[0], margin=0.10, lw=2.8)

ax[0].set_xlabel('Duration')
ax[0].set_ylabel('# pixels')
ax[0].set_xlim(0.8, 5.2)
ax[0].set_ylim(6, 46)
ax[0].legend(frameon=False, loc='upper left')

# Right: more realistic dendrites (tapered trunks with branching)
x_positions = [0.0, 0.9, 1.8, 2.7]
for xp in x_positions:
    draw_dendrite(ax[1], x0=xp, height=1.35)

ax[1].set_xlim(-0.5, 3.2)
ax[1].set_ylim(-0.05, 1.6)
ax[1].axis('off')

plt.tight_layout()

# Save the figure
plt.savefig('mock_figure.pdf', dpi=300, bbox_inches='tight')
print("Figure saved as mock_figure.pdf")

# Auto-open the figure on macOS
import subprocess
import os
try:
    if os.system('which open > /dev/null 2>&1') == 0:  # macOS
        subprocess.run(['open', 'mock_figure.pdf'])
        print("Opening figure with default PDF viewer...")
    else:
        print("Auto-open not available. Check mock_figure.pdf manually.")
except:
    print("Could not auto-open. Check mock_figure.pdf manually.")

plt.close()  # Close to free memory
