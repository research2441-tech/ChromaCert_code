"""Fig. 1: overall ChromaCert framework (grayscale, no overlapping blocks)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.2,
                     'mathtext.fontset': 'dejavuserif'})
fig, ax = plt.subplots(figsize=(7.16, 4.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

BOX = {}


def box(key, x, y, w, h, title, body, fill='white', lw=0.9, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fill, ec='black', lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 1.4, title, ha='center', va='top', fontsize=6.3, fontweight='bold' if bold else 'normal')
    if body:
        ax.text(x + w / 2, y + h - 4.4, body, ha='center', va='top', fontsize=5.5, linespacing=1.3)
    BOX[key] = (x, y, w, h)


def side(key, s, f=0.5):
    x, y, w, h = BOX[key]
    return {'l': (x - 0.3, y + h * f), 'r': (x + w + 0.3, y + h * f), 't': (x + w * f, y + h + 0.3),
            'b': (x + w * f, y - 0.3)}[s]


def arrow(p, q, dashed=False, label=None, lpos=0.5, loff=(0, 1.2), cs='arc3,rad=0', lw=0.9):
    a = FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=lw, color='black',
                        linestyle=(0, (4, 2.5)) if dashed else '-', connectionstyle=cs, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        mx, my = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=5.4,
                bbox=dict(fc='white', ec='none', pad=0.6))


# ---- inference path (top row) ------------------------------------------------
G1, G2 = '#f2f2f2', '#dedede'
X = [0.4, 20.6, 40.8, 61.0, 81.2]
W = 16.0
box('in', X[0], 36, W, 17, 'Input design', 'text, shapes,\nphotos, background\n+ locked brand\ncolours $\\mathcal{L}$', fill=G1)
box('graph', X[1], 36, W, 17, 'Element graph', 'nodes: role, box,\nz-order, colour\nstats, CLIP vector\nedges: on, under,\nadjacent, same role', fill=G1)
box('net', X[2], 36, W, 17, 'Relation-biased\nTransformer $f_\\theta$', '\n$L$ biased blocks,\nhead bias $b_{r}$\n+ overlap $\\kappa\\, o_{ij}$', fill=G2)
box('mdd', X[3], 36, W, 17, 'Masked discrete\ndiffusion', '\nOKLab codebook,\n$K=256$ tokens,\n$S$ unmasking steps', fill=G2)
box('acc', X[4], 36, W, 17, 'Accessibility\ncertification', '\nOKLCH repair;\nescalate layers\nwith $s_i<\\hat\\lambda$', fill=G1)

arrow(side('in', 'r'), side('graph', 'l'))
arrow(side('graph', 'r'), side('net', 'l'))
arrow(side('net', 'r', 0.66), side('mdd', 'l', 0.66), label='$p_\\theta$', loff=(0, 1.9))
arrow(side('mdd', 'l', 0.34), side('net', 'r', 0.34), label='$c^{(s)}$', loff=(0, -1.8))
arrow(side('mdd', 'r'), side('acc', 'l'), label='$\\hat c$', loff=(0, 1.7))
arrow(side('acc', 't'), (X[4] + W / 2, 56.2))
ax.text(X[4] + W / 2, 57.6, 'certified colouring', ha='center', va='center', fontsize=6.3, style='italic')

# ---- guidance ------------------------------------------------------------------
box('guide', X[3], 17.6, W, 13, 'Harmony\nguidance', '\nhue template,\nCIEDE2000 gap,\nlightness prior', fill='white')
arrow(side('guide', 't'), side('mdd', 'b'), label='$-\\gamma E_i(k)$', loff=(4.0, 0))

# ---- calibration ---------------------------------------------------------------
box('cal', X[4], 17.6, W, 13, 'Conformal risk\ncontrol', '\npixel audit over\n16 observers', fill='white')
arrow(side('cal', 't'), side('acc', 'b'), label='$\\hat\\lambda$', loff=(2.6, 0))
box('calset', X[4], 1.5, W, 11.5, 'Calibration\ndesigns', '\n$n$ held-out\ntemplates', fill=G1)
arrow(side('calset', 't'), side('cal', 'b'))

# ---- training (bottom row) ---------------------------------------------------
box('data', X[0], 1.5, W, 13, 'Designer\ncolourings', '\ntraining\ntemplates', fill=G1)
box('elbo', X[1], 1.5, W, 13, 'Masked-diffusion\nELBO $\\mathcal{L}_{\\mathrm{MDM}}$', '\nrandom locks,\nmask rate $t$', fill='white')
box('dpo', X[2], 1.5, W, 13, 'Preference loss\n$\\mathcal{L}_{\\mathrm{PREF}}$', '\ndesigner $\\succ$ sample,\nfrozen $f_{\\mathrm{ref}}$', fill='white')
box('pairs', X[3], 1.5, W, 11.5, 'Preference pairs', '$f_{\\mathrm{ref}}$ samples with\nmean $\\Delta E_{00}\\geq 5$', fill='white')
arrow(side('data', 'r'), side('elbo', 'l'))
arrow(side('pairs', 'l', 0.5), side('dpo', 'r', 0.44))
arrow(side('data', 't', 0.5), side('in', 'b', 0.5), label='templates', loff=(4.0, 0))
# dashed reverse-mode gradient arrows into f_theta
arrow(side('elbo', 't', 0.62), side('net', 'b', 0.22), dashed=True, cs='arc3,rad=-0.18',
      label='$\\nabla_\\theta \\mathcal{L}_{\\mathrm{MDM}}$', lpos=0.45, loff=(-6.2, 0.4))
arrow(side('dpo', 't', 0.55), side('net', 'b', 0.55), dashed=True,
      label='$\\nabla_\\theta \\mathcal{L}_{\\mathrm{PREF}}$', lpos=0.5, loff=(5.6, 0))

# legend for line styles (kept clear of every block)
ax.plot([20.75, 25.25], [-1.6, -1.6], color='black', lw=0.9)
ax.text(26.0, -1.6, 'forward / inference flow', va='center', fontsize=6.3)
ax.plot([48.0, 52.5], [-1.6, -1.6], color='black', lw=0.9, linestyle=(0, (4, 2.5)))
ax.text(53.3, -1.6, 'reverse-mode gradient (training only)', va='center', fontsize=6.3)
ax.set_ylim(-3.4, 59.4)
fig.savefig('figs/fig1_framework.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
print('ok')
