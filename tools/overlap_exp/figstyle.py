"""NeurIPS camera-ready matplotlib style. Import this; do not re-derive per figure.

WHY EACH SETTING IS WHAT IT IS

  Type 42 fonts.  Type 3 (matplotlib's default for PDF) is rejected outright by many
  venues' checkers. This is the single most common camera-ready rejection.

  STIXGeneral.  NeurIPS sets its body in Times. This machine has no Times, Nimbus Roman
  or Liberation Serif -- but STIXGeneral ships with matplotlib and is Times-metric
  compatible, so figure type matches body type. Paired with mathtext.fontset='stix' so
  symbols come from the same family instead of falling back to DejaVu.

  5.5 in.  NeurIPS text width. Build the figure at the size it will print and never
  \\includegraphics[scale=...] it -- scaling in LaTeX breaks the match between figure
  type and body type, which is what makes a figure look bolted on.

  8 pt / 7 pt.  At 5.5 in and no scaling, 8 pt in the figure IS 8 pt on the page,
  against a 10 pt body. Any smaller stops being legible in print.

COLOUR.  The palette is validated, not chosen by eye: OKLab CVD delta-E 26.8 against a
target of 8.0, checked with the Machado-Oliveira-Fernandes 2009 severity-1.0 simulation
for both protanopia and deuteranopia, in light and dark surfaces. Hatching is applied on
top so the figure also survives greyscale printing -- colour alone is never the only
encoding.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

TEXT_W = 5.5          # NeurIPS \textwidth, inches
HALF_W = 2.68         # two side by side with a gutter

# validated palette -- see palette-validator-no-node
GEMM = "#C56604"
COMM = "#617BDC"
BASE = "#C9C4BB"      # static floor: a neutral, not a third peer hue
INK = "#1A1A19"
MUTED = "#6E6A63"
GRID = "#E2DED7"
HATCH = {"gemm": "///", "comm": "\\\\\\", "base": ""}


def use():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,          # never Type 3
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.0, "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 200, "savefig.dpi": 600,   # PNG is the deliverable; 600 dpi
        # at 5.5 in is 3300 px, enough for print. Re-enable the PDF line in save() if a
        # camera-ready ever needs vector -- the Type 42 settings above are still correct.
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
        "axes.prop_cycle": mpl.cycler(color=[GEMM, COMM, MUTED]),
        "hatch.linewidth": 0.5,
    })


def despine(ax, grid_axis="y"):
    """Recessive grid behind the marks, no box."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelcolor=INK)


def diverging():
    """Two-pole map with a NEUTRAL grey midpoint, for a signed quantity.

    Savings and losses are opposite polarities, not two categories and not a magnitude,
    so the form is diverging: the two validated hues at the poles and grey at zero. A
    sequential ramp here would imply "more is more" across a sign change, and a rainbow
    would imply an ordering the data does not have.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "sav", [GEMM, "#E9A464", "#F2F0EC", "#9AA6DF", COMM])


def matrix(ax, rows, cols, vals, fmt="{:+.0f}", vmin=None, vmax=None, cmap=None,
           fontsize=6.0):
    """A labelled cell grid: every value sits at a named row and a named column, so a
    reader can point at any cell and read off exactly which configuration it is.

    This exists because a 66-point scatter is unreadable at the level of the individual
    point -- the trend is visible but no one can say what any given dot represents. A
    matrix trades the continuous x-axis for addressability.
    """
    import numpy as np
    a = np.array(vals, dtype=float)
    if vmin is None:
        m = np.nanmax(np.abs(a)); vmin, vmax = -m, m
    im = ax.imshow(a, cmap=cmap or diverging(), vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols); ax.set_yticklabels(rows)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    # a hairline between cells, drawn in the surface colour so cells stay separate
    ax.set_xticks(np.arange(len(cols) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isnan(a[i, j]):
                ax.text(j, i, "--", ha="center", va="center", fontsize=fontsize,
                        color=MUTED)
                continue
            # ink colour follows the cell's lightness, not its hue
            t = abs(a[i, j]) / max(abs(vmin), abs(vmax))
            ax.text(j, i, fmt.format(a[i, j]), ha="center", va="center",
                    fontsize=fontsize, color="white" if t > 0.55 else INK)
    return im


def save(fig, stem, outdir="figs"):
    import os
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(png)
    plt.close(fig)
    return (png,)
