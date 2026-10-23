"""Figure 1: how the measurement models connect.

    python3 make_fig1.py --out results/fig_pipeline.png
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, EDGE, SURF = "#0b0b0b", "#52514e", "#3a3a38", "#fcfcfb"
FILL = {"in": "#eef3fb", "phys": "#e8f6ef", "prob": "#fdf0e9", "meas": "#f3f2ef", "est": "#f1eefc"}

# name: (center x, center y, width, height, title, body)
BOXES = {
    "in":   (1.05, 2.0, 1.75, 1.15, "Inputs", "pose, targets,\nreflectivity, fog"),
    "phys": (3.35, 2.0, 1.95, 1.15, "Physics-based", r"range $r$, angle $\alpha$," + "\n" + r"signal $S$; FOV, LOS"),
    "prob": (5.75, 2.0, 1.95, 1.15, "Probabilistic", r"$P_{\mathrm{det}}$, $\sigma_{\mathrm{eff}}$," + "\n" + r"mixture $p(z)$"),
    "meas": (8.05, 2.0, 1.75, 1.15, "Measurements", r"scan $Z=\{z_i\}$"),
    "est":  (5.75, 0.45, 3.40, 0.85, "Estimation-theoretic", r"how precisely $Z$ locates the robot"),
}


def edge(name, side):
    x, y, w, h, *_ = BOXES[name]
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]


def arrow(ax, pts):
    """Polyline through pts with an arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=EDGE, lw=1.6, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.6, mutation_scale=14, shrinkA=0, shrinkB=0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_pipeline.png")
    a = ap.parse_args(argv)

    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    ax.set_xlim(0, 9.1)
    ax.set_ylim(-0.1, 2.75)
    ax.axis("off")
    fig.patch.set_facecolor(SURF)

    for key, (x, y, w, h, title, body) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=FILL[key], ec=EDGE, lw=1.2))
        ax.text(x, y + h * 0.20, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, y - h * 0.17, body, ha="center", va="center", fontsize=9.5, color=INK2, linespacing=1.3)

    gap = 0.03
    # main chain: inputs -> physics -> probabilistic -> measurements
    for a_, b_ in (("in", "phys"), ("phys", "prob"), ("prob", "meas")):
        (x0, y0), (x1, y1) = edge(a_, "r"), edge(b_, "l")
        arrow(ax, [(x0 + gap, y0), (x1 - gap, y1)])
    # probabilistic -> estimation (noise levels and weights)
    (x0, y0), (x1, y1) = edge("prob", "b"), edge("est", "t")
    arrow(ax, [(x0, y0 - gap), (x1, y1 + gap)])
    ax.text(x0 + 0.08, (y0 + y1) / 2, r"weights $w_{\mathrm{hit}}/\sigma_{\mathrm{eff}}^2$", fontsize=8.5,
            color=INK2, va="center")
    # physics -> estimation (geometry / sensitivity), elbow
    xp, yp = edge("phys", "b")
    xe, ye = edge("est", "l")
    arrow(ax, [(xp, yp - gap), (xp, ye), (xe - gap, ye)])
    ax.text(xp - 0.08, (yp + ye) / 2, "geometry", fontsize=8.5, color=INK2, va="center", ha="right")
    # measurements -> estimation (estimator), elbow
    xm, ym = edge("meas", "b")
    xr, yr = edge("est", "r")
    arrow(ax, [(xm, ym - gap), (xm, yr), (xr + gap, yr)])
    ax.text(xm + 0.08, (ym + yr) / 2, "estimator", fontsize=8.5, color=INK2, va="center", ha="left")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor=SURF)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURF)
    print("wrote", out, "and", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()