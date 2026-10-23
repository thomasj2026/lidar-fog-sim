"""
Produces figures + results/summary.json + results/summary.md:
  1 geometry check (raycast vs hand-computed ranges)
  2  signal chain curves: P_det, sigma_eff vs range / incidence / fog
  3  Monte Carlo per-beam statistics vs model predictions (dropout, sigma)
  4  fog mixture: empirical vs predicted component masses
  5  observability: Fisher-information rank, single wall vs corner
  6  estimator Monte Carlo: empirical covariance vs inverse Fisher information
  7  per-beam information vs incidence angle (optimal oblique band)
  8  CRB heat map over sensor position in the cluttered room
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .estimation import crb, gn_localize, information_matrix
from .model import SensorParams, Wall, mixture_weights, p_detect, p_observe, predict, raycast, received_signal, sample_scan
from .scene import load_scene

# Reference categorical palette (slots 1-3) and light-surface ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCENES = pathlib.Path(__file__).resolve().parents[1] / "config" / "scenes.yaml"
PRM = SensorParams()
TH = np.linspace(-np.pi, np.pi, 720)


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def logticks(ax, ticks):
    """Readable tick labels on a log axis (instead of 6x10^0)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def wall_at(D):
    """Wall x = D, free side -x, sensor at origin facing +x (beam angle = incidence angle)."""
    return [Wall([D, -60.0], [D, 60.0], 0.8)]


def save(fig, out, name):
    fig.patch.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)



def exp_geometry():
    walls = wall_at(4.0)
    deg = np.array([0, 15, 30, 45, 60, -30])
    r, _, a = raycast([0, 0], 0.0, np.deg2rad(deg), walls, PRM)
    hand = 4.0 / np.cos(np.deg2rad(deg))
    rows = [dict(theta_deg=int(d), hand=float(h), model=float(m), err=float(m - h)) for d, h, m in zip(deg, hand, r)]
    return dict(rows=rows, max_abs_err=float(np.abs(r - hand).max()))


def exp_signal(out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    # (a) P_det vs incidence at D = 1 m, three fog levels
    g = np.deg2rad(np.linspace(0, 85, 300))
    for beta, c in zip((0.0, 0.05, 0.1), (BLUE, ORANGE, AQUA)):
        p = [predict([0, 0, 0], wall_at(1.0), [x], beta, PRM).p_det[0] for x in g]
        ax[0].plot(np.rad2deg(g), p, color=c, lw=2, label=f"β = {beta:g} 1/m")
    style(ax[0], "incidence angle α [deg]  (wall at 1 m)", "P_det", "(a) detection vs grazing angle")
    ax[0].legend(frameon=False, fontsize=9)
    # (b) P_det vs range for bright / dark target
    r = np.linspace(0.5, 11.9, 300)
    for rho, c in ((0.8, BLUE), (0.1, ORANGE)):
        p = [predict([0, 0, 0], [Wall([x, -60], [x, 60], rho)], [0.0], 0.0, PRM).p_det[0] for x in r]
        ax[1].plot(r, p, color=c, lw=2, label=f"ρ = {rho:g}")
    style(ax[1], "range r [m]  (α = 0)", "P_det", "(b) detection vs range and reflectivity")
    ax[1].legend(frameon=False, fontsize=9)
    # (c) sigma_eff vs range for three incidence angles
    for al, c in zip((0, 45, 70), (BLUE, ORANGE, AQUA)):
        rr, ss = [], []
        for x in np.linspace(0.6, 11.5, 200):
            D = x * np.cos(np.deg2rad(al))
            pr = predict([0, 0, 0], wall_at(D), [np.deg2rad(al)], 0.0, PRM)
            if pr.idx[0] >= 0:
                rr.append(pr.r[0]); ss.append(pr.sigma_eff[0] * 100)
        ax[2].plot(rr, ss, color=c, lw=2, label=f"α = {al}°")
    ax[2].set_yscale("log")
    logticks(ax[2], [1, 2, 4, 8])
    style(ax[2], "range r [m]", "σ_eff [cm]", "(c) range uncertainty")
    ax[2].legend(frameon=False, fontsize=9)
    save(fig, out, "fig_signal_chain.png")


def exp_mc_per_beam(out, K):
    prm = PRM.with_(z_rand=0.0)
    rng = np.random.default_rng(10)
    res = {}
    # sweep alpha at D = 1 m
    al = np.deg2rad(np.linspace(0, 84, 29))
    emp_drop, pred_drop, emp_sd, pred_sd = [], [], [], []
    for a in al:
        pr = predict([0, 0, 0], wall_at(1.0), [a], 0.0, prm)
        z = sample_scan([0, 0, 0], wall_at(1.0), np.full(K, a), 0.0, prm, rng)
        d = z[np.isfinite(z)] - pr.r[0]
        emp_drop.append(np.isinf(z).mean()); pred_drop.append(1 - pr.p_det[0])
        emp_sd.append(d.std() if len(d) > 50 else np.nan); pred_sd.append(pr.sigma_eff[0])
    res["alpha_sweep_max_dropout_err"] = float(np.nanmax(np.abs(np.array(emp_drop) - pred_drop)))
    rel = np.abs(np.array(emp_sd) - pred_sd) / np.array(pred_sd)
    res["alpha_sweep_max_sigma_rel_err"] = float(np.nanmax(rel[np.array(pred_drop) < 0.5]))
    # sweep range at alpha = 0 (dark target for dropout, bright for sigma)
    rr = np.linspace(0.5, 11.8, 24)
    e_drop, p_drop, e_sd, p_sd, p_drop_bright = [], [], [], [], []
    for r in rr:
        dark = [Wall([r, -60], [r, 60], 0.1)]
        pr = predict([0, 0, 0], dark, [0.0], 0.0, prm)
        z = sample_scan([0, 0, 0], dark, np.zeros(K), 0.0, prm, rng)
        e_drop.append(np.isinf(z).mean()); p_drop.append(1 - pr.p_det[0])
        bright = wall_at(r)
        pr = predict([0, 0, 0], bright, [0.0], 0.0, prm)
        p_drop_bright.append(1 - pr.p_det[0])          # model only: no extra random draws
        z = sample_scan([0, 0, 0], bright, np.zeros(K), 0.0, prm, rng)
        e_sd.append(np.std(z[np.isfinite(z)] - pr.r[0])); p_sd.append(pr.sigma_eff[0])
    res["range_sweep_max_dropout_err"] = float(np.max(np.abs(np.array(e_drop) - p_drop)))
    res["range_sweep_max_sigma_rel_err"] = float(np.max(np.abs(np.array(e_sd) - p_sd) / np.array(p_sd)))

    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    A = np.rad2deg(al)
    ax[0, 0].plot(A, pred_drop, color=BLUE, lw=2, label="model 1 − P_det")
    ax[0, 0].plot(A, emp_drop, "o", color=ORANGE, ms=6, label="Monte Carlo")
    style(ax[0, 0], "incidence angle α [deg]  (wall at 1 m)", "dropout probability", "(a) dropout vs incidence")
    ax[0, 0].legend(frameon=False, fontsize=9)
    ax[0, 1].plot(A, np.array(pred_sd) * 100, color=BLUE, lw=2, label="model σ_eff")
    ax[0, 1].plot(A, np.array(emp_sd) * 100, "o", color=ORANGE, ms=6, label="Monte Carlo std")
    ax[0, 1].set_yscale("log")
    logticks(ax[0, 1], [1, 2, 5, 10, 20])
    style(ax[0, 1], "incidence angle α [deg]", "σ [cm]", "(b) range std vs incidence")
    ax[0, 1].legend(frameon=False, fontsize=9)
    ax[1, 0].plot(rr, p_drop, color=BLUE, lw=2, label="model, ρ = 0.1")
    ax[1, 0].plot(rr, e_drop, "o", color=ORANGE, ms=6, label="Monte Carlo, ρ = 0.1")
    ax[1, 0].plot(rr, p_drop_bright, color=AQUA, lw=2, label="model, ρ = 0.8")
    style(ax[1, 0], "range r [m]  (α = 0)", "dropout probability", "(c) dropout vs range and reflectivity")
    ax[1, 0].legend(frameon=False, fontsize=9)
    ax[1, 1].plot(rr, np.array(p_sd) * 100, color=BLUE, lw=2, label="model σ_eff")
    ax[1, 1].plot(rr, np.array(e_sd) * 100, "o", color=ORANGE, ms=6, label="Monte Carlo std")
    style(ax[1, 1], "range r [m]  (ρ = 0.8, α = 0)", "σ [cm]", "(d) range std vs range")
    ax[1, 1].legend(frameon=False, fontsize=9)
    save(fig, out, "fig_mc_per_beam.png")
    return res


def exp_fog(out, K):
    prm = PRM.with_(z_rand=0.0)
    rng = np.random.default_rng(11)
    betas = np.linspace(0, 0.25, 11)
    D = 4.0
    walls = wall_at(D)
    emp = np.zeros((len(betas), 3)); pred = np.zeros((len(betas), 3))
    for i, b in enumerate(betas):
        pr = predict([0, 0, 0], walls, [0.0], b, prm)
        z = sample_scan([0, 0, 0], walls, np.zeros(K), b, prm, rng)
        s = 5 * pr.sigma_eff[0]
        none = np.isinf(z)
        hit = np.isfinite(z) & (np.abs(z - pr.r[0]) < s)
        short = np.isfinite(z) & (z < pr.r[0] - s)
        emp[i] = [hit.mean(), short.mean(), none.mean()]
        pred[i] = [pr.w_hit[0], pr.w_short[0], pr.w_max[0]]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, (name, c) in enumerate((("target return", BLUE), ("fog (early) return", ORANGE), ("no return", AQUA))):
        ax.plot(betas, pred[:, k], color=c, lw=2, label=f"{name} (model)")
        ax.plot(betas, emp[:, k], "o", color=c, ms=6, mfc="none", mew=2)
    ax.plot([], [], "o", color=INK2, mfc="none", mew=2, ms=6, label="Monte Carlo (markers)")
    style(ax, "fog extinction β [1/m]  (wall at 4 m)", "probability", "Mixture masses: model vs Monte Carlo")
    ax.legend(frameon=False, fontsize=9)
    save(fig, out, "fig_fog_mixture.png")
    return dict(max_mass_err=float(np.abs(emp - pred).max()))


def exp_observability():
    out = {}
    for name in ("wall", "corner"):
        sc = load_scene(SCENES, name)
        I = information_matrix([0, 0, 0], sc.walls, TH, 0.0, PRM)
        C, rank, ev = crb(I)
        nullv = np.linalg.eigh(I)[1][:, 0]
        out[name] = dict(rank=rank, eigenvalues=[float(e) for e in ev],
                         weakest_direction=[float(v) for v in nullv])
    return out


def exp_crb_vs_mc(out, K):
    sc = load_scene(SCENES, "corner")
    x_true = np.array([0.0, 0.0, 0.0])
    betas = [0.0, 0.03, 0.06, 0.10]
    rng = np.random.default_rng(12)
    rows = []
    for b in betas:
        C = crb(information_matrix(x_true, sc.walls, TH, b, PRM))[0]
        sd_pred = np.sqrt(np.diag(C))
        xs = []
        for _ in range(K):
            z = sample_scan(x_true, sc.walls, TH, b, PRM, rng)
            x0 = x_true + rng.normal(0, [0.08, 0.08, 0.03])
            xs.append(gn_localize(z, TH, sc.walls, x0, PRM, b)[0])
        xs = np.array(xs) - x_true
        sd_emp = xs.std(axis=0, ddof=1)
        rows.append(dict(beta=b, sd_pred=sd_pred.tolist(), sd_emp=sd_emp.tolist(),
                         ratio=(sd_emp / sd_pred).tolist(), bias=xs.mean(axis=0).tolist()))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, (nm, scale, unit) in enumerate((("σ_x", 1e3, "mm"), ("σ_y", 1e3, "mm"), ("σ_ψ", 180 / np.pi * 1e3, "mdeg"))):
        ax[k].plot(betas, [r["sd_pred"][k] * scale for r in rows], color=BLUE, lw=2, label="inverse Fisher information")
        ax[k].plot(betas, [r["sd_emp"][k] * scale for r in rows], "o", color=ORANGE, ms=8, label=f"estimator Monte Carlo (K={K})")
        style(ax[k], "fog extinction β [1/m]", f"{nm} [{unit}]", nm)
        if k == 0:
            ax[k].legend(frameon=False, fontsize=8)
    save(fig, out, "fig_crb_vs_estimator.png")
    return rows


def exp_incidence_information(out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    res = {}
    g = np.deg2rad(np.linspace(0, 85, 171))
    for D, c in zip((1.0, 2.0, 4.0), (BLUE, ORANGE, AQUA)):
        walls = wall_at(D)
        I_D = []
        for x in g:
            I = information_matrix([0, 0, 0], walls, [x], 0.0, PRM)
            I_D.append(I[0, 0])
        I_D = np.array(I_D)
        # closed-form cross-check where the beam is in range
        pr = [predict([0, 0, 0], walls, [x], 0.0, PRM) for x in g]
        cf = np.array([(p.w_hit[0] / (np.cos(x) ** 2 * p.sigma_eff[0] ** 2)) if p.idx[0] >= 0 else 0.0 for p, x in zip(pr, g)])
        assert np.allclose(I_D, cf, rtol=1e-9, atol=1e-9), "closed form disagrees with information_matrix"
        k = int(np.argmax(I_D))
        res[f"D={D:g}"] = dict(gamma_opt_deg=float(np.rad2deg(g[k])), gain_vs_normal=float(I_D[k] / I_D[0]))
        ax.plot(np.rad2deg(g), I_D / I_D[0], color=c, lw=2, label=f"D = {D:g} m")
        ax.plot(np.rad2deg(g[k]), I_D[k] / I_D[0], "o", color=c, ms=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    logticks(ax, [1e-3, 1e-2, 1e-1, 1])
    style(ax, "incidence angle γ [deg]", "I_D(γ) / I_D(0)", "Information about perpendicular distance, per beam")
    ax.legend(frameon=False, fontsize=9)
    save(fig, out, "fig_incidence_information.png")
    return res


def exp_crb_map(out, n):
    sc = load_scene(SCENES, "clutter")
    xs = np.linspace(-4.6, 4.6, n)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    summary = {}
    maps = []
    for beta in (0.0, 0.08):
        M = np.full((n, n), np.nan)
        for i, y in enumerate(xs):
            for j, x in enumerate(xs):
                if not sc.is_free([x, y], 0.3):
                    continue
                C = crb(information_matrix([x, y, 0.0], sc.walls, TH, beta, PRM))[0]
                M[i, j] = np.sqrt(C[0, 0] + C[1, 1]) * 1e3
        maps.append(M)
        summary[f"beta={beta:g}"] = dict(median_mm=float(np.nanmedian(M)), max_mm=float(np.nanmax(M)))
    vmax = np.nanmax(maps)
    for k, (M, beta) in enumerate(zip(maps, (0.0, 0.08))):
        im = ax[k].imshow(M, origin="lower", extent=[xs[0], xs[-1], xs[0], xs[-1]], cmap="Blues", vmin=0, vmax=vmax)
        for b in sc.boxes:
            from .scene import box_walls
            pts = np.array([w.a for w in box_walls(b["center"], b["size"], b.get("yaw", 0.0), 1.0)] + [box_walls(b["center"], b["size"], b.get("yaw", 0.0), 1.0)[0].a])
            ax[k].plot(pts[:, 0], pts[:, 1], color=INK, lw=1.5)
        style(ax[k], "x [m]", "y [m]", f"Position CRB, √(σ_x²+σ_y²) [mm], β = {beta:g}")
        ax[k].grid(False)
        fig.colorbar(im, ax=ax[k], shrink=0.85)
    save(fig, out, "fig_crb_map.png")
    return summary


def exp_scan_figure(out):
    sc = load_scene(SCENES, "clutter")
    rng = np.random.default_rng(3)
    x = np.array([-0.5, -1.0, 0.0])
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for w in sc.walls:
        ax.plot([w.a[0], w.b[0]], [w.a[1], w.b[1]], color=INK, lw=1.5)
    for beta, c, lab in ((0.1, ORANGE, "fog β = 0.1 1/m"), (0.0, BLUE, "clear air β = 0")):
        z = sample_scan(x, sc.walls, TH, beta, PRM, rng)
        ok = np.isfinite(z)
        a = x[2] + TH[ok]
        ax.plot(x[0] + z[ok] * np.cos(a), x[1] + z[ok] * np.sin(a), ".", color=c, ms=4, label=lab)
    ax.plot(*x[:2], "^", color=INK, ms=10)
    ax.set_aspect("equal")
    style(ax, "x [m]", "y [m]", "Simulated scan, cluttered room (dark box: ρ = 0.15)")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, markerscale=2.5)
    save(fig, out, "fig_scan_clutter.png")


def exp_fov(out, res=0.05):
    """Field of view + line of sight + detection for a dark point target (rho = 0.1),
    270-degree scanner, two poses, clear air vs fog.  Deterministic (no random draws)."""
    sc = load_scene(SCENES, "clutter")
    fov = (-np.deg2rad(135.0), np.deg2rad(135.0))
    poses = {"A": np.array([-3.5, -3.5, np.pi / 4]), "B": np.array([0.0, -1.0, np.pi / 2])}
    betas = (0.0, 0.1)
    xs = np.arange(-5 + res / 2, 5, res)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    free = np.array([sc.is_free(q, 0.0) for q in pts])
    cell = res * res
    res_out = {}
    fig, ax = plt.subplots(2, 2, figsize=(9.4, 8.8))
    for i, (name, x) in enumerate(poses.items()):
        d = {"pose": [float(v) for v in x]}
        for j, b in enumerate(betas):
            P, f, l = p_observe(pts, x, sc.walls, b, PRM, 0.1, fov, return_parts=True)
            d["area_free_m2"] = float(free.sum() * cell)
            d["area_fov_m2"] = float((f & free).sum() * cell)
            d["area_fov_los_m2"] = float((f & l & free).sum() * cell)
            d[f"area_obs_beta={b:g}_m2"] = float(((P >= 0.5) & free).sum() * cell)
            M = np.where(free, P, np.nan).reshape(X.shape)
            a = ax[i, j]
            im = a.imshow(M, origin="lower", extent=[-5, 5, -5, 5], cmap="Blues", vmin=0, vmax=1)
            for w in sc.walls:
                a.plot([w.a[0], w.b[0]], [w.a[1], w.b[1]], color=INK, lw=1.4)
            for e in fov:                                          # FOV edges
                a.plot([x[0], x[0] + 1.6 * np.cos(x[2] + e)], [x[1], x[1] + 1.6 * np.sin(x[2] + e)], color=ORANGE, lw=1.5, ls="--")
            a.annotate("", xy=(x[0] + 0.9 * np.cos(x[2]), x[1] + 0.9 * np.sin(x[2])), xytext=(x[0], x[1]),
                       arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
            a.plot(*x[:2], "o", color=ORANGE, ms=6)
            style(a, "x [m]", "y [m]", f"pose {name}, β = {b:g} 1/m")
            a.grid(False)
            a.set_xlim(-5.2, 5.2); a.set_ylim(-5.2, 5.2); a.set_aspect("equal")
        res_out[f"pose{name}"] = d
    fig.colorbar(im, ax=ax, shrink=0.6, label="P_obs (dark target, ρ = 0.1)")
    fig.patch.set_facecolor(SURF)
    fig.savefig(out / "fig_fov.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # effective range: P_obs(r) = 0.5 along an unobstructed ray (bisection)
    def r_half(beta):
        f = lambda r: (1 - PRM.z_rand) * np.exp(-PRM.kappa * beta * r) * p_detect(received_signal(r, 0.0, 0.1, beta, PRM), PRM) - 0.5
        lo, hi = 0.2, PRM.r_max
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)
    res_out["r_half_m"] = {"beta=0": float(r_half(0.0)), "beta=0.1": float(r_half(0.1))}
    return res_out



def md_summary(S):
    L = ["Summary", ""]
    L += ["## A. Geometry (raycast vs hand-computed)", f"max |error| = {S['geometry']['max_abs_err']:.2e} m", ""]
    L += ["## C. Per-beam Monte Carlo vs model"] + [f"- {k}: {v:.4f}" for k, v in S["mc_per_beam"].items()] + [""]
    L += ["## D. Fog mixture", f"max |empirical − predicted| mass = {S['fog']['max_mass_err']:.4f}", ""]
    L += ["## E. Observability (rank of Fisher information, state = (x, y, psi))"]
    for k, v in S["observability"].items():
        L.append(f"- {k}: rank {v['rank']}, eigenvalues {['%.3g' % e for e in v['eigenvalues']]}, weakest direction {['%.3f' % e for e in v['weakest_direction']]}")
    L += ["", "## F. Estimator Monte Carlo vs inverse Fisher information (corner scene)",
          "| beta | sd_x pred / emp [mm] | sd_y pred / emp [mm] | sd_psi pred / emp [mdeg] | ratios (emp/pred) |", "|---|---|---|---|---|"]
    for r in S["crb_vs_mc"]:
        p, e = np.array(r["sd_pred"]), np.array(r["sd_emp"])
        L.append(f"| {r['beta']:g} | {p[0]*1e3:.2f} / {e[0]*1e3:.2f} | {p[1]*1e3:.2f} / {e[1]*1e3:.2f} | "
                 f"{np.rad2deg(p[2])*1e3:.2f} / {np.rad2deg(e[2])*1e3:.2f} | {', '.join('%.2f' % q for q in r['ratio'])} |")
    L += ["", "## G. Optimal incidence for distance information"] + [f"- {k}: {v}" for k, v in S["incidence_information"].items()]
    L += ["", "## H. CRB map (clutter)"] + [f"- {k}: {v}" for k, v in S["crb_map"].items()]
    if "fov" in S:
        L += ["", "## I. FOV / LOS / detection coverage (270 deg, rho = 0.1)"] + [f"- {k}: {v}" for k, v in S["fov"].items()]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo draws")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    K, Kest, n = (4000, 60, 21) if a.quick else (40000, 300, 41)
    S = {}
    S["geometry"] = exp_geometry()
    exp_signal(out)
    S["mc_per_beam"] = exp_mc_per_beam(out, K)
    S["fog"] = exp_fog(out, K)
    S["observability"] = exp_observability()
    S["crb_vs_mc"] = exp_crb_vs_mc(out, Kest)
    S["incidence_information"] = exp_incidence_information(out)
    S["crb_map"] = exp_crb_map(out, n)
    exp_scan_figure(out)
    S["fov"] = exp_fov(out)
    (out / "summary.json").write_text(json.dumps(S, indent=2))
    (out / "summary.md").write_text(md_summary(S))
    print(md_summary(S))


if __name__ == "__main__":
    main()
