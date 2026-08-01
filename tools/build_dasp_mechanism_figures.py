from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs_confirmation_v6_10scenario"
FIGURES = ROOT / "ACCESS_latex_template_20260513" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#0b5f91"
DARK = "#243b53"
MID = "#627d98"
RED = "#b83a3a"
GREEN = "#2f855a"
PALE_BLUE = "#eaf3fb"
PALE_GOLD = "#fff4dc"
PALE_GREEN = "#eaf7ef"
PALE_RED = "#fcebea"

INK = "#202020"
LINE = "#4a4a4a"
GRAY_1 = "#f5f5f5"
GRAY_2 = "#e7e7e7"
GRAY_3 = "#d2d2d2"
TITLE_BLUE = "#cfe5f5"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def box(ax, x, y, w, h, text, fc=PALE_BLUE, ec=DARK, fs=8, lw=1.0, radius=0.06):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}", facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=DARK)
    return patch


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.1, style="-|>", rad=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def diamond(ax, x, y, w, h, text, fc=PALE_GOLD, ec=RED, fs=8, lw=1.0):
    patch = plt.Polygon(
        [[x + w / 2, y + h], [x + w, y + h / 2], [x + w / 2, y], [x, y + h / 2]],
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=DARK)
    return patch


def save(fig, name):
    fig.savefig(FIGURES / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_bundle(fig, stem):
    for suffix in ("pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", pad_inches=0.04, dpi=600)
    plt.close(fig)


def academic_box(ax, x, y, w, h, text, fc=GRAY_1, fs=7.0, rounded=False, lw=0.9, weight="semibold"):
    if rounded:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.16",
            facecolor=fc,
            edgecolor=LINE,
            linewidth=lw,
        )
    else:
        patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=LINE, linewidth=lw)
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        fontweight=weight,
        linespacing=1.12,
    )
    return patch


def academic_titled_box(ax, x, y, w, h, title, body, fc=GRAY_1, title_fs=9.2, body_fs=9.0, lw=0.9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.16",
        facecolor=fc,
        edgecolor=LINE,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.20,
        title,
        ha="center",
        va="center",
        fontsize=title_fs,
        color=INK,
        fontweight="semibold",
    )
    ax.text(
        x + w / 2,
        y + (h - 0.27) / 2,
        body,
        ha="center",
        va="center",
        fontsize=body_fs,
        color=INK,
        fontweight="normal",
        linespacing=1.08,
    )
    return patch


def academic_ellipse(ax, x, y, w, h, text, fs=7.2):
    patch = Ellipse((x + w / 2, y + h / 2), w, h, facecolor="white", edgecolor=LINE, linewidth=0.95)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold")
    return patch


def academic_diamond(ax, x, y, w, h, text, fs=6.8):
    patch = plt.Polygon(
        [[x + w / 2, y + h], [x + w, y + h / 2], [x + w / 2, y], [x, y + h / 2]],
        facecolor=GRAY_2,
        edgecolor=LINE,
        linewidth=0.95,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        fontweight="bold",
        linespacing=1.05,
    )
    return patch


def poly_arrow(ax, points, label=None, label_xy=None, dashed=False, lw=0.9):
    xs, ys = zip(*points)
    linestyle = (0, (3.0, 2.4)) if dashed else "-"
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=LINE, linewidth=lw, linestyle=linestyle, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            linestyle=linestyle,
            color=LINE,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=6.3, color=INK, fontweight="bold")


def mechanism_overview():
    fig, ax = plt.subplots(figsize=(4.3, 8.2))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 24.6)
    ax.axis("off")

    academic_ellipse(ax, 3.75, 23.45, 2.50, 0.72, "Start", 7.4)
    academic_box(ax, 2.15, 21.95, 5.70, 0.88, "Receive network-flow window $W_t$\nand delayed labels $Y_{t-1}$", GRAY_1, 7.0)
    academic_box(ax, 2.15, 20.50, 5.70, 0.88, "Prediction-first modular inference\nand prequential output $\\hat{y}_t$", GRAY_1, 7.0)
    academic_box(ax, 2.15, 19.05, 5.70, 0.88, "Extract four module representations\n$\\{h_{j,t}\\}_{j=1}^{4}$", GRAY_1, 7.0)
    academic_box(ax, 2.15, 17.60, 5.70, 0.88, "Module-wise multi-scale MMD\nand robust median/MAD normalization", GRAY_1, 6.8)
    academic_diamond(ax, 3.15, 15.75, 3.70, 1.30, "MMD severity\ntriggered?", 6.9)

    academic_box(ax, 0.35, 13.55, 3.00, 1.25, "Gradient conflict $c_{j,t}$\n$p_{j,t}=r_{j,t}/(1+\\lambda_c c_{j,t})$\nTop-$k$ positive routing", GRAY_2, 6.25)
    academic_box(ax, 0.35, 11.85, 3.00, 0.95, "Triggered repair\n$E=2$", GRAY_3, 6.7, rounded=True)

    academic_box(ax, 4.05, 14.00, 3.65, 0.86, "Evaluate guard-split\nMacro-F1 $G_t$", GRAY_1, 6.7)
    academic_diamond(ax, 4.50, 12.05, 2.75, 1.30, "$G_t<0.80$?", 6.8)
    academic_box(ax, 3.65, 10.25, 2.75, 1.05, "Emergency repair\nconflict top-$k$, $E=3$", GRAY_3, 6.25, rounded=True)
    academic_box(ax, 6.85, 10.25, 2.80, 1.05, "Background consolidation\nconflict top-$k$, $p=0.25$", GRAY_2, 5.9, rounded=True)

    academic_box(ax, 2.15, 8.55, 5.70, 0.90, "Class-balanced rehearsal\ncurrent labels + bounded replay", GRAY_1, 6.8)
    academic_box(ax, 2.15, 7.10, 5.70, 0.90, "Construct strict optimizer mask\nactive modules $\\mathcal{A}_t$ + shared classifier", GRAY_1, 6.65)
    academic_box(ax, 2.15, 5.65, 5.70, 0.90, "Apply localized update\ninactive modules remain byte-identical", GRAY_1, 6.65)
    academic_box(ax, 2.15, 4.20, 5.70, 0.90, "Update bounded memory, reference bank,\nrobust statistics, and audit trace", GRAY_1, 6.55)
    academic_ellipse(ax, 3.35, 2.75, 3.30, 0.78, "End / next window", 7.0)

    for y_top, y_bottom in [(23.45, 22.83), (21.95, 21.38), (20.50, 19.93), (19.05, 18.48), (17.60, 17.05)]:
        poly_arrow(ax, [(5.0, y_top), (5.0, y_bottom)])

    poly_arrow(ax, [(3.15, 16.40), (1.85, 16.40), (1.85, 14.80)], "Yes", (2.48, 16.62))
    poly_arrow(ax, [(5.0, 15.75), (5.0, 14.86)], "No", (5.32, 15.25))
    poly_arrow(ax, [(1.85, 13.55), (1.85, 12.80)])
    poly_arrow(ax, [(5.88, 14.00), (5.88, 13.35)])
    poly_arrow(ax, [(4.50, 12.70), (5.02, 12.70), (5.02, 11.30)], "Yes", (4.18, 12.90))
    poly_arrow(ax, [(7.25, 12.70), (8.25, 12.70), (8.25, 11.30)], "No", (7.74, 12.90))

    poly_arrow(ax, [(1.85, 11.85), (1.85, 9.82), (5.0, 9.82), (5.0, 9.45)])
    poly_arrow(ax, [(5.02, 10.25), (5.02, 9.45)])
    poly_arrow(ax, [(8.25, 10.25), (8.25, 9.82), (5.0, 9.82), (5.0, 9.45)])
    for y_top, y_bottom in [(8.55, 8.00), (7.10, 6.55), (5.65, 5.10), (4.20, 3.53)]:
        poly_arrow(ax, [(5.0, y_top), (5.0, y_bottom)])

    ax.text(
        5.0,
        1.80,
        "Invariant: $|\\mathcal{A}_t|\\leq k$ and no all-module fallback.",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
        fontstyle="italic",
    )
    save(fig, "dasp_mechanism_overview.pdf")


def routing_mechanics():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    modules = ["M1", "M2", "M3", "M4"]
    requirement = np.array([0.18, 0.82, 0.46, 0.70])
    conflict = np.array([0.10, 0.22, 0.61, 0.18])
    plasticity = requirement / (1.0 + conflict)
    y = np.arange(4)
    ax.barh(y + 0.16, requirement, height=0.28, color="#8fb6d6", label="MMD requirement $r_{j,t}$")
    ax.barh(y - 0.16, conflict, height=0.28, color="#e7a56b", label="gradient conflict $c_{j,t}$")
    for yi, pi in zip(y, plasticity):
        ax.text(1.01, yi, f"p={pi:.2f}", va="center", fontsize=7, color=RED, fontweight="bold")
    ax.set_yticks(y, modules); ax.set_xlim(0, 1.17); ax.invert_yaxis(); ax.set_xlabel("Normalized value")
    ax.set_title("Conflict-aware plasticity before top-$k$", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.2); ax.set_axisbelow(True)
    ax2 = axes[1]
    ax2.axis("off")
    ax2.text(0.5, 0.92, "Top-$k$ routing mask", ha="center", fontsize=9, fontweight="bold", color=BLUE)
    active_indices = set(np.argsort(plasticity)[-3:])
    for i, (m, p) in enumerate(zip(modules, plasticity)):
        yy = 0.74 - i * 0.16
        active = i in active_indices
        fc = PALE_GREEN if active else "#f2f2f2"
        ec = GREEN if active else MID
        box(ax2, 0.12, yy, 0.43, 0.09, f"{m}  {p:.2f}", fc, ec, 7, 0.8, 0.02)
        ax2.text(0.68, yy + 0.045, "OPEN" if active else "FROZEN", ha="center", va="center", fontsize=7, color=GREEN if active else MID, fontweight="bold")
    ax2.text(0.5, 0.10, "$p_{j,t}=r_{j,t}/(1+\\lambda c_{j,t})$\n$k=3$ is an upper bound.", ha="center", va="center", fontsize=7.5, color=DARK)
    fig.tight_layout(pad=0.6)
    save(fig, "dasp_routing_mechanics.pdf")


def timeline():
    fig, ax = plt.subplots(figsize=(9.2, 2.7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.text(0.1, 2.92, "Prediction-first delayed-label timeline", fontsize=11, fontweight="bold", color=BLUE)
    for i, label in enumerate(["$W_{t-1}$", "$W_t$", "$W_{t+1}$", "$W_{t+2}$"]):
        x = 0.8 + i * 2.45
        ax.plot([x, x], [0.65, 2.45], color=MID, lw=1.0)
        ax.text(x, 2.57, label, ha="center", fontsize=9, fontweight="bold")
    ax.annotate("predict", (0.8, 1.95), (2.0, 1.95), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2), fontsize=8, color=BLUE)
    ax.annotate("labels for $W_{t-1}$ arrive", (3.25, 1.45), (1.0, 1.45), arrowprops=dict(arrowstyle="->", color=RED, lw=1.2), fontsize=8, color=RED)
    box(ax, 2.9, 0.82, 1.45, 0.45, r"query $\leq 128$ labels", PALE_GOLD, RED, 7.5, 0.8, 0.03)
    box(ax, 5.35, 0.82, 1.45, 0.45, "attribute + repair", PALE_GREEN, GREEN, 7.5, 0.8, 0.03)
    box(ax, 7.8, 0.82, 1.45, 0.45, "mode gate + update", PALE_GREEN, GREEN, 7.5, 0.8, 0.03)
    arrow(ax, 4.4, 1.04, 5.3, 1.04, GREEN)
    arrow(ax, 6.85, 1.04, 7.75, 1.04, GREEN)
    ax.text(0.8, 0.32, "No label from $W_t$ is used to produce prediction on $W_t$.", fontsize=7.5, color=DARK)
    save(fig, "dasp_prediction_first_timeline.pdf")


def guard_flow():
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.set_xlim(0, 8.2); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(0.1, 3.94, "Performance-gated local repair modes in modular DASP", fontsize=11, fontweight="bold", color=BLUE)
    box(ax, 0.25, 2.85, 1.45, 0.58, "MMD severity\n$s_t$", PALE_GOLD, RED, 8)
    diamond(ax, 2.05, 2.70, 1.45, 0.88, "drift\ntrigger?", PALE_GOLD, RED, 8)
    box(ax, 4.05, 2.85, 1.55, 0.58, "Triggered repair\n$p=r/(1+\\lambda c)$", PALE_GREEN, GREEN, 7.6)
    arrow(ax, 1.75, 3.14, 2.00, 3.14, RED)
    arrow(ax, 3.52, 3.14, 4.00, 3.14, GREEN)
    ax.text(3.68, 3.28, "yes", fontsize=7.5, color=GREEN)

    box(ax, 0.25, 1.35, 1.55, 0.58, "Guard-split\nMacro-F1", PALE_BLUE, BLUE, 8)
    diamond(ax, 2.05, 1.20, 1.45, 0.88, "below\n0.80?", PALE_GOLD, RED, 8)
    box(ax, 4.05, 1.70, 1.55, 0.58, "Emergency repair\nconflict top-$k$, $E=3$", PALE_RED, RED, 7.4)
    box(ax, 4.05, 0.58, 1.55, 0.58, "Background consolidation\nconflict top-$k$, $p=0.25$", PALE_GREEN, GREEN, 7.0)
    arrow(ax, 2.78, 2.68, 1.05, 1.98, RED, 0.9, rad=0.0)
    ax.text(1.85, 2.30, "no", fontsize=7.5, color=RED)
    arrow(ax, 1.82, 1.64, 2.00, 1.64, BLUE)
    arrow(ax, 3.52, 1.64, 4.00, 1.99, RED)
    arrow(ax, 3.52, 1.64, 4.00, 0.87, GREEN)
    ax.text(3.70, 1.92, "yes", fontsize=7.5, color=RED)
    ax.text(3.70, 0.98, "no", fontsize=7.5, color=GREEN)

    box(ax, 6.25, 1.75, 1.65, 0.72, "Strict mask + replay\nupdate routed modules\nand classifier", PALE_GREEN, GREEN, 7.4)
    arrow(ax, 5.63, 3.14, 7.05, 2.50, GREEN, 0.9, rad=0.0)
    arrow(ax, 5.63, 1.99, 6.20, 2.05, GREEN, 0.9)
    arrow(ax, 5.63, 0.87, 6.45, 1.70, GREEN, 0.9)
    ax.text(0.25, 0.15, "The guard split selects emergency versus background behavior; the current modular_dasp path applies the localized update directly.", fontsize=7.2, color=DARK)
    save(fig, "dasp_guard_flow.pdf")


def _window_algorithm_code_draft():
    fig, ax = plt.subplots(figsize=(12.4, 8.3))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 8.3)
    ax.axis("off")

    ax.text(6.2, 8.04, "Implemented execution path of modular_dasp", ha="center", va="center", fontsize=12.0, fontweight="bold", color=INK)
    ax.text(6.2, 7.78, "Mapped from StreamingExperiment.run() and _update() in src/dasp_nids/runner.py", ha="center", va="center", fontsize=6.8, color=LINE, fontstyle="italic")

    lane_specs = [
        (0.25, 5.45, 11.90, 2.10, "A. Arrival of current window $W_t$: drift scoring and prediction before adaptation", "#f7f7f7"),
        (0.25, 1.35, 11.90, 3.90, "B. One-window-delayed update for queued item.window", "#fbfbfb"),
        (0.25, 0.18, 11.90, 0.98, "C. State transition after direct localized repair", "#f7f7f7"),
    ]
    for x, y, w, h, label, fill in lane_specs:
        lane = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fill, edgecolor=LINE, linewidth=0.9)
        ax.add_patch(lane)
        ax.text(x + 0.16, y + h - 0.18, label, ha="left", va="center", fontsize=7.1, fontweight="bold", color=INK)

    academic_box(
        ax,
        0.55,
        5.82,
        2.35,
        1.18,
        "Warm-up: 4 windows\n_train_online on train split\nmemory.add(full windows)\ncopy reference_model; calibrate MMD",
        GRAY_2,
        6.0,
        rounded=True,
    )
    academic_box(ax, 3.25, 6.25, 1.55, 0.58, "Current unlabeled\nwindow $W_t$", GRAY_3, 6.8, rounded=True)
    academic_box(ax, 5.20, 6.55, 2.25, 0.62, "model($W_t$)\n$\\hat y_t$ + prequential metrics", GRAY_1, 6.5, rounded=True)
    academic_box(ax, 5.20, 5.72, 2.25, 0.62, "reference_model($W_t$)\nfour module representations", GRAY_1, 6.35, rounded=True)
    academic_box(
        ax,
        7.82,
        5.62,
        2.15,
        0.76,
        "LayerDriftMonitor.evaluate\nMMD $\\rightarrow z_j \\rightarrow r_j$\nseverity = mean top-3\ntrigger: severity $>0.15$, persistence = 1",
        GRAY_2,
        5.75,
        rounded=True,
    )
    academic_box(
        ax,
        8.45,
        6.62,
        2.65,
        0.64,
        "PendingUpdate queue\nstores window, DriftDecision, prediction\npop when label_delay = 1",
        GRAY_2,
        5.95,
        rounded=True,
    )

    poly_arrow(ax, [(4.80, 6.54), (5.20, 6.86)])
    poly_arrow(ax, [(4.80, 6.54), (5.20, 6.03)])
    poly_arrow(ax, [(7.45, 6.86), (8.45, 6.94)])
    poly_arrow(ax, [(7.45, 6.03), (7.82, 6.03)])
    poly_arrow(ax, [(9.98, 6.38), (9.98, 6.62)])
    poly_arrow(ax, [(2.90, 6.41), (3.25, 6.54)], dashed=True, lw=0.8)
    poly_arrow(ax, [(2.90, 6.14), (5.20, 6.03)], dashed=True, lw=0.8)

    academic_box(ax, 3.20, 4.45, 2.65, 0.58, "Pop $W_{t-1}$, stored DriftDecision,\nand newly revealed labels", GRAY_3, 6.25, rounded=True)
    academic_box(
        ax,
        0.55,
        3.55,
        2.20,
        0.68,
        "_split_train_guard: 75/25\nclass-balanced train budget = 128\nguard holdout retained",
        GRAY_1,
        6.05,
        rounded=True,
    )
    academic_box(
        ax,
        0.55,
        2.38,
        2.20,
        0.78,
        "ReservoirMemory, capacity 1024\nprobe sample $\\leq128$ for conflict\nbalanced replay $\\leq512$ for training",
        GRAY_1,
        5.9,
        rounded=True,
    )
    academic_box(
        ax,
        3.15,
        3.55,
        2.15,
        0.68,
        "_gradient_conflicts\ncurrent-train vs memory-probe gradients\n$c_j=\\max(0,-\\cos(g_j^{new},g_j^{mem}))$",
        GRAY_2,
        5.85,
        rounded=True,
    )
    academic_diamond(ax, 5.70, 3.48, 1.75, 0.98, "stored MMD\ntriggered?", 6.4)
    academic_box(
        ax,
        7.85,
        4.28,
        3.55,
        0.66,
        "Triggered repair: $p_j=r_j/(1+c_j)$\ntop-3 positive $p_j$; fallback to max module; classifier = 1; $E=2$",
        GRAY_3,
        6.05,
        rounded=True,
    )
    academic_diamond(ax, 5.75, 2.23, 1.65, 0.92, "guard Macro-F1\n$<0.80$?", 6.2)
    academic_box(
        ax,
        7.85,
        3.22,
        3.55,
        0.66,
        "semi_local_repair: top-3 conflict modules\n$p_j=\\min(1,\\max(0.25,c_j))$; classifier = 1; $E=3$",
        GRAY_3,
        5.95,
        rounded=True,
    )
    academic_box(
        ax,
        7.85,
        2.18,
        3.55,
        0.66,
        "background_local_consolidation: top-3 conflict modules\n$p_j=0.25$; classifier = 1; $E=2$",
        GRAY_2,
        5.85,
        rounded=True,
    )
    academic_box(
        ax,
        3.15,
        1.42,
        8.25,
        0.66,
        "_localized_rehearsal_batch: current weight 3 + balanced replay weight 1\n_train_dasp: strict_freeze=True; use_projection_memory=False\nAdamW includes only active modules and classifier; direct update, no candidate commit/rollback",
        GRAY_1,
        5.55,
        rounded=True,
    )

    poly_arrow(ax, [(9.78, 6.62), (9.78, 5.35), (4.53, 5.35), (4.53, 5.03)])
    poly_arrow(ax, [(4.53, 4.45), (4.53, 4.30), (1.65, 4.30), (1.65, 4.23)])
    poly_arrow(ax, [(2.75, 3.89), (3.15, 3.89)])
    poly_arrow(ax, [(1.65, 3.55), (1.65, 3.16)])
    poly_arrow(ax, [(2.75, 2.77), (3.95, 2.77), (3.95, 3.55)])
    poly_arrow(ax, [(5.30, 3.89), (5.70, 3.97)])
    poly_arrow(ax, [(7.45, 3.97), (7.65, 4.61), (7.85, 4.61)], "Yes", (7.58, 4.40))
    poly_arrow(ax, [(6.58, 3.48), (6.58, 3.15)], "No", (6.88, 3.34))
    poly_arrow(ax, [(2.75, 3.70), (5.15, 3.70), (5.15, 2.69), (5.75, 2.69)], dashed=True, lw=0.8)
    poly_arrow(ax, [(7.40, 2.69), (7.62, 3.55), (7.85, 3.55)], "Yes", (7.56, 3.25))
    poly_arrow(ax, [(7.40, 2.69), (7.62, 2.51), (7.85, 2.51)], "No", (7.58, 2.30))

    ax.plot([9.625, 9.625], [4.28, 4.12], color=LINE, linewidth=0.85)
    ax.plot([9.625, 9.625], [3.22, 3.06], color=LINE, linewidth=0.85)
    ax.plot([9.625, 9.625], [2.18, 2.08], color=LINE, linewidth=0.85)
    ax.plot([9.625, 11.60], [4.12, 4.12], color=LINE, linewidth=0.85)
    ax.plot([9.625, 11.60], [3.06, 3.06], color=LINE, linewidth=0.85)
    ax.plot([9.625, 11.60], [2.08, 2.08], color=LINE, linewidth=0.85)
    ax.plot([11.60, 11.60], [2.08, 4.12], color=LINE, linewidth=0.85)
    poly_arrow(ax, [(11.60, 2.08), (11.60, 1.78), (11.40, 1.78)])

    academic_box(ax, 0.55, 0.35, 3.20, 0.56, "memory.add(train subset)\nreservoir capacity remains 1024", GRAY_1, 5.9, rounded=True)
    academic_box(ax, 4.15, 0.35, 3.55, 0.56, "reference_model $\\leftarrow$ updated model\nrecalibrate on recent 4 windows; reset monitor", GRAY_1, 5.65, rounded=True)
    academic_box(ax, 8.10, 0.35, 3.50, 0.56, "log $r_j,c_j,p_j$, active modules, gradients\nand global_update = False", GRAY_1, 5.75, rounded=True)
    poly_arrow(ax, [(7.28, 1.42), (7.28, 1.18), (2.15, 1.18), (2.15, 0.91)])
    poly_arrow(ax, [(7.28, 1.42), (7.28, 0.91)])
    poly_arrow(ax, [(7.28, 1.42), (9.85, 1.18), (9.85, 0.91)])
    poly_arrow(ax, [(7.70, 0.67), (8.10, 0.67)])
    poly_arrow(ax, [(11.60, 0.67), (12.00, 0.67), (12.00, 6.54), (11.10, 6.54)], dashed=True, lw=0.8)
    ax.text(11.88, 3.62, "next stream window", rotation=90, ha="center", va="center", fontsize=5.9, color=INK)
    plt.close(fig)


def _phase_container(ax, y, h, label, width=11.64):
    patch = FancyBboxPatch(
        (0.18, y),
        width,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor="#fbfbfb",
        edgecolor=LINE,
        linewidth=0.85,
    )
    ax.add_patch(patch)
    ax.text(0.36, y + h - 0.22, label, ha="left", va="center", fontsize=9.2, fontweight="bold", color=INK)


def _state_box(ax, x, y, w, h, text, fs=8.6):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="white",
        edgecolor=LINE,
        linewidth=0.8,
        linestyle=(0, (4, 2.5)),
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.08)
    return patch


def window_algorithm():
    fig, ax = plt.subplots(figsize=(7.25, 7.65))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 15.2)
    ax.axis("off")

    ax.text(7.0, 14.95, "Delayed-label DASP architecture", ha="center", va="center", fontsize=11.0, fontweight="bold", color=INK)
    ax.plot([0.55, 1.25], [14.58, 14.58], color=LINE, linewidth=1.0)
    ax.add_patch(FancyArrowPatch((1.02, 14.58), (1.25, 14.58), arrowstyle="-|>", mutation_scale=8, color=LINE, linewidth=1.0))
    ax.text(1.42, 14.58, "current data / processing", ha="left", va="center", fontsize=8.8, color=INK)
    ax.plot([7.15, 7.85], [14.58, 14.58], color=LINE, linewidth=0.9, linestyle=(0, (4, 2.5)))
    ax.add_patch(FancyArrowPatch((7.62, 14.58), (7.85, 14.58), arrowstyle="-|>", mutation_scale=8, color=LINE, linewidth=0.9, linestyle=(0, (4, 2.5))))
    ax.text(8.02, 14.58, "delayed feedback / persistent state", ha="left", va="center", fontsize=8.8, color=INK)

    _phase_container(ax, 11.35, 3.00, "A. Pre-adaptation inference at time $t$ (labels for $W_t$ are unavailable)", width=13.62)
    _phase_container(ax, 3.35, 7.75, "B. Delayed-label localized adaptation at time $t+1$", width=13.62)
    _phase_container(ax, 0.20, 2.90, "C. State and memory update", width=13.62)

    academic_titled_box(ax, 0.55, 12.62, 1.85, 0.82, "Current", "window $W_t$", GRAY_3)
    academic_titled_box(ax, 2.70, 12.62, 1.95, 0.82, "Active model", "prediction $\\hat y_t$", GRAY_1)
    academic_titled_box(ax, 4.95, 12.62, 2.10, 0.82, "Reference", "representations", GRAY_1)
    academic_titled_box(ax, 7.35, 12.44, 3.25, 1.18, "Layer-wise monitoring", "current/reference feature drift\nstore prediction for\ndelayed scoring", GRAY_2, title_fs=9.0, body_fs=8.8)
    academic_titled_box(ax, 10.90, 12.62, 2.35, 0.82, "Enqueue", "current state", GRAY_3)
    for x1, x2 in [(2.40, 2.70), (4.65, 4.95), (7.05, 7.35), (10.60, 10.90)]:
        poly_arrow(ax, [(x1, 13.03), (x2, 13.03)], lw=0.9)

    _state_box(ax, 5.00, 11.72, 2.00, 0.48, "Reference snapshot $\\mathcal{R}_t$", fs=8.7)
    _state_box(ax, 11.05, 11.72, 2.05, 0.48, "Pending-update queue", fs=8.7)
    poly_arrow(ax, [(6.00, 12.20), (6.00, 12.62)], dashed=True, lw=0.8)
    poly_arrow(ax, [(12.08, 12.62), (12.08, 12.20)], dashed=True, lw=0.8)

    academic_titled_box(ax, 4.20, 9.80, 5.60, 0.84, "Delayed labels $Y_t$ arrive at $t+1$", "restore stored decision/state; split train and guard", GRAY_3, title_fs=9.2, body_fs=8.9)
    poly_arrow(ax, [(12.08, 11.72), (12.08, 10.22), (9.80, 10.22)], dashed=True, lw=0.8)

    academic_titled_box(ax, 0.65, 8.25, 3.80, 0.98, "Stored drift evidence", "module requirements $r_j$\nseverity $s_t$", GRAY_1)
    academic_titled_box(ax, 5.00, 8.25, 3.40, 0.98, "Performance evidence", "guard Macro-F1 vs $\\epsilon_{F1}$", GRAY_1)
    academic_titled_box(ax, 8.80, 8.10, 4.55, 1.28, "Module-wise gradient conflict", "current-window vs replay gradients\n$C_j=\\max\\{0,-\\cos(g^{cur}_j,g^{rep}_j)\\}$\nhigh $C_j$: greater forgetting risk", GRAY_2, title_fs=8.9, body_fs=8.6)
    _state_box(ax, 10.20, 7.55, 2.80, 0.40, "Reservoir memory $\\mathcal{M}_t$", fs=8.3)
    poly_arrow(ax, [(11.60, 7.95), (11.60, 8.10)], dashed=True, lw=0.8)

    ax.plot([2.55, 11.12], [9.55, 9.55], color=LINE, linewidth=0.8)
    ax.plot([7.00, 7.00], [9.86, 9.55], color=LINE, linewidth=0.8)
    for center_x, top_y in ((2.55, 9.23), (6.70, 9.23), (11.08, 9.38)):
        poly_arrow(ax, [(center_x, 9.55), (center_x, top_y)], lw=0.8)

    ax.plot([2.55, 11.08], [7.88, 7.88], color=LINE, linewidth=0.8)
    for center_x, bottom_y in ((2.55, 8.25), (6.70, 8.25), (11.08, 8.10)):
        ax.plot([center_x, center_x], [bottom_y, 7.88], color=LINE, linewidth=0.8)
    poly_arrow(ax, [(2.35, 7.88), (2.35, 7.46)], lw=0.9)

    academic_titled_box(
        ax,
        2.00,
        6.30,
        10.00,
        1.16,
        "Repair-policy decision tree",
        "confirmed drift ($s_t>\\theta_s$) $\\rightarrow$ Triggered repair\nperformance loss without confirmed drift $\\rightarrow$ Semi-local repair\notherwise $\\rightarrow$ Background consolidation",
        GRAY_1,
        title_fs=9.2,
        body_fs=8.8,
    )

    academic_titled_box(ax, 0.55, 4.92, 4.10, 1.18, "Triggered repair", "stronger adaptation of\ndrift-attributed modules\nselect up to $k_p$ modules", GRAY_3, body_fs=8.9)
    academic_titled_box(ax, 4.95, 4.92, 4.10, 1.18, "Semi-local repair", "moderate repair of\nhigh-conflict modules\nselect up to $k_p$ modules", GRAY_3, body_fs=8.9)
    academic_titled_box(ax, 9.35, 4.92, 4.10, 1.18, "Background consolidation", "light rehearsal\nfor retention\nsmall plasticity", GRAY_2, body_fs=8.9)
    ax.plot([2.60, 11.40], [6.20, 6.20], color=LINE, linewidth=0.8)
    ax.plot([7.00, 7.00], [6.30, 6.20], color=LINE, linewidth=0.8)
    for center_x in (2.60, 7.00, 11.40):
        poly_arrow(ax, [(center_x, 6.20), (center_x, 6.10)], lw=0.8)

    academic_titled_box(ax, 1.05, 3.65, 5.30, 0.88, "Localized rehearsal batch", "current samples + balanced replay", GRAY_1)
    academic_titled_box(ax, 7.60, 3.52, 5.30, 1.08, "Selective optimizer update", "update active modules; freeze unaffected\nno candidate rollback gate in current code", GRAY_1, body_fs=8.8)
    for center_x in (2.60, 7.00, 11.40):
        ax.plot([center_x, center_x], [4.92, 4.72], color=LINE, linewidth=0.8)
    ax.plot([2.60, 11.40], [4.72, 4.72], color=LINE, linewidth=0.8)
    poly_arrow(ax, [(3.70, 4.72), (3.70, 4.53)], lw=0.8)
    poly_arrow(ax, [(6.35, 4.09), (7.60, 4.09)], lw=0.9)

    academic_titled_box(ax, 0.60, 0.78, 3.80, 0.94, "Memory update", "append labeled samples\nto bounded reservoir", GRAY_1)
    academic_titled_box(ax, 5.10, 0.78, 3.80, 0.94, "Reference update", "refresh snapshot only after\nan accepted repair", GRAY_1)
    academic_titled_box(ax, 9.60, 0.78, 3.80, 0.94, "Audit update", "record routing evidence,\nactive modules, and cost", GRAY_1)
    poly_arrow(ax, [(10.25, 3.52), (10.25, 2.35)], lw=0.9)
    ax.plot([2.50, 11.50], [2.35, 2.35], color=LINE, linewidth=0.8)
    for center_x in (2.50, 7.00, 11.50):
        poly_arrow(ax, [(center_x, 2.35), (center_x, 1.70)], lw=0.8)

    save_bundle(fig, "dasp_window_algorithm")


def algorithm_flowchart():
    fig, ax = plt.subplots(figsize=(10.0, 7.25))
    ax.set_xlim(0, 20.0)
    ax.set_ylim(0, 15.0)
    ax.axis("off")

    ax.text(10.0, 14.62, "Delayed-label DASP algorithm flowchart", ha="center", va="center", fontsize=16.0, fontweight="bold", color=INK)
    ax.plot([0.55, 1.35], [14.15, 14.15], color=LINE, linewidth=1.1)
    ax.add_patch(FancyArrowPatch((1.08, 14.15), (1.35, 14.15), arrowstyle="-|>", mutation_scale=10, color=LINE, linewidth=1.1))
    ax.text(1.55, 14.15, "current processing", ha="left", va="center", fontsize=12.0, color=INK)
    ax.plot([11.20, 12.00], [14.15, 14.15], color=LINE, linewidth=1.0, linestyle=(0, (4, 2.5)))
    ax.add_patch(FancyArrowPatch((11.73, 14.15), (12.00, 14.15), arrowstyle="-|>", mutation_scale=10, color=LINE, linewidth=1.0, linestyle=(0, (4, 2.5))))
    ax.text(12.20, 14.15, "delayed or persistent-state link", ha="left", va="center", fontsize=12.0, color=INK)

    ax.text(0.45, 13.55, "A. Pre-adaptation inference at iteration $t$", ha="left", va="center", fontsize=13.0, fontweight="bold", color=INK)
    academic_ellipse(ax, 0.30, 12.20, 1.80, 0.88, "Current\n$W_t$", 12.2)
    academic_titled_box(ax, 2.30, 12.08, 2.45, 1.12, "Predict", "before labels", GRAY_1, title_fs=12.5, body_fs=12.0)
    academic_titled_box(ax, 4.95, 12.00, 3.25, 1.28, "Representations", "module-wise\ncurrent features", GRAY_1, title_fs=12.2, body_fs=11.7)
    academic_titled_box(ax, 8.40, 12.00, 3.15, 1.28, "MMD evidence", "current vs frozen\nreference", GRAY_1, title_fs=12.2, body_fs=11.5)
    academic_titled_box(ax, 11.75, 12.00, 3.45, 1.28, "PendingUpdate", r"store $W_t$ + $\mathcal{E}_t$" + "\nfor one-window delay", GRAY_2, title_fs=12.2, body_fs=11.3)
    academic_titled_box(ax, 15.75, 12.00, 3.95, 1.28, "Reference snapshot", "final warm-up or\nlatest DASP repair", GRAY_1, title_fs=11.8, body_fs=11.2)
    for x1, x2 in ((2.10, 2.30), (4.75, 4.95), (8.20, 8.40), (11.55, 11.75)):
        poly_arrow(ax, [(x1, 12.64), (x2, 12.64)], lw=1.0)
    poly_arrow(ax, [(15.75, 12.64), (15.45, 12.64), (15.45, 11.60), (9.98, 11.60), (9.98, 12.00)], dashed=True, lw=0.9)

    ax.text(0.45, 11.25, r"B. Delayed-label localized adaptation for $\tau=t-1$", ha="left", va="center", fontsize=13.0, fontweight="bold", color=INK)
    academic_ellipse(ax, 0.30, 9.38, 1.85, 0.90, r"Labels" + "\n" + r"$Y_{\tau}$", 12.2)
    academic_titled_box(ax, 2.35, 9.22, 3.05, 1.22, "Restore evidence", r"matching $W_{\tau}$" + "\n" + r"and $\mathcal{E}_{\tau}$", GRAY_2, title_fs=12.0, body_fs=11.4)
    academic_titled_box(ax, 5.60, 9.14, 3.45, 1.38, "Training + holdout", "repair samples +\nperformance gate", GRAY_1, title_fs=11.8, body_fs=11.4)
    academic_titled_box(ax, 9.25, 9.06, 3.65, 1.54, "Gradient conflict", "current-window vs\nreplay gradients", GRAY_1, title_fs=11.3, body_fs=11.2)
    academic_diamond(ax, 13.10, 9.05, 3.05, 1.62, "MMD severity\ntriggered?", 11.5)
    academic_diamond(ax, 16.75, 9.05, 3.05, 1.62, "Holdout F1\n$<0.80$?", 11.2)
    for x1, x2 in ((2.15, 2.35), (5.40, 5.60), (9.05, 9.25), (12.90, 13.10)):
        poly_arrow(ax, [(x1, 9.83), (x2, 9.83)], lw=1.0)
    poly_arrow(ax, [(13.48, 12.00), (13.48, 10.82), (3.88, 10.82), (3.88, 10.44)], dashed=True, lw=0.95)
    poly_arrow(ax, [(16.15, 9.86), (16.75, 9.86)], label="No", label_xy=(16.45, 10.10), lw=1.0)

    academic_titled_box(ax, 0.30, 6.75, 6.20, 1.42, "Triggered attributed repair", "stronger adaptation of\nattributed modules", GRAY_3, title_fs=12.3, body_fs=11.9)
    academic_titled_box(ax, 6.90, 6.75, 6.20, 1.42, "Emergency repair", "performance-gated;\nhigh-conflict modules", GRAY_3, title_fs=11.5, body_fs=11.9)
    academic_titled_box(ax, 13.50, 6.75, 6.20, 1.42, "Background consolidation", "local rehearsal\nfor retention", GRAY_2, title_fs=11.8, body_fs=11.9)
    poly_arrow(ax, [(14.63, 9.05), (14.63, 8.55), (3.40, 8.55), (3.40, 8.17)], label="Yes", label_xy=(13.82, 8.78), lw=1.0)
    poly_arrow(ax, [(18.28, 9.05), (18.28, 8.55), (10.00, 8.55), (10.00, 8.17)], label="Yes", label_xy=(17.50, 8.78), lw=1.0)
    poly_arrow(ax, [(19.80, 9.86), (19.88, 9.86), (19.88, 8.55), (16.60, 8.55), (16.60, 8.17)], label="No", label_xy=(19.52, 8.78), lw=1.0)

    for center_x in (3.40, 10.00, 16.60):
        ax.plot([center_x, center_x], [6.75, 6.40], color=LINE, linewidth=0.9)
    ax.plot([3.40, 16.60], [6.40, 6.40], color=LINE, linewidth=0.9)
    poly_arrow(ax, [(10.00, 6.40), (10.00, 5.28), (2.45, 5.28), (2.45, 5.06)], lw=1.0)
    ax.text(0.45, 5.72, "C. Localized optimization and state update", ha="left", va="center", fontsize=13.0, fontweight="bold", color=INK)

    academic_titled_box(ax, 0.30, 3.72, 4.30, 1.34, "Localized rehearsal", "delayed current +\nbalanced replay", GRAY_1, title_fs=11.8, body_fs=11.5)
    academic_titled_box(ax, 4.85, 3.62, 5.10, 1.54, "Selective optimizer update", "routed modules + classifier;\nfreeze others", GRAY_1, title_fs=11.8, body_fs=11.1)
    academic_titled_box(ax, 10.20, 3.72, 3.65, 1.34, "Memory + audit", "append labels +\nrouting trace", GRAY_1, title_fs=12.0, body_fs=11.3)
    academic_titled_box(ax, 14.10, 3.72, 3.95, 1.34, "Refresh reference", "after every applied\nDASP repair", GRAY_1, title_fs=12.0, body_fs=11.4)
    academic_ellipse(ax, 18.30, 3.92, 1.40, 0.92, "Next\nwindow", 11.7)
    poly_arrow(ax, [(4.60, 4.39), (4.85, 4.39)], lw=1.0)
    poly_arrow(ax, [(9.95, 4.39), (10.20, 4.39)], lw=1.0)
    poly_arrow(ax, [(13.85, 4.39), (14.10, 4.39)], lw=1.0)
    poly_arrow(ax, [(18.05, 4.39), (18.30, 4.39)], lw=1.0)

    academic_titled_box(ax, 10.00, 1.68, 4.20, 1.20, "Reservoir memory", "balanced replay +\nbounded storage", GRAY_2, title_fs=11.8, body_fs=11.2)
    poly_arrow(ax, [(12.10, 2.88), (12.10, 3.72)], dashed=True, lw=0.9)
    poly_arrow(ax, [(10.00, 2.28), (2.45, 2.28), (2.45, 3.72)], dashed=True, lw=0.9)
    poly_arrow(ax, [(16.08, 5.06), (16.08, 5.42), (19.82, 5.42), (19.82, 12.64), (19.70, 12.64)], dashed=True, lw=0.9)

    save_bundle(fig, "dasp_algorithm_flowchart")

def simplified_architecture():
    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(7.0, 6.08, "Simplified delayed-label DASP pipeline", ha="center", va="center", fontsize=11.0, fontweight="bold", color=INK)
    academic_titled_box(ax, 0.35, 4.75, 1.80, 0.88, "$t$: current", "window $W_t$", GRAY_3)
    academic_titled_box(ax, 2.45, 4.75, 2.35, 0.88, "Predict + monitor", "prediction + MMD", GRAY_1)
    academic_titled_box(ax, 5.10, 4.75, 2.15, 0.88, "PendingUpdate", "stored evidence", GRAY_2, title_fs=8.7)
    academic_titled_box(ax, 7.55, 4.75, 2.65, 0.88, r"$t$: labels $Y_{\tau}$", r"$\tau=t-1$", GRAY_1)
    academic_titled_box(ax, 10.50, 4.75, 3.15, 0.88, "Repair-policy tree", "MMD, holdout F1,\nmodule conflict", GRAY_1, title_fs=8.8, body_fs=8.5)
    for x1, x2 in [(2.15, 2.45), (4.80, 5.10), (7.25, 7.55), (10.20, 10.50)]:
        poly_arrow(ax, [(x1, 5.19), (x2, 5.19)], lw=0.9)

    academic_titled_box(ax, 0.45, 2.90, 4.05, 1.08, "Triggered attributed repair", "stronger adaptation of\nattributed modules", GRAY_3, title_fs=8.5)
    academic_titled_box(ax, 4.98, 2.90, 4.05, 1.08, "Emergency repair", "performance-gated;\nhigh-conflict modules", GRAY_3, title_fs=8.7)
    academic_titled_box(ax, 9.50, 2.90, 4.05, 1.08, "Background consolidation", "local rehearsal\nfor retention", GRAY_2, title_fs=8.2)
    ax.plot([2.48, 11.53], [4.45, 4.45], color=LINE, linewidth=0.8)
    ax.plot([12.08, 12.08], [4.80, 4.45], color=LINE, linewidth=0.8)
    for center_x in (2.48, 7.00, 11.53):
        poly_arrow(ax, [(center_x, 4.45), (center_x, 3.98)], lw=0.8)

    academic_titled_box(ax, 0.65, 1.08, 3.65, 0.98, "Localized rehearsal", "current samples +\nbalanced replay", GRAY_1)
    academic_titled_box(ax, 5.18, 1.08, 3.65, 0.98, "Selective update", "adapt active modules;\nfreeze unaffected modules", GRAY_1)
    academic_titled_box(ax, 9.70, 1.08, 3.65, 0.98, "State update", "reservoir, reference,\nand audit", GRAY_1)
    for center_x in (2.48, 7.00, 11.53):
        ax.plot([center_x, center_x], [2.90, 2.54], color=LINE, linewidth=0.8)
    ax.plot([2.48, 11.53], [2.54, 2.54], color=LINE, linewidth=0.8)
    poly_arrow(ax, [(2.48, 2.54), (2.48, 2.06)], lw=0.8)
    poly_arrow(ax, [(4.30, 1.57), (5.18, 1.57)], lw=0.9)
    poly_arrow(ax, [(8.83, 1.57), (9.70, 1.57)], lw=0.9)

    save_bundle(fig, "dasp_window_algorithm_simplified")


def experiment_matrix():
    df = pd.read_csv(RESULTS / "full_summary.csv")
    order_d = ["cic-ids2017", "nf-bot-iot-v3", "nf-cse-cic-ids2018-v3", "nf-ton-iot-v3", "nf-unsw-nb15-v3"]
    order_s = ["S0", "S1-Sudden", "S2-Gradual", "S3-Incremental", "S4-Recurrent", "S5-Blip", "S6-Combined", "S7-Local-Low", "S7-Local-Medium", "S7-Local-High"]
    ax_data = df[df.method.isin(["modular_dasp", "ssf"])].pivot_table(index=["dataset", "scenario", "stream_seed", "model_seed"], columns="method", values="mean_macro_f1")
    ax_data["gap"] = ax_data["modular_dasp"] - ax_data["ssf"]
    mat = ax_data.reset_index().pivot_table(index="dataset", columns="scenario", values="gap", aggfunc="mean").reindex(index=order_d, columns=order_s)
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    image = ax.imshow(mat.values, cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="auto")
    ax.set_xticks(np.arange(10), ["S0", "Sudden", "Gradual", "Incr.", "Recur.", "Blip", "Comb.", "Local-L", "Local-M", "Local-H"], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(np.arange(5), ["CIC-IDS2017", "NF-BoT-IoT", "NF-CSE-CIC", "NF-ToN-IoT", "NF-UNSW"], fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i,j]:+.3f}", ha="center", va="center", fontsize=6.6)
    ax.set_title("Confirmation design and paired DASP--SSF gap", fontsize=10, fontweight="bold")
    ax.set_xlabel("10 primary scenarios; each cell averages 5 stream seeds and 2 model seeds", fontsize=7.5)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Macro-F1 gap")
    fig.tight_layout()
    save(fig, "dasp_experiment_matrix.pdf")


def deployment_blueprint():
    fig, ax = plt.subplots(figsize=(10.2, 3.1))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 3.1); ax.axis("off")
    ax.text(0.08, 2.82, "Deployment blueprint: prediction, adaptation, and audit are separate planes", fontsize=10.5, fontweight="bold", color=BLUE)
    stages=[("Traffic ingress", PALE_BLUE, BLUE), ("DASP predictor", PALE_BLUE, BLUE), ("Delayed-label\nqueue", PALE_GOLD, RED), ("Repair service", PALE_GREEN, GREEN), ("Model registry\nand audit log", "#f2f2f2", MID)]
    xs=[0.15,2.05,4.0,5.95,8.0]
    for (label,fc,ec),x in zip(stages,xs):
        box(ax,x,1.62,1.45,0.62,label,fc,ec,7.8)
    for x in [1.62,3.52,5.47,7.52]:
        arrow(ax,x,1.93,x+0.34,1.93,DARK,1.0)
    box(ax,2.05,0.48,1.45,0.50,"prequential\nmetrics",PALE_BLUE,BLUE,7.5)
    box(ax,5.95,0.48,1.45,0.50,"mode gate +\nlocal update",PALE_GREEN,GREEN,7.5)
    box(ax,8.0,0.48,1.45,0.50,"signed\ntrace", "#f2f2f2", MID,7.5)
    arrow(ax,2.78,1.58,2.78,1.02,BLUE,0.9)
    arrow(ax,6.68,1.58,6.68,1.02,GREEN,0.9)
    arrow(ax,8.73,1.58,8.73,1.02,MID,0.9)
    ax.text(0.15,0.12,"The deployed predictor can be monitored continuously while the repair path remains bounded, reviewable, and reversible.",fontsize=7.3,color=DARK)
    save(fig,"dasp_deployment_blueprint.pdf")


def locality_mask():
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.set_xlim(0, 8.4); ax.set_ylim(0, 3.0); ax.axis("off")
    ax.text(0.1, 2.72, "Structural locality: the router changes the update mask, not the model interface", fontsize=10.5, fontweight="bold", color=BLUE)
    labels = ["low-level\nflow stats", "protocol\npatterns", "temporal\nrelations", "contextual\nfeatures"]
    for i, label in enumerate(labels):
        x = 0.25 + i * 1.55
        box(ax, x, 1.72, 1.25, 0.52, label, PALE_BLUE, BLUE, 7.5)
        box(ax, x, 0.74, 1.25, 0.52, "frozen" if i in [0,2] else "plastic", "#f2f2f2" if i in [0,2] else PALE_GREEN, MID if i in [0,2] else GREEN, 8, 0.9)
        arrow(ax, x+0.625, 1.68, x+0.625, 1.32, MID, 0.9)
    box(ax, 6.55, 1.72, 1.45, 0.52, "fusion\nclassifier", PALE_GOLD, RED, 8)
    for x in [1.5, 3.05, 4.6, 6.15]:
        arrow(ax, x, 1.98, 6.45, 1.98, RED, 0.8, rad=0.03)
    ax.text(0.2, 0.18, "Illustrative mask: active modules are selected by evidence; inactive modules remain byte-identical.", fontsize=7.4, color=DARK)
    save(fig, "dasp_locality_mask.pdf")


def main():
    mechanism_overview()
    routing_mechanics()
    timeline()
    guard_flow()
    window_algorithm()
    algorithm_flowchart()
    simplified_architecture()
    experiment_matrix()
    locality_mask()
    deployment_blueprint()
    print(FIGURES)

if __name__ == "__main__":
    main()








