#!/usr/bin/env python3
"""
scripts/plot_metrics.py — Publication-ready visualization of FL benchmark results.

Reads *_metrics.json files from results/ and generates four comparison charts:
accuracy vs rounds, loss vs rounds, convergence bar chart, and client weight
distributions.  Outputs to results/figures/ as PNG (300 DPI) and/or PDF.

Usage:  python scripts/plot_metrics.py [--results-dir results] [--format both]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend — safe for headless servers

logger = logging.getLogger(__name__)

# Strategy key -> display label.  "r3fl" is visually distinguished as "Ours".
STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    "fedavg": "FedAvg",
    "krum": "Krum",
    "trimmed_mean": "Trimmed Mean",
    "fltrust": "FLTrust",
    "r3fl": "R3-FL (Ours)",
}

# Colorblind-friendly palette (tab10 subset; R3-FL gets the most salient colour).
STRATEGY_COLORS: dict[str, str] = {
    "r3fl": "#1f77b4",       # blue  — prominent
    "fedavg": "#ff7f0e",     # orange
    "krum": "#2ca02c",       # green
    "trimmed_mean": "#d62728",  # red
    "fltrust": "#9467bd",    # purple
}

STRATEGY_MARKERS: dict[str, str] = {
    "r3fl": "*",
    "fedavg": "o",
    "krum": "s",
    "trimmed_mean": "^",
    "fltrust": "D",
}

FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_LEGEND = 14, 12, 10, 10
LINEWIDTH_DEFAULT, LINEWIDTH_OURS = 1.5, 2.8
MARKERSIZE_DEFAULT, MARKERSIZE_OURS = 5, 10
FIGSIZE_LINE, FIGSIZE_BAR, FIGSIZE_DIST = (10, 6), (8, 6), (12, 5)


def _display_name(strategy: str) -> str:
    """Return a human-readable label for *strategy*, falling back to title-case."""
    return STRATEGY_DISPLAY_NAMES.get(strategy, strategy.replace("_", " ").title())


def _style_kwargs(strategy: str) -> dict[str, Any]:
    """Return matplotlib keyword arguments that visually distinguish R3-FL."""
    is_ours = strategy == "r3fl"
    return {
        "color": STRATEGY_COLORS.get(strategy, "#7f7f7f"),
        "marker": STRATEGY_MARKERS.get(strategy, "x"),
        "linewidth": LINEWIDTH_OURS if is_ours else LINEWIDTH_DEFAULT,
        "markersize": MARKERSIZE_OURS if is_ours else MARKERSIZE_DEFAULT,
        "zorder": 10 if is_ours else 2,
    }


def _apply_common_style(ax: matplotlib.axes.Axes) -> None:
    """Apply shared formatting (grid, tick sizes) to an axes object."""
    ax.grid(True, which="major", linestyle="-", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK)
    ax.tick_params(axis="both", which="minor", labelsize=FONT_TICK - 2)


def load_metrics(results_dir: Path) -> list[dict[str, Any]]:
    """Load all *_metrics.json files, sorted by strategy name for determinism."""
    pattern = "*_metrics.json"
    files = sorted(results_dir.glob(pattern))
    if not files:
        logger.warning("No metric files matching '%s' found in %s", pattern, results_dir)
        return []

    data: list[dict[str, Any]] = []
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            if "strategy" not in parsed:
                logger.warning("Skipping %s — missing 'strategy' key", filepath.name)
                continue
            data.append(parsed)
            logger.info("Loaded metrics for strategy '%s' from %s", parsed["strategy"], filepath.name)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s — %s", filepath.name, exc)
    return sorted(data, key=lambda d: d["strategy"])


def plot_accuracy(data: list[dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    """Chart 1 — Validation Accuracy vs Communication Rounds (line plot)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for entry in data:
        strategy = entry["strategy"]
        rounds = entry.get("rounds", list(range(1, len(entry["accuracy"]) + 1)))
        ax.plot(
            rounds,
            entry["accuracy"],
            label=_display_name(strategy),
            markevery=max(1, len(rounds) // 10),
            **_style_kwargs(strategy),
        )

    ax.set_xlabel("Communication Round", fontsize=FONT_LABEL)
    ax.set_ylabel("Validation Accuracy", fontsize=FONT_LABEL)
    ax.set_title("Validation Accuracy vs Communication Rounds", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    _apply_common_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_dir / "accuracy_vs_rounds", fmt, dpi)
    plt.close(fig)
    logger.info("Saved accuracy chart")


def plot_loss(data: list[dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    """Chart 2 — Validation Loss vs Communication Rounds (line plot)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    for entry in data:
        strategy = entry["strategy"]
        rounds = entry.get("rounds", list(range(1, len(entry["loss"]) + 1)))
        ax.plot(
            rounds,
            entry["loss"],
            label=_display_name(strategy),
            markevery=max(1, len(rounds) // 10),
            **_style_kwargs(strategy),
        )

    ax.set_xlabel("Communication Round", fontsize=FONT_LABEL)
    ax.set_ylabel("Validation Loss", fontsize=FONT_LABEL)
    ax.set_title("Validation Loss vs Communication Rounds", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    ax.set_yscale("log")
    _apply_common_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_dir / "loss_vs_rounds", fmt, dpi)
    plt.close(fig)
    logger.info("Saved loss chart")


def plot_convergence(data: list[dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    """Chart 3 — Rounds to Target Accuracy (bar chart)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    strategies: list[str] = []
    rounds_to_target: list[float] = []
    bar_colors: list[str] = []
    hatch_patterns: list[str] = []

    for entry in data:
        strategy = entry["strategy"]
        convergence = entry.get("convergence_round")
        num_rounds = entry.get("num_rounds", len(entry.get("accuracy", [])))
        strategies.append(_display_name(strategy))
        bar_colors.append(STRATEGY_COLORS.get(strategy, "#7f7f7f"))

        if convergence is not None:
            rounds_to_target.append(convergence)
            hatch_patterns.append("")
        else:
            # Strategy never converged — show max rounds with hatching
            rounds_to_target.append(num_rounds)
            hatch_patterns.append("//")

    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, rounds_to_target, color=bar_colors, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, hatch_patterns):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("gray")

    # Add value labels on bars
    for bar, val, hatch in zip(bars, rounds_to_target, hatch_patterns):
        label = f"{int(val)}" if not hatch else f"{int(val)} (N/A)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            label,
            ha="center",
            va="bottom",
            fontsize=FONT_TICK,
        )

    target_acc = None
    for entry in data:
        if entry.get("target_accuracy") is not None:
            target_acc = entry["target_accuracy"]
            break

    title_suffix = f" (target = {target_acc})" if target_acc is not None else ""
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, fontsize=FONT_TICK)
    ax.set_ylabel("Rounds to Target Accuracy", fontsize=FONT_LABEL)
    ax.set_title(f"Convergence Speed{title_suffix}", fontsize=FONT_TITLE)
    _apply_common_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_dir / "convergence_bar", fmt, dpi)
    plt.close(fig)
    logger.info("Saved convergence bar chart")


def plot_weight_distributions(
    data: list[dict[str, Any]], output_dir: Path, fmt: str, dpi: int
) -> None:
    """Chart 4 — Client Weight Distributions (box + strip overlay).
    Compares R3-FL vs FedAvg, honest vs malicious (first malicious_fraction clients).
    """
    # Identify R3-FL and FedAvg entries
    r3fl_entry = None
    fedavg_entry = None
    for entry in data:
        if entry["strategy"] == "r3fl":
            r3fl_entry = entry
        elif entry["strategy"] == "fedavg":
            fedavg_entry = entry

    if r3fl_entry is None:
        logger.warning("R3-FL metrics not found — skipping weight distribution chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DIST, sharey=True)

    entries_to_plot = []
    if fedavg_entry is not None:
        entries_to_plot.append(("FedAvg", fedavg_entry, axes[0]))
    else:
        # If no FedAvg, use the left panel for an informational note
        axes[0].text(
            0.5, 0.5, "FedAvg data\nnot available",
            transform=axes[0].transAxes, ha="center", va="center", fontsize=FONT_LABEL,
        )
        axes[0].set_title("FedAvg", fontsize=FONT_TITLE)
    entries_to_plot.append(("R3-FL (Ours)", r3fl_entry, axes[1]))

    for label, entry, ax in entries_to_plot:
        client_weights_all = entry.get("client_weights", [])
        if not client_weights_all:
            ax.text(
                0.5, 0.5, "No weight data",
                transform=ax.transAxes, ha="center", va="center", fontsize=FONT_LABEL,
            )
            ax.set_title(label, fontsize=FONT_TITLE)
            continue

        # Use the last round's weights
        last_weights = np.array(client_weights_all[-1], dtype=float)
        num_clients = len(last_weights)
        malicious_frac = entry.get("malicious_fraction", 0.3)
        num_malicious = int(num_clients * malicious_frac)

        malicious_weights = last_weights[:num_malicious]
        honest_weights = last_weights[num_malicious:]

        # Box plots side by side
        bp = ax.boxplot(
            [honest_weights, malicious_weights],
            tick_labels=["Honest", "Malicious"],
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        bp["boxes"][0].set_facecolor("#2ca02c80")  # green (honest)
        bp["boxes"][1].set_facecolor("#d6272880")   # red   (malicious)

        # Overlay individual points (strip / jitter)
        rng = np.random.default_rng(42)
        for idx, (weights, x_center) in enumerate(
            [(honest_weights, 1), (malicious_weights, 2)]
        ):
            jitter = rng.uniform(-0.12, 0.12, size=len(weights))
            color = "#2ca02c" if idx == 0 else "#d62728"
            ax.scatter(
                x_center + jitter, weights, alpha=0.45, s=12, color=color, zorder=3
            )

        ax.set_title(label, fontsize=FONT_TITLE)
        ax.set_ylabel("Client Weight" if ax == axes[0] else "", fontsize=FONT_LABEL)
        _apply_common_style(ax)

    fig.suptitle("Client Weight Distributions (Last Round)", fontsize=FONT_TITLE, y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir / "weight_distributions", fmt, dpi)
    plt.close(fig)
    logger.info("Saved weight distribution chart")


def _save_figure(fig: matplotlib.figure.Figure, stem: Path, fmt: str, dpi: int) -> None:
    """Save *fig* to disk in the requested format(s)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("png", "both"):
        fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    if fmt in ("pdf", "both"):
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")


def plot_all(results_dir: Path, output_dir: Path, fmt: str, dpi: int, style: str) -> None:
    """Load data and generate all charts."""
    # Apply matplotlib style
    try:
        plt.style.use(style)
    except OSError:
        logger.warning("Style '%s' not found — falling back to default", style)

    data = load_metrics(results_dir)
    if not data:
        logger.error("No valid metric files found — nothing to plot")
        sys.exit(1)

    logger.info(
        "Generating charts for %d strategies: %s",
        len(data),
        ", ".join(d["strategy"] for d in data),
    )

    plot_accuracy(data, output_dir, fmt, dpi)
    plot_loss(data, output_dir, fmt, dpi)
    plot_convergence(data, output_dir, fmt, dpi)
    plot_weight_distributions(data, output_dir, fmt, dpi)

    logger.info("All charts saved to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready charts from FL benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing *_metrics.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to save generated figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "both"],
        default="both",
        help="Output format for saved figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in dots per inch (PNG only)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="seaborn-v0_8-paper",
        help="Matplotlib style sheet name",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    plot_all(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        fmt=args.format,
        dpi=args.dpi,
        style=args.style,
    )
