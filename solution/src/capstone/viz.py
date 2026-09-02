"""Shared charting house style for C-suite reporting.

Every phase produces findings backed by charts. This module centralises the
visual system so charts across phases read as one deck:

  * neutral off-white surface, hairline solid grid, no top/right spines
  * left-aligned title that states the *takeaway*, muted subtitle, source footer
  * a fixed, CVD-validated categorical palette (never cycle / never recolor on
    filter) and a reserved status palette for Paid / Pending / Rejected states
  * direct value labels on bars (relief rule: yellow status fill is < 3:1 on the
    surface, so every segment/bar carries a visible label; the report also ships
    the table view)

Palette source: dataviz skill reference instance (validated light mode).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e7e7e3"

# Fixed categorical order - assign by slot, never cycle past slot 8.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Sequential blue ramp (light -> dark) for magnitude encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Reserved status palette - use ONLY when colour *means* state.
STATUS = {
    "good": "#008300",
    "warning": "#eda100",
    "serious": "#eb6834",
    "critical": "#e34948",
    "neutral": "#b7b7b3",
}

# Domain state -> status colour (claim outcomes, DQ severities).
CLAIM_STATUS_COLORS = {"Paid": STATUS["good"], "Pending": STATUS["warning"], "Rejected": STATUS["critical"]}
SEVERITY_COLORS = {"ERROR": STATUS["critical"], "WARN": STATUS["warning"], "INFO": CATEGORICAL[0]}


def apply_house_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def new_figure(width: float = 8.5, height: float = 4.8):
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width)) if text else text


def finalize(fig, ax, *, title: str, subtitle: str | None, source: str, out_path: Path) -> Path:
    """Left-aligned takeaway title + muted subtitle + source footer, then save.

    Header and footer heights are reserved in inches so they never collide with
    the plot regardless of figure size.
    """
    h = fig.get_figheight()
    subtitle = _wrap(subtitle, 110) if subtitle else None
    sub_lines = subtitle.count("\n") + 1 if subtitle else 0

    title_in = 0.30                      # top margin -> title baseline
    sub_in = title_in + 0.24             # title -> subtitle baseline
    header_in = (sub_in + 0.20 * sub_lines + 0.14) if subtitle else (title_in + 0.24)
    footer_in = 0.34

    if hasattr(ax, "set_title"):
        ax.set_title("")
    fig.text(0.015, 1 - title_in / h, title, ha="left", va="top",
             fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.015, 1 - sub_in / h, subtitle, ha="left", va="top",
                 fontsize=9.5, color=INK_SECONDARY)
    fig.text(0.015, 0.12 / h, source, ha="left", va="bottom", fontsize=7.5, color=INK_MUTED)

    fig.tight_layout(rect=(0, footer_in / h, 1, 1 - header_in / h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def label_bars(ax, bars, values, *, fmt="{:.0f}", horizontal=False, inside=False, color=INK) -> None:
    for bar, val in zip(bars, values):
        if horizontal:
            w = bar.get_width()
            x = w / 2 if inside else w
            ax.text(x, bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center",
                    ha="center" if inside else "left",
                    fontsize=8.5, color=("white" if inside else color),
                    fontweight="bold" if inside else "normal")
        else:
            h = bar.get_height()
            y = h / 2 if inside else h
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    fmt.format(val), ha="center",
                    va="center" if inside else "bottom",
                    fontsize=8.5, color=("white" if inside else color),
                    fontweight="bold" if inside else "normal")


def heatmap(ax, matrix, row_labels, col_labels, *, fmt="{:.0f}",
            normalize_rows=True, cbar_label=None):
    """House-style heatmap (e.g. a confusion matrix).

    Shading uses ``SEQUENTIAL_BLUE`` on the row-normalised values so the colour
    encodes *rate* while the printed annotation keeps the raw count. Every cell
    is labelled (the grid/annotation, not colour, carries the number).
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    m = np.asarray(matrix, dtype=float)
    shade = m / m.sum(axis=1, keepdims=True).clip(min=1) if normalize_rows else m / m.max().clip(min=1)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    im = ax.imshow(shade, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.tick_params(top=False, bottom=True, left=True, right=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if shade[i, j] > 0.55 else INK)
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label(cbar_label, color=INK_SECONDARY, fontsize=8)
        cb.ax.tick_params(labelsize=7, color=INK_MUTED)
    return im


def money(value: float) -> str:
    """Compact currency label (data is in INR-like units)."""
    for unit, div in (("Cr", 1e7), ("L", 1e5), ("K", 1e3)):
        if abs(value) >= div:
            return f"{value / div:,.1f}{unit}"
    return f"{value:,.0f}"
