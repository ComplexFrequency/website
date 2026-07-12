from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cache_io import load_or_compute_array

ArrayT = TypeVar("ArrayT", bound=np.ndarray)


@dataclass(frozen=True)
class Palette:
    ink: str
    muted: str
    accent: str
    danger: str
    grid: str


PALETTE = Palette(
    ink="#7d6b56",
    muted="#a08f78",
    accent="#d9773a",
    danger="#c4543a",
    grid="#7d6b5640",
)

FIGURE_WIDTH_IN: float = 7.6
PNG_DPI: int = 200
BASE_FONT_SIZE: float = 11.0
TITLE_FONT_SIZE: float = 12.5
PANEL_SUBTITLE_SIZE: float = 10.0
LEGEND_FONT_SIZE: float = 9.5

HAIRLINE_TRACE: dict[str, object] = {
    "linewidth": 0.28,
    "alpha": 0.72,
    "solid_capstyle": "round",
    "zorder": 2,
}


def apply():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "axes.titlesize": PANEL_SUBTITLE_SIZE,
            "figure.titlesize": TITLE_FONT_SIZE,
            "figure.titleweight": "regular",
            "figure.figsize": (FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.55),
            "figure.dpi": PNG_DPI,
            "savefig.dpi": PNG_DPI,
            "svg.fonttype": "none",
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "axes.edgecolor": PALETTE.ink,
            "axes.labelcolor": PALETTE.ink,
            "xtick.color": PALETTE.ink,
            "ytick.color": PALETTE.ink,
            "text.color": PALETTE.ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": PALETTE.grid,
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.2,
            "legend.labelspacing": 0.35,
            "lines.linewidth": 1.4,
            "axes.unicode_minus": False,
        }
    )


def figure_title(figure: Figure, text: str):
    figure.suptitle(
        text,
        x=0.01,
        y=0.995,
        ha="left",
        va="top",
        color=PALETTE.ink,
        fontsize=TITLE_FONT_SIZE,
    )


def panel_subtitle(axes: Axes, text: str):
    axes.set_title(
        text,
        loc="left",
        color=PALETTE.ink,
        fontsize=PANEL_SUBTITLE_SIZE,
        pad=6,
    )


def style_legend(axes: Axes, **kwargs: object):
    defaults: dict[str, object] = {
        "frameon": False,
        "fontsize": LEGEND_FONT_SIZE,
        "labelcolor": PALETTE.ink,
        "handlelength": 1.8,
        "borderaxespad": 0.3,
    }
    defaults.update(kwargs)
    legend = axes.legend(**defaults)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(PALETTE.ink)
    return legend


def figure_legend(
    figure: Figure,
    handles: list[object],
    labels: list[str],
    **kwargs: object,
):
    defaults: dict[str, object] = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 0.94),
        "ncol": len(labels),
        "frameon": False,
        "fontsize": LEGEND_FONT_SIZE - 0.5,
        "handlelength": 1.6,
        "columnspacing": 1.2,
        "handletextpad": 0.4,
        "borderaxespad": 0.0,
    }
    defaults.update(kwargs)
    legend = figure.legend(handles=handles, labels=labels, **defaults)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(PALETTE.ink)
    return legend


def plot_hairline_trace(
    axes: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str | None = None,
    label: str | None = None,
    **overrides: object,
):
    style = dict(HAIRLINE_TRACE)
    style["color"] = color or PALETTE.ink
    if label is not None:
        style["label"] = label
    style.update(overrides)
    return axes.plot(x, y, **style)[0]



def new_figure(
    height_ratio: float = 0.55,
    *,
    width: float = FIGURE_WIDTH_IN,
) -> tuple[Figure, Axes]:
    apply()
    figure, axes = plt.subplots(figsize=(width, width * height_ratio))
    style_axes(axes)
    return figure, axes


def style_axes(axes: Axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_linewidth(0.8)
    axes.spines["bottom"].set_linewidth(0.8)
    axes.spines["left"].set_color(PALETTE.ink)
    axes.spines["bottom"].set_color(PALETTE.ink)
    axes.grid(True, axis="y", linestyle=":", linewidth=0.7, color=PALETTE.grid)
    axes.grid(False, axis="x")
    axes.tick_params(colors=PALETTE.ink, width=0.8)
    axes.set_facecolor("none")


def annotate(
    axes: Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    color: str | None = None,
    ha: str = "left",
    va: str = "center",
):
    axes.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="data",
        color=color or PALETTE.ink,
        ha=ha,
        va=va,
        fontsize=BASE_FONT_SIZE - 0.5,
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE.muted,
            "lw": 0.7,
            "shrinkA": 0,
            "shrinkB": 2,
        },
    )


def save_figure(figure: Figure, stem: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.15)
    figure.savefig(
        png_path,
        format="png",
        dpi=PNG_DPI * 2,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    return svg_path, png_path


def load_or_compute(
    cache_path: Path,
    compute: Callable[..., ArrayT],
    *args: object,
    **kwargs: object,
) -> ArrayT:
    return load_or_compute_array(cache_path, compute, *args, **kwargs)
