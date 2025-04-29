#!/usr/bin/env python3
"""visualize_optim_state.py

Produce layer‑wise visualisations for the first and second moment
statistics (``exp_avg`` and ``exp_avg_sq``) contained in a *fully
un‑sharded* optimizer checkpoint produced by ``dump_optim_state.py``.

The script scans every parameter, groups them by *layer index* (inferred
from typical naming conventions ``...layers.<idx>...``) and then plots:

  • For ``exp_avg``: mean absolute value per layer
  • For ``exp_avg_sq``: mean value per layer (i.e. running variance)

All plots are saved as PNG inside a user‑specified output directory
(default: ``<ckpt_dir>/optim_stats``).

Example usage
-------------

    python scripts/visualize_optim_state.py \
        --optim_state /scratch/.../export_full_state/optimizer_full.pt

Optional flags:
  --out_dir /path/to/save/plots
  --no_exp_avg     # skip first‑moment plots
  --no_exp_avg_sq  # skip second‑moment plots
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")  # matches e.g. '...layers.12.'


def _infer_layer(name: str) -> int | None:
    """Return integer layer index if present in parameter name, else None."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _layer_stats_full(
    state: Dict[str, Dict[str, Any]],
    key: str,
) -> dict[int, dict[str, float]]:
    """Return per‑layer statistics (mean, min, max) over all optimiser values.

    For ``exp_avg`` we operate on absolute values to avoid sign cancellation.
    """
    collect: dict[int, list[torch.Tensor]] = defaultdict(list)
    for pname, p_state in state.items():
        lid = _infer_layer(pname)
        if lid is None:
            continue
        tensor = p_state.get(key)
        if tensor is not None and torch.is_tensor(tensor) and tensor.numel() > 0:
            t = tensor.abs() if key == "exp_avg" else tensor
            collect[lid].append(t.flatten().cpu())

    if not collect:
        raise RuntimeError(f"No tensor entries found for key '{key}' – maybe not present in state file?")

    stats: dict[int, dict[str, float]] = {}
    for lid, tensors in collect.items():
        cat = torch.cat(tensors)
        stats[lid] = {
            "mean": float(cat.mean().item()),
            "min": float(cat.min().item()),
            "max": float(cat.max().item()),
        }
    return stats


def _plot_layerwise(
    layer_ids: List[int],
    values: List[float],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(layer_ids, values, marker="o")
    plt.title(title)
    plt.xlabel("Layer index")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------- NEW: helpers ------------------------------

SAMPLE_LIMIT = 200_000  # max elements to keep per layer for histogram

# For per‑parameter histograms we keep more elements by default
PARAM_HIST_LIMIT = 1_000_000  # adjust as needed


def _collect_layer_values(state: Dict[str, Dict[str, Any]], key: str, abs_vals: bool = False) -> Dict[int, torch.Tensor]:
    """Return a dict[layer_id → 1‑D tensor] containing (optionally absolute) values.

    To keep memory usage manageable, each layer is randomly down‑sampled to
    *at most* ``SAMPLE_LIMIT`` elements.
    """
    layers: dict[int, list[torch.Tensor]] = defaultdict(list)
    for pname, p_state in state.items():
        layer_id = _infer_layer(pname)
        if layer_id is None:
            continue
        tensor = p_state.get(key)
        if tensor is not None and torch.is_tensor(tensor) and tensor.numel() > 0:
            t = tensor.detach().flatten().cpu()
            if abs_vals:
                t = t.abs()
            # optional down‑sample
            if t.numel() > SAMPLE_LIMIT:
                idx = torch.randperm(t.numel())[:SAMPLE_LIMIT]
                t = t[idx]
            layers[layer_id].append(t)

    # concatenate per layer
    return {lid: torch.cat(chunks) for lid, chunks in layers.items() if chunks}


def _plot_hist(values: torch.Tensor, title: str, *, ax: plt.Axes | None = None, bins: int = 100) -> None:
    """Plot histogram on the given axis or into a fresh figure."""
    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(values.numpy(), bins=bins, log=True, color="#1f77b4", alpha=0.75, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("log‑frequency")
    ax.grid(linewidth=0.3, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        fig.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)


# --------------------------- per‑parameter helper ---------------------------


def _plot_hist_save(values: torch.Tensor, title: str, save_path: Path, *, bins: int = 100) -> None:
    """Save a standalone histogram figure for *values*.

    Comparable to :func:`_plot_hist`, but writes directly to *save_path* and
    applies a higher element cap (`PARAM_HIST_LIMIT`) suitable for
    per‑parameter visualisations.
    """
    if values.numel() == 0:
        return

    if values.numel() > PARAM_HIST_LIMIT:
        idx = torch.randperm(values.numel())[:PARAM_HIST_LIMIT]
        values = values[idx]

    plt.figure(figsize=(6, 4))
    plt.hist(values.numpy(), bins=bins, log=True, color="#1f77b4", alpha=0.75, edgecolor="black")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("log‑frequency")
    plt.grid(linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Additional helper: heatmap plotting for exp_avg_sq

def _plot_heatmap(values: torch.Tensor, title: str, save_path: Path, *, max_side: int = 128) -> None:
    """Save a heatmap image for the given 1‑D tensor *values*.

    The tensor is (optionally) down‑sampled and reshaped into a roughly‑square
    matrix so that ``plt.imshow`` can visualise it as a heatmap.  The colourbar
    uses a logarithmic scale to improve visibility for parameters that span
    several orders of magnitude.

    Args:
        values: 1‑D tensor containing the optimiser statistics for a particular
            layer (``exp_avg_sq``).  Must contain at least one element.
        title: Title for the plot.
        save_path: Destination ``.png`` file path.
        max_side: Maximum side length (⩾ 1) for the square heatmap grid.  The
            total number of elements therefore is ``max_side ** 2``.  Elements
            beyond this limit are randomly sampled to keep the plot size
            manageable.
    """
    if values.numel() == 0:
        return  # nothing to plot

    # Down‑sample if necessary.
    total_limit = max_side * max_side
    if values.numel() > total_limit:
        idx = torch.randperm(values.numel())[: total_limit]
        values = values[idx]

    # Pad with NaNs if we have fewer than required so reshape works nicely.
    if values.numel() < total_limit:
        pad = total_limit - values.numel()
        values = torch.cat([values, torch.full((pad,), float("nan"))])

    # Reshape into square grid.
    grid = values.view(max_side, max_side)

    # Plot heatmap.
    plt.figure(figsize=(6, 6))
    # Use logarithmic colour scale; add a small epsilon to avoid log(0).
    import numpy as np
    grid_np = grid.abs().numpy()  # ensure positive values for log scale
    eps = np.finfo(grid_np.dtype).tiny
    img = plt.imshow(np.log10(grid_np + eps), cmap="viridis", aspect="auto")
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label("log10(value)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -------------------- NEW: per‑parameter 2‑D heatmaps --------------------

def _downsample_tensor_2d(t: torch.Tensor, max_side: int = 256) -> torch.Tensor:
    """Down‑sample a 2‑D tensor so that both sides are ≤ ``max_side``.

    Uses simple strided slicing – sufficient for visualisation purposes.
    """
    h, w = t.shape
    if h <= max_side and w <= max_side:
        return t
    step_h = max(1, int(torch.ceil(torch.tensor(h / max_side)).item()))
    step_w = max(1, int(torch.ceil(torch.tensor(w / max_side)).item()))
    return t[::step_h, ::step_w]


def _plot_heatmap_2d(tensor: torch.Tensor, title: str, save_path: Path, *, max_side: int = 1024) -> None:
    """Plot a heatmap for a single 2‑D tensor (absolute values, log10 scale)."""
    if tensor.numel() == 0:
        return
    # Ensure CPU, float32 for plotting
    data = tensor.detach().abs().cpu()

    # Down‑sample only if *both* dimensions are substantially larger than max_side
    if max(data.shape) > max_side * 1.5:  # allow a bit of slack
        data = _downsample_tensor_2d(data, max_side=max_side)

    import numpy as np
    arr = data.numpy()
    eps = np.finfo(arr.dtype).tiny
    plt.figure(figsize=(6, 5))
    img = plt.imshow(np.log10(arr + eps), cmap="viridis", aspect="auto" )
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label("log10(|value|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Visualise exp_avg / exp_avg_sq statistics per layer")
    parser.add_argument("--optim_state", type=Path, required=True, help="Path to optimizer_full.pt")
    parser.add_argument("--out_dir", type=Path, default=None, help="Destination folder for PNG plots (default: <ckpt_dir>/optim_stats)")
    parser.add_argument("--no_exp_avg", action="store_true", help="Disable exp_avg plots")
    parser.add_argument("--no_exp_avg_sq", action="store_true", help="Disable exp_avg_sq plots")
    args = parser.parse_args()

    if not args.optim_state.is_file():
        raise FileNotFoundError(args.optim_state)

    ckpt_dir = args.optim_state.parent
    out_dir = args.out_dir or (ckpt_dir / "optim_stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading optimiser state: {args.optim_state}")
    ckpt: Dict[str, Any] = torch.load(args.optim_state, map_location="cpu")  # type: ignore[arg-type]
    state: Dict[str, Dict[str, Any]] = ckpt["state"]

    # ------------------------------------------------------------------
    # Layer‑wise stats (mean/min/max) and JSON dumping
    # ------------------------------------------------------------------
    stats_json: dict[str, dict[str, dict[str, float]]] = {}

    if not args.no_exp_avg:
        print("Computing layer‑wise stats for exp_avg …")
        exp_stats = _layer_stats_full(state, "exp_avg")
        stats_json["exp_avg"] = {f"layer_{lid}": s for lid, s in exp_stats.items()}
    else:
        exp_stats = {}

    if not args.no_exp_avg_sq:
        print("Computing layer‑wise stats for exp_avg_sq …")
        sq_stats = _layer_stats_full(state, "exp_avg_sq")
        stats_json["exp_avg_sq"] = {f"layer_{lid}": s for lid, s in sq_stats.items()}
    else:
        sq_stats = {}

    # Dump JSON
    import json, math

    json_path = out_dir / "layer_stats.json"
    with json_path.open("w") as f:
        json.dump(stats_json, f, indent=2, allow_nan=False)
    print(f"  → layer stats written to {json_path}")

    # ------------------------------------------------------------------
    # Combined figure with subplots
    # ------------------------------------------------------------------
    if exp_stats or sq_stats:
        print("Creating combined mean/min/max figure …")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        def _plot_stats(ax, stats_dict, title):
            lids = sorted(stats_dict)
            means = [stats_dict[l]["mean"] for l in lids]
            mins = [stats_dict[l]["min"] for l in lids]
            maxs = [stats_dict[l]["max"] for l in lids]
            ax.plot(lids, means, label="mean", marker="o")
            ax.plot(lids, mins, label="min", linestyle="--")
            ax.plot(lids, maxs, label="max", linestyle="--")
            ax.set_title(title)
            ax.set_ylabel("value")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        if exp_stats:
            _plot_stats(axes[0], exp_stats, "|exp_avg| statistics per layer")
        else:
            axes[0].set_visible(False)

        if sq_stats:
            _plot_stats(axes[1], sq_stats, "exp_avg_sq statistics per layer")
        else:
            axes[1].set_visible(False)

        axes[-1].set_xlabel("Layer index")
        fig.tight_layout()
        combo_path = out_dir / "layer_stats_combined.png"
        fig.savefig(combo_path, dpi=150)
        plt.close(fig)
        print(f"  → saved: {combo_path}")

    # ------------------------------------------------------------------
    # Per‑parameter histograms & heatmaps
    # ------------------------------------------------------------------

    print("\nGenerating per‑parameter histograms and heatmaps …")

    hist_dir = out_dir / "per_param_histograms"
    hist_dir.mkdir(exist_ok=True, parents=True)
    heat_dir = out_dir / "per_param_heatmaps"
    heat_dir.mkdir(exist_ok=True, parents=True)

    for pname, p_state in state.items():
        safe_name = pname.replace("/", "_").replace(".", "_")

        # ------------------- Histogram(s) -------------------
        if not args.no_exp_avg and "exp_avg" in p_state:
            tensor = p_state["exp_avg"]
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                vals = tensor.detach().flatten().abs().cpu()
                _plot_hist_save(
                    vals,
                    f"|exp_avg| – {pname}",
                    hist_dir / f"hist_exp_avg_{safe_name}.png",
                )

                # Optional heatmap for 2‑D exp_avg
                if tensor.dim() == 2:
                    _plot_heatmap_2d(
                        tensor,
                        f"{pname} (exp_avg)",
                        heat_dir / f"heatmap_exp_avg_{safe_name}.png",
                    )

        if not args.no_exp_avg_sq and "exp_avg_sq" in p_state:
            tensor = p_state["exp_avg_sq"]
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                vals = tensor.detach().flatten().cpu()
                _plot_hist_save(
                    vals,
                    f"exp_avg_sq – {pname}",
                    hist_dir / f"hist_exp_avg_sq_{safe_name}.png",
                )

                # ----------------- Heatmap(s) -----------------
                if tensor.dim() == 2:
                    _plot_heatmap_2d(
                        tensor,
                        f"{pname} (exp_avg_sq)",
                        heat_dir / f"heatmap_{safe_name}.png",
                    )

    print("Per‑parameter histograms saved in", hist_dir)
    if not args.no_exp_avg_sq:
        print("Per‑parameter heatmaps saved in", heat_dir)


if __name__ == "__main__":
    main() 