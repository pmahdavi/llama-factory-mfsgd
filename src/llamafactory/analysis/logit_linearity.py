"""logit_linearity.py
Implementation of local logit linearity analysis for fully fine-tuned LLaMA-Factory checkpoints.

The analysis estimates how locally *linear* the model’s last-token logit mapping is with respect to
small perturbations in parameter space.  For a collection of random directions and radii, we compare
true logit deltas against their first-order predictions obtained via the Jacobian–vector product (JVP)
and report aggregate relative error and cosine similarity statistics.

Typical CLI usage (through `llamafactory-cli`):

    llamafactory-cli analyze-logits path/to/config.yaml

The YAML file follows the usual LLaMA-Factory convention.  It may either contain analysis-specific
arguments at the top level or nest them under an `analysis:` section.  Unknown keys are ignored.
A minimal example is provided in `configs/analysis/llama_math_linearity.yaml`.

Interpretation of the JSON report created in `output_dir`:
    • For every probe radius we store the mean/σ of relative error and cosine similarity over all
      (direction, example) pairs.
    • `linearity_radius` denotes the largest radius whose metrics satisfy both
        – mean relative error  ≤  `relerr_threshold`  AND
        – mean cosine similarity ≥ `cos_threshold`.

NOTE:  The implementation relies only on `torch`, `transformers`, and `datasets`.  Heavy-weight
packages are deliberately avoided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

from ..extras.logging import get_logger
from ..extras.misc import get_current_device

hf_logging.set_verbosity_error()  # silence HF warnings for clean CLI
logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


def _default_radii() -> List[float]:
    return [0.1, 0.3, 1.0]


@dataclass
class AnalysisConfig:
    """Container for all analysis hyper-parameters."""

    # model / data ----------------------------------------------------------------
    model_name_or_path: str = field(metadata={"help": "Path or HF id of the fully fine-tuned checkpoint."})
    dataset: Optional[str] = field(
        default=None, metadata={"help": "HF dataset name to use as probe set (optional)."}
    )
    split: str = field(default="train", metadata={"help": "Dataset split to sample from."})
    text_field: Optional[str] = field(
        default=None, metadata={"help": "Column within the dataset containing the text prompt."}
    )

    max_samples: int = field(default=64, metadata={"help": "Number of probe prompts to sample."})
    cutoff_len: int = field(default=1024, metadata={"help": "Maximum token length per example."})

    # output ----------------------------------------------------------------------
    output_dir: str = field(default="logit_linearity", metadata={"help": "Directory to write the report."})
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite output directory if it already exists."}
    )
    report_filename: str = field(
        default="logit_linearity_report.json", metadata={"help": "JSON filename for metrics."}
    )

    # perturbation hyper-parameters ------------------------------------------------
    perturbation_std: float = field(default=0.001, metadata={"help": "Std used to sample directions."})
    num_directions: int = field(default=8, metadata={"help": "Number of random directions."})
    radii: List[float] = field(default_factory=_default_radii, metadata={"help": "Probe radii."})

    # parameter subset ------------------------------------------------------------
    param_scope: str = field(
        default="full", metadata={"help": 'Either "full" or "last_blocks:k" to restrict params.'}
    )

    # evaluation criteria ---------------------------------------------------------
    cos_threshold: float = field(default=0.95)
    relerr_threshold: float = field(default=0.2)

    # misc ------------------------------------------------------------------------
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})

    @staticmethod
    def from_yaml(path: str) -> "AnalysisConfig":
        """Load from YAML file accepting both flat or nested (`analysis:`) structure."""
        cfg = OmegaConf.load(path)
        cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        # Merge nested analysis section if present
        analysis_section = cfg_dict.pop("analysis", {}) if isinstance(cfg_dict, Mapping) else {}
        merged: Dict[str, Any] = {**cfg_dict, **analysis_section}

        # filter out unexpected keys
        allowed = {f.name for f in dataclass_fields(AnalysisConfig)}
        clean: Dict[str, Any] = {k: v for k, v in merged.items() if k in allowed}
        # type: ignore[arg-type]
        return AnalysisConfig(**clean)


def dataclass_fields(cls):  # small helper without importing dataclasses.fields repeatedly
    return getattr(cls, "__dataclass_fields__")  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def _select_param_indices(model, scope: str) -> List[str]:
    """Return parameter names that should be perturbed according to *scope*."""
    if scope == "full":
        return [n for n, _ in model.named_parameters()]

    if scope.startswith("last_blocks:"):
        try:
            k = int(scope.split(":", 1)[1])
        except ValueError:
            raise ValueError("Invalid param_scope format. Use 'last_blocks:k'.")
        # heuristic: identify blocks by common naming e.g. '.layers.' or '.h.' etc.
        layer_names = [name for name, _ in model.named_parameters() if ".layers." in name or ".layer." in name or ".h." in name]
        if not layer_names:
            return [n for n, _ in model.named_parameters()]  # fallback to full
        # sort by layer index (simple parse of first int encountered after pattern)
        def _extract_idx(n):
            for part in n.split('.'):
                if part.isdigit():
                    return int(part)
            return -1
        sorted_pairs = sorted({n: _extract_idx(n) for n in layer_names}.items(), key=lambda kv: kv[1])
        last_indices = {n for n, idx in sorted_pairs[-k:]}
        return [n for n in last_indices]

    raise ValueError(f"Unknown param_scope: {scope}")


def _sample_directions(model, names: List[str], std: float, num: int, device: torch.device):
    """Sample *num* Gaussian directions restricted to given parameter *names*."""
    directions: List[Dict[str, torch.Tensor]] = []
    param_tensors = {n: p.detach() for n, p in model.named_parameters() if n in names}
    for _ in range(num):
        dir_dict = {}
        # accumulate squared norm for global normalisation
        total_sq_norm = torch.zeros((), device=device)
        for n, base in param_tensors.items():
            rnd = torch.randn_like(base, device=device) * std
            dir_dict[n] = rnd
            total_sq_norm += (rnd ** 2).sum()
        total_norm = torch.sqrt(total_sq_norm) + 1e-12
        for n in dir_dict:
            dir_dict[n] = dir_dict[n] / total_norm  # unit direction
        directions.append(dir_dict)
    return directions


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------


def _forward_last_token_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[:, -1, :]  # (batch, vocab)


def _apply_param_delta(model, delta: Mapping[str, torch.Tensor], scale: float = 1.0):
    """In-place add *scale* × *delta* to parameters."""
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if name in delta:
                tensor.add_(delta[name] * scale)


@torch.no_grad()
def _compute_true_delta(model, delta: Mapping[str, torch.Tensor], radius: float, batch_inputs):
    input_ids, attention_mask = batch_inputs
    baseline = _forward_last_token_logits(model, input_ids, attention_mask)
    _apply_param_delta(model, delta, radius)
    perturbed = _forward_last_token_logits(model, input_ids, attention_mask)
    _apply_param_delta(model, delta, -radius)  # revert
    return perturbed - baseline, baseline


def _compute_pred_delta_jvp(
    model, delta: Mapping[str, torch.Tensor], radius: float, batch_inputs
):
    """Use torch.func.jvp to obtain first-order prediction of the logit change."""
    input_ids, attention_mask = batch_inputs

    # Convert model to functional form once to avoid recompilation
    from torch.func import functional_call, jvp

    params = {k: v.detach().requires_grad_(True) for k, v in model.named_parameters()}

    def _func(p):
        return functional_call(model, p, (input_ids,), {"attention_mask": attention_mask}).logits[:, -1, :]

    # jvp requires tangents to be same structure as params
    tangent = {k: delta[k] * radius if k in delta else torch.zeros_like(v) for k, v in params.items()}
    _, pred = jvp(_func, (params,), (tangent,))  # type: ignore[arg-type]
    return pred


def _aggregate_stats(rel_errs: List[float], cos_sims: List[float]):
    import numpy as np

    arr1 = np.array(rel_errs)
    arr2 = np.array(cos_sims)
    return {
        "mean_relative_error": float(arr1.mean()),
        "std_relative_error": float(arr1.std()),
        "mean_cosine_similarity": float(arr2.mean()),
        "std_cosine_similarity": float(arr2.std()),
    }


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None):  # noqa: C901 complexity acceptable
    parser = argparse.ArgumentParser(description="Local logit linearity analysis")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args, extras = parser.parse_known_args(argv)
    if extras:
        logger.warning_rank0(f"Unused CLI args: {extras}")

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    cfg = AnalysisConfig.from_yaml(args.config)
    set_seed(cfg.seed)

    # prepare output dir
    if os.path.exists(cfg.output_dir):
        if cfg.overwrite_output_dir:
            logger.warning_rank0(f"Overwriting existing output dir: {cfg.output_dir}")
        else:
            raise FileExistsError(f"output_dir {cfg.output_dir} already exists (pass overwrite_output_dir: true).")
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = get_current_device()
    logger.info_rank0(f"Using device: {device} (may be CPU if CUDA unavailable)")

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"  # ensure left padding for causal

    _bf16_ok = False
    if torch.cuda.is_available():
        _bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    torch_dtype = torch.bfloat16 if _bf16_ok else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch_dtype, device_map=None)
    except Exception as e:
        logger.warning_rank0(f"Failed to load with dtype {torch_dtype}: {e}. Falling back to float32.")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.float32, device_map=None)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False  # safety, we only use JVP

    # ------------------------------------------------------------------
    # Load / build probe dataset
    # ------------------------------------------------------------------
    if cfg.dataset is not None:
        ds = load_dataset(cfg.dataset, split=cfg.split)
        if cfg.text_field is None:
            guess_field = next(iter(ds.column_names))
            logger.warning_rank0(f"'text_field' not provided; defaulting to '{guess_field}'.")
            cfg.text_field = guess_field
        ds_slice = ds.select(range(min(cfg.max_samples, len(ds))))
        texts = [str(x[cfg.text_field]) for x in ds_slice]
    else:
        logger.warning_rank0("No dataset specified – using fallback toy prompts.")
        texts = [
            "Hello, my name is",
            "The capital of France is",
            "The quick brown fox",
            "In 2025, AI systems will",
        ]
        texts = texts[: cfg.max_samples]

    encodings = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=cfg.cutoff_len,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # ------------------------------------------------------------------
    # Prepare random directions
    # ------------------------------------------------------------------
    names_to_perturb = _select_param_indices(model, cfg.param_scope)
    directions = _sample_directions(model, names_to_perturb, cfg.perturbation_std, cfg.num_directions, device)

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------
    metrics: Dict[float, Dict[str, List[float]]] = {
        r: {"rel_err": [], "cos_sim": []} for r in cfg.radii
    }

    batch_inputs = (input_ids, attention_mask)

    for d_idx, delta in enumerate(directions):
        logger.info_rank0(f"Processing direction {d_idx + 1}/{cfg.num_directions}")
        for radius in cfg.radii:
            true_delta, _ = _compute_true_delta(model, delta, radius, batch_inputs)
            pred_delta = _compute_pred_delta_jvp(model, delta, radius, batch_inputs)
            # Flatten to compute metrics
            t_vec = true_delta.flatten()
            p_vec = pred_delta.flatten()
            rel_err = (t_vec - p_vec).norm() / (t_vec.norm() + 1e-12)
            cos_sim = F.cosine_similarity(t_vec, p_vec, dim=0)
            metrics[radius]["rel_err"].append(rel_err.item())
            metrics[radius]["cos_sim"].append(cos_sim.item())

    # ------------------------------------------------------------------
    # Aggregate statistics & derive linearity radius
    # ------------------------------------------------------------------
    report_metrics: Dict[str, Any] = {}
    linearity_radius = None
    for r in sorted(cfg.radii):
        stats = _aggregate_stats(metrics[r]["rel_err"], metrics[r]["cos_sim"])
        report_metrics[str(r)] = stats
        if (
            stats["mean_relative_error"] <= cfg.relerr_threshold
            and stats["mean_cosine_similarity"] >= cfg.cos_threshold
        ):
            linearity_radius = r  # largest radius satisfying (because r sorted ascending)

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report = {
        "config": asdict(cfg),
        "metrics_per_radius": report_metrics,
        "linearity_radius": linearity_radius,
    }

    report_path = os.path.join(cfg.output_dir, cfg.report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info_rank0(f"Analysis completed. Report saved to {report_path}.")

    # Print summary to stdout
    print("===== Logit Linearity Analysis Summary =====")
    print(json.dumps({"linearity_radius": linearity_radius, "metrics_per_radius": report_metrics}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])