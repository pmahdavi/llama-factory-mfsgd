#!/usr/bin/env python3
"""dump_optim_state.py

Reload a finished LLaMA‑Factory training run (DeepSpeed ZeRO‑3) and
export a *consolidated* copy of
  • the model weights (already gathered by DeepSpeed on save)
  • the optimiser state_dict (mapped to parameter names, single file)

The script purposefully performs **zero** training steps – it only
forces the Trainer/DeepSpeed engine to load the last checkpoint so that
all ranks collectively rebuild the FP32 optimiser on rank‑0.  After that
we just serialise the state.

Typical usage (same world‑size as the original training):

    # 2‑GPU example
    python -m torch.distributed.run \
        --nproc_per_node 2 scripts/dump_optim_state.py \
        --run_dir /scratch/xxx/my_run_dir \
        --checkpoint checkpoint-1318

If you trained on a single GPU you can simply run:

    python scripts/dump_optim_state.py --run_dir /scratch/.../my_run_dir

The script assumes the original `training_config.yaml` lives directly in
`<run_dir>/training_config.yaml` (the layout produced by `run.py`).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import Trainer
from llamafactory.hparams.parser import get_train_args
from llamafactory.model import load_model, load_tokenizer
import yaml  # local import to avoid extra dependency if unused elsewhere

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_latest_ckpt(run_dir: Path) -> Path:
    """Return the *numerically* latest `checkpoint-*` directory."""
    ckpts = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* folders found under {run_dir}")
    # sort by global step suffix (after the dash)
    latest = max(ckpts, key=lambda p: int(p.name.split("-", 1)[1]))
    return latest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():  # noqa: C901 – keep linear for readability
    parser = argparse.ArgumentParser(description="Consolidate optimiser state from a DeepSpeed run")
    parser.add_argument("--run_dir", required=True, type=Path, help="Path to the original training run directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Name of checkpoint folder to load (e.g. checkpoint-1318). If omitted, uses the latest one.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Destination folder for consolidated files (default: <run_dir>/export_full_state)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    # ---------------------------------------------------------------------
    # Resolve checkpoint & output paths
    # ---------------------------------------------------------------------
    if args.checkpoint is None:
        ckpt_dir = _find_latest_ckpt(run_dir)
    else:
        ckpt_dir = run_dir / args.checkpoint
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if args.output_dir is None:
        # Append checkpoint name to the default output directory
        output_dir_name = f"export_full_state_{ckpt_dir.name}"
        output_dir: Path = (run_dir / output_dir_name).expanduser().resolve()
    else:
        # Use user-provided output directory
        output_dir: Path = args.output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("================ Consolidation summary ================")
    print(f"Run dir        : {run_dir}")
    print(f"Checkpoint     : {ckpt_dir}")
    print(f"Output dir     : {output_dir}")
    print("=======================================================")

    # ---------------------------------------------------------------------
    # Load original training arguments
    # ---------------------------------------------------------------------
    cfg_path = run_dir / "training_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"training_config.yaml not found at {cfg_path}")

    cfg_dict = yaml.safe_load(cfg_path.read_text())
    model_args, data_args, training_args, finetuning_args, gen_args = get_train_args(cfg_dict)

    # Modify arguments so that Trainer builds but performs zero optimisation
    training_args.output_dir = str(output_dir)
    training_args.resume_from_checkpoint = str(ckpt_dir)
    training_args.num_train_epochs = 0
    training_args.max_steps = 0
    training_args.do_train = True  # still need the train path so DS loads
    # Ensure we never accidentally overwrite things mid‑script
    training_args.save_steps = 10**9
    # Disable evaluation requirement (original config had eval_strategy="steps")
    if hasattr(training_args, "evaluation_strategy"):
        training_args.evaluation_strategy = "no"
    if hasattr(training_args, "eval_strategy"):
        training_args.eval_strategy = "no"

    # ---------------------------------------------------------------------
    # Build model & tokenizer (unwrapped)
    # ---------------------------------------------------------------------
    tok_mod = load_tokenizer(model_args)
    # Need trainable params so DeepSpeed can build the base optimizer
    model = load_model(tok_mod["tokenizer"], model_args, finetuning_args, is_trainable=True)

    # Dummy dataset so Trainer initialises happily
    class _DummyDS(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"input_ids": torch.tensor([0])}

    dummy_ds = _DummyDS()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dummy_ds,
        tokenizer=tok_mod["tokenizer"],
    )

    # ---------------------------------------------------------------------
    # Trigger DeepSpeed engine loading (zero training steps)
    # ---------------------------------------------------------------------
    trainer.train(resume_from_checkpoint=str(ckpt_dir))

    # ---------------------------------------------------------------------
    # ❶ Build a *fully unsharded* (CPU) optimiser state_dict
    #     – iterate over every parameter, gather its FP32 optimiser tensors
    #     – construct a normal PyTorch‑style state dict keyed by parameter names
    # ---------------------------------------------------------------------
    if hasattr(trainer, "deepspeed"):
        import torch.distributed as dist

        ds_opt = trainer.deepspeed.optimizer
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

        full_state: dict[str, dict] | None = {} if trainer.is_world_process_zero() else None
        param_names: list[str] = []

        for name, param in unwrapped_model.named_parameters():
            # ensure every rank participates in the collective all‑gather
            opt_entry: dict[str, torch.Tensor | int | float] = {}

            # Under ZeRO‑3 the per‑param optimiser tensors live inside the flat group
            group_idx, *_ = ds_opt.grad_position[ds_opt.get_param_id(param)]
            fp32_flat_param = ds_opt.fp32_partitioned_groups_flat[group_idx]
            base_state = ds_opt.optimizer.state[fp32_flat_param]

            for key, example_val in base_state.items():
                is_tensor = torch.is_tensor(example_val)
                is_collectable = is_tensor and example_val.dim() > 0 and example_val.numel() > 0

                if is_collectable:
                    # tensor with real elements → all‑gather from ZeRO
                    gathered = ds_opt.get_full_hp_param(param, optim_state_key=key)
                    if trainer.is_world_process_zero():
                        opt_entry[key] = gathered.cpu()
                else:
                    # scalar tensors (0‑dim) or non‑tensors: value identical on every rank
                    if trainer.is_world_process_zero():
                        opt_entry[key] = example_val.clone().cpu() if is_tensor else example_val
                    # keep ranks in sync so loop length identical
                    dist.barrier()

            if trainer.is_world_process_zero():
                full_state[name] = opt_entry
                param_names.append(name)

        # Only rank‑0 serialises to disk
        if trainer.is_world_process_zero():
            # minimal param_group: keep hyper‑params from first group
            base_pg = ds_opt.optimizer.param_groups[0].copy()
            base_pg["params"] = param_names
            # prune objects that are not JSON/torch‑save friendly (e.g. tensor refs)
            base_pg.pop("params_flat", None)

            torch.save({"state": full_state, "param_groups": [base_pg]}, output_dir / "optimizer_full.pt")
            print(f"✓ Fully‑unsharded optimiser state saved to {output_dir/'optimizer_full.pt'}")

    # ---------------------------------------------------------------------
    # Use trainer.save_model() to trigger checkpoint saving mechanism
    # This *might* save a consolidated optimizer state directly.
    # ---------------------------------------------------------------------
    print(f"\n--- Attempting to save state using trainer.save_model({output_dir}) ---")
    try:
        trainer.save_model(output_dir) 
        print(f"✓ trainer.save_model() completed. Check contents of {output_dir}")
        print(f"  Look for 'optimizer.pt' or sharded state inside subdirectories.")
    except Exception as e:
        print(f"[ERROR] trainer.save_model() failed: {e}")

    # Optional: Save unwrapped model weights separately if needed, but might be redundant
    # print("[INFO] Saving unwrapped model weights separately...")
    # unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    # try:
    #     unwrapped_model.save_pretrained(output_dir, safe_serialization=False)
    #     tok_mod["tokenizer"].save_pretrained(output_dir)
    #     print(f"✓ Unwrapped model and tokenizer saved to {output_dir}")
    # except Exception as e:
    #     print(f"[WARN] Could not save unwrapped model/tokenizer separately to {output_dir}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main() 