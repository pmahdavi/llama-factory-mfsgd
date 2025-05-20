import os
import socket
import datetime
from typing import Optional, TYPE_CHECKING

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing_extensions import override

from llamafactory.extras.logging import get_logger


if TYPE_CHECKING:
    from llamafactory.hparams import FinetuningArguments


logger = get_logger(__name__)


class MemorySnapshotCallback(TrainerCallback):
    """Record GPU memory allocations over time and dump a snapshot.

    The snapshot is saved as a ``*.pickle`` file inside the current run's ``output_dir``.
    Can be configured to dump early after a certain number of optimizer steps.
    """

    def __init__(
        self,
        max_entries: int = 100_000,
        start_recording_manually: bool = False,
        profile_memory_stop_step: Optional[int] = None,
        profile_memory_stop_accumulation_step: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_entries = max_entries
        self.start_recording_manually = start_recording_manually
        self.profile_memory_stop_step = profile_memory_stop_step
        self.profile_memory_stop_accumulation_step = profile_memory_stop_accumulation_step
        self.enabled = torch.cuda.is_available()
        self.snapshot_path: Optional[str] = None
        self._recording_started = False
        self._atexit_registered = False
        self._early_dump_done = False
        self._accumulation_count_at_first_step = 0

    def _prepare_snapshot_path(self, output_dir: str, step: Optional[int] = None) -> None:
        """Prepares the snapshot path, incorporating step if provided."""
        if not self.enabled:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        host = socket.gethostname()
        step_suffix = f"_step{step}" if step is not None else ""
        # Ensure output_dir exists
        os.makedirs(output_dir, exist_ok=True)
        self.snapshot_path = os.path.join(output_dir, f"{host}_{timestamp}{step_suffix}_mem_snapshot.pickle")
        logger.info(f"[MemorySnapshotCallback] Snapshot path set to: {self.snapshot_path}")

    def _do_dump_and_stop(self, output_dir: str, current_step: Optional[int] = None) -> None:
        """Dumps the memory snapshot and stops recording. Logs actions."""
        if not self._recording_started or not self.enabled:
            if self._early_dump_done:
                 logger.info("[MemorySnapshotCallback] Early dump already performed, skipping dump.")
            return

        logger.info(f"[MemorySnapshotCallback] Attempting to dump memory snapshot (current_step: {current_step}).")
        self._prepare_snapshot_path(output_dir, step=current_step)

        try:
            if self.snapshot_path is None:
                logger.warning("[MemorySnapshotCallback] Snapshot path is None, cannot dump.")
                return

            if hasattr(torch.cuda, "memory_summary"):
                logger.info("[MemorySnapshotCallback] GPU Memory Summary:")
                logger.info(torch.cuda.memory_summary())
            torch.cuda.memory._dump_snapshot(self.snapshot_path)
            logger.info(f"[MemorySnapshotCallback] Snapshot successfully dumped to {self.snapshot_path}.")
            if current_step is not None:
                self._early_dump_done = True
        except Exception as e:
            logger.warning(f"[MemorySnapshotCallback] Failed to dump snapshot: {e}")
        finally:
            logger.info("[MemorySnapshotCallback] Stopping CUDA memory history recording.")
            torch.cuda.memory._record_memory_history(enabled=None)
            self._recording_started = False

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.enabled or self._early_dump_done:
            return

        if getattr(args, "local_process_index", 0) != 0:
            logger.info("[MemorySnapshotCallback] Not on local_process_index 0, skipping recording setup.")
            self.enabled = False
            return

        self._prepare_snapshot_path(args.output_dir)

        should_start_here = True
        if self.start_recording_manually:
            logger.info("[MemorySnapshotCallback] Assuming memory history recording was started manually by `profile_memory_from_start`.")
            self._recording_started = True
            should_start_here = False
        
        if should_start_here:
            logger.info(f"[MemorySnapshotCallback] Starting CUDA memory history recording with max_entries={self.max_entries}")
            torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
            self._recording_started = True

        if not self._atexit_registered and self._recording_started:
            import atexit
            def _handler(callback_instance):
                if callback_instance._recording_started and not callback_instance._early_dump_done and callback_instance.enabled:
                    logger.warning("[MemorySnapshotCallback][atexit] Exiting, attempting to dump snapshot.")
                    if callback_instance.snapshot_path and os.path.exists(os.path.dirname(callback_instance.snapshot_path)):
                         callback_instance._do_dump_and_stop(os.path.dirname(callback_instance.snapshot_path))
                    else:
                         logger.warning("[MemorySnapshotCallback][atexit] Cannot determine output directory for dump.")
                elif callback_instance._recording_started:
                    torch.cuda.memory._record_memory_history(enabled=None)

            atexit.register(_handler, self)
            self._atexit_registered = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.enabled or not self._recording_started or self._early_dump_done:
            return

        if (
            self.profile_memory_stop_accumulation_step is not None
            and state.global_step == 0
        ):
            self._accumulation_count_at_first_step += 1
            logger.info(
                f"[MemorySnapshotCallback] Accumulation step {self._accumulation_count_at_first_step} "
                f"at global_step {state.global_step}."
            )
            if self._accumulation_count_at_first_step >= self.profile_memory_stop_accumulation_step:
                logger.info(
                    f"[MemorySnapshotCallback] Reached target accumulation count "
                    f"{self._accumulation_count_at_first_step} (limit {self.profile_memory_stop_accumulation_step}) "
                    f"at global_step {state.global_step}. Dumping snapshot and stopping."
                )
                self._do_dump_and_stop(args.output_dir, current_step=state.global_step)
                return

        if self.profile_memory_stop_step is not None and state.global_step >= self.profile_memory_stop_step:
            logger.info(
                f"[MemorySnapshotCallback] Reached target optimizer step {state.global_step} "
                f"(limit {self.profile_memory_stop_step}). Dumping snapshot and stopping."
            )
            self._do_dump_and_stop(args.output_dir, current_step=state.global_step)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.enabled or not self._recording_started or self._early_dump_done:
            return
        
        logger.info("[MemorySnapshotCallback] Training ended. Dumping final snapshot (if not dumped early).")
        self._do_dump_and_stop(args.output_dir) 