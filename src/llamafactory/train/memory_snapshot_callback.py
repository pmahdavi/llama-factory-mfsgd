import os
import socket
import datetime
from typing import Optional

import torch
from transformers import TrainerCallback
from typing_extensions import override


class MemorySnapshotCallback(TrainerCallback):
    """Record GPU memory allocations over time and dump a snapshot at the end of training.

    The snapshot is saved as a ``*.pickle`` file inside the current run's ``output_dir``.
    Only the process with ``local_rank == 0`` will attempt to dump the snapshot to avoid
    concurrent writes when running under distributed training.
    """

    def __init__(self, max_entries: int = 100_000) -> None:  # noqa: D401
        super().__init__()
        self.max_entries = max_entries
        self.enabled = torch.cuda.is_available()
        self.snapshot_path: Optional[str] = None
        self._recording_started = False
        self._atexit_registered = False

    # ---------------------------------------------------------------------
    # TrainerCallback hooks
    # ---------------------------------------------------------------------
    @override
    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        if not self.enabled:
            return

        # We only want one process to write the snapshot in DDP setups to
        # prevent contention. ``local_process_index`` is 0 for the first
        # process on each node, while ``process_index`` is global. We choose
        # to record for the local process 0 of each node such that each node
        # can generate its own snapshot, which is often more informative than
        # picking a single global rank.
        if getattr(args, "local_process_index", 0) != 0:
            return

        # Start recording memory events.
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        self._recording_started = True

        # Build deterministic filename inside the run's output directory.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        host = socket.gethostname()
        filename = f"{host}_{timestamp}_mem_snapshot.pickle"
        self.snapshot_path = os.path.join(args.output_dir, filename)

        # Register an atexit handler so that we still dump a snapshot if the
        # training loop exits prematurely (e.g. due to an OutOfMemoryError).
        # The handler is only registered once to avoid duplicate dumps when
        # multiple callbacks might be instantiated (should not happen but we
        # are defensive).
        if not self._atexit_registered:
            import atexit

            def _handler(cb_self: "MemorySnapshotCallback") -> None:  # noqa: D401
                if cb_self._recording_started:
                    try:
                        if cb_self.snapshot_path is None:
                            # Fallback path if train_begin failed to build one.
                            cb_self.snapshot_path = os.path.join(args.output_dir, "mem_snapshot.pickle")
                        os.makedirs(args.output_dir, exist_ok=True)
                        torch.cuda.memory._dump_snapshot(cb_self.snapshot_path)
                    except Exception:
                        pass  # we are in interpreter-shutdown; be silent.
                    finally:
                        torch.cuda.memory._record_memory_history(enabled=None)

            atexit.register(_handler, self)
            self._atexit_registered = True

    @override
    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if not self._recording_started:
            return

        # Dump the snapshot and stop recording. Ignore any failure so that the
        # training script never crashes because of the profiler.
        try:
            assert self.snapshot_path is not None
            os.makedirs(args.output_dir, exist_ok=True)
            torch.cuda.memory._dump_snapshot(self.snapshot_path)
        except Exception as e:  # pragma: no cover
            # Best-effort logging: use print to avoid complications with logger setup.
            print(f"[MemorySnapshotCallback] Failed to dump snapshot: {e}")
        finally:
            # Whether or not we succeeded, disable recording.
            torch.cuda.memory._record_memory_history(enabled=None)
            self._recording_started = False 