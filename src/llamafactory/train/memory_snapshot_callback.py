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

    def __init__(self, max_entries: int = 100_000, start_recording_manually: bool = False) -> None:  # noqa: D401
        super().__init__()
        self.max_entries = max_entries
        self.start_recording_manually = start_recording_manually
        self.enabled = torch.cuda.is_available()
        self.snapshot_path: Optional[str] = None
        self._recording_started = False
        self._atexit_registered = False
        self._step_count = 0 # Add step counter

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

        if not self.start_recording_manually:
            # Start recording memory events if not started manually by the user.
            print(f"[MemorySnapshotCallback] Starting CUDA memory history recording with max_entries={self.max_entries}")
            torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
            self._recording_started = True
        else:
            # If started manually, assume it's active. The callback will still handle dumping.
            # We need to check if it *is* actually active if possible, or trust the user.
            # For now, we'll trust the user has started it.
            # A check like `torch.cuda.memory._is_snapshot_collection_enabled()` is internal.
            print("[MemorySnapshotCallback] Assuming memory history recording was started manually by the user.")
            self._recording_started = True # Set to true so on_train_end and atexit will try to dump.

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
                            # Assuming 'args' from on_train_begin is accessible via closure for output_dir
                            # This part might need careful review if args is not reliably in scope
                            # For simplicity, we'll rely on self.snapshot_path being set.
                            # If it's critical, args would need to be passed to _handler or stored in self.
                            # However, the current structure sets snapshot_path before registering atexit.
                            # A more robust fallback might be a predefined path or disabling if path is None.
                            # For now, let's assume snapshot_path is usually set.
                            # If snapshot_path is None, it will use "mem_snapshot.pickle" in args.output_dir
                            # which might be an issue if args is not available.
                            # The original code uses `args.output_dir` directly in the fallback,
                            # which relies on closure. Let's keep that pattern.
                            pass # Original code has a fallback using args, which should work via closure.

                        # Ensure output directory exists
                        if cb_self.snapshot_path:
                            output_dir_for_atexit = os.path.dirname(cb_self.snapshot_path)
                            if not output_dir_for_atexit and hasattr(args, "output_dir"): # Fallback if only filename was set
                                output_dir_for_atexit = args.output_dir
                        elif hasattr(args, "output_dir"):
                            output_dir_for_atexit = args.output_dir
                        else: # Absolute last resort
                            output_dir_for_atexit = "."
                        os.makedirs(output_dir_for_atexit, exist_ok=True)
                        
                        # Use a default snapshot path if necessary, within the determined output_dir
                        current_snapshot_path = cb_self.snapshot_path
                        if not current_snapshot_path:
                             current_snapshot_path = os.path.join(output_dir_for_atexit, "mem_snapshot_atexit.pickle")

                        # Print memory summary
                        if hasattr(torch.cuda, "memory_summary"): # Check if available
                            print("[MemorySnapshotCallback][atexit] GPU Memory Summary:")
                            try:
                                print(torch.cuda.memory_summary())
                            except Exception as e_sum:
                                print(f"[MemorySnapshotCallback][atexit] Failed to print memory summary: {e_sum}")
                        
                        print(f"[MemorySnapshotCallback][atexit] Dumping memory snapshot to: {current_snapshot_path}")
                        torch.cuda.memory._dump_snapshot(current_snapshot_path)
                    except Exception as e_dump:
                        print(f"[MemorySnapshotCallback][atexit] Failed to dump snapshot or print summary: {e_dump}")
                        pass  # we are in interpreter-shutdown; be silent on snapshot dump failure
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
            # Print memory summary
            if hasattr(torch.cuda, "memory_summary"): # Check if available
                print("[MemorySnapshotCallback] GPU Memory Summary:")
                print(torch.cuda.memory_summary())
            torch.cuda.memory._dump_snapshot(self.snapshot_path)
        except Exception as e:  # pragma: no cover
            # Best-effort logging: use print to avoid complications with logger setup.
            print(f"[MemorySnapshotCallback] Failed to dump snapshot: {e}")
        finally:
            # Whether or not we succeeded, disable recording.
            torch.cuda.memory._record_memory_history(enabled=None)
            self._recording_started = False 

    @override
    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if not self._recording_started:
            return

        self._step_count += 1
        if self._step_count >= 10:
            if getattr(args, "local_process_index", 0) == 0: # Ensure only main process dumps
                print(f"[MemorySnapshotCallback] Reached {self._step_count} steps. Dumping memory snapshot.")
                try:
                    assert self.snapshot_path is not None
                    os.makedirs(args.output_dir, exist_ok=True)
                    if hasattr(torch.cuda, "memory_summary"):
                        print("[MemorySnapshotCallback] GPU Memory Summary (at step 10):")
                        print(torch.cuda.memory_summary())
                    torch.cuda.memory._dump_snapshot(self.snapshot_path)
                    print(f"[MemorySnapshotCallback] Snapshot dumped to {self.snapshot_path}")
                except Exception as e:
                    print(f"[MemorySnapshotCallback] Failed to dump snapshot at step 10: {e}")
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)
                    self._recording_started = False
                    print("[MemorySnapshotCallback] Memory history recording stopped after 10 steps.")
                    # Potentially set control.should_training_stop = True if profiling is the main goal
                    # For now, let's assume training continues without profiling. 