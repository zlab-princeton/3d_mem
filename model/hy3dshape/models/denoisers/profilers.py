from __future__ import annotations
import time
import torch
import torch.nn as nn
from collections import defaultdict

try:
    import torch.cuda.nvtx as _nvtx
    _HAS_NVTX = True
except Exception:
    _HAS_NVTX = False

def _evt_pair():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None

class BlockTimer:
    """
    Attach forward/backward hooks to modules and measure time with CUDA events.
    This version CORRECTLY defers synchronization until results are requested.
    """
    def __init__(self, use_cuda_events: bool = True, use_nvtx: bool = True):
        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()
        self.use_nvtx = use_nvtx
        # We will now store a list of event pairs to be processed later
        self.event_pairs = defaultdict(lambda: {"fwd": [], "bwd": []})
        self.wall_times = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0, "cnt": 0})

    def _fwd_pre(self, name):
        def _hook(module, inputs):
            if self.use_nvtx: _nvtx.range_push(f"{name}/fwd")
            if self.use_cuda_events:
                s, e = _evt_pair()
                s.record()
                # Store the event pair for later processing
                self.event_pairs[name]["fwd"].append((s, e))
            else:
                module.__bt_t0_fwd = time.perf_counter()
        return _hook

    def _fwd_post(self, name):
        def _hook(module, inputs, output):
            if self.use_cuda_events:
                # Get the end event from the list and RECORD it. DO NOT SYNCHRONIZE.
                end_event = self.event_pairs[name]["fwd"][-1][1]
                end_event.record()
            else:
                dt = time.perf_counter() - getattr(module, "__bt_t0_fwd", time.perf_counter())
                t = self.wall_times[name]
                t["fwd"] += float(dt)
                t["cnt"] += 1
            if self.use_nvtx: _nvtx.range_pop()
        return _hook

    def _bwd_pre(self, name):
        def _hook(module, grad_input):
            if self.use_nvtx: _nvtx.range_push(f"{name}/bwd")
            if self.use_cuda_events:
                s, e = _evt_pair()
                s.record()
                self.event_pairs[name]["bwd"].append((s, e))
            else:
                module.__bt_t0_bwd = time.perf_counter()
        return _hook

    def _bwd_post(self, name):
        def _hook(module, grad_input, grad_output):
            if self.use_cuda_events:
                # Get the end event and RECORD it. DO NOT SYNCHRONIZE.
                end_event = self.event_pairs[name]["bwd"][-1][1]
                end_event.record()
            else:
                dt = time.perf_counter() - getattr(module, "__bt_t0_bwd", time.perf_counter())
                self.wall_times[name]["bwd"] += float(dt)
            if self.use_nvtx: _nvtx.range_pop()
        return _hook

    def attach(self, module: nn.Module, name: str):
        """Attach hooks; returns handles so caller can keep refs if desired."""
        h1 = module.register_forward_pre_hook(self._fwd_pre(name))
        h2 = module.register_forward_hook(self._fwd_post(name))
        h3 = module.register_full_backward_pre_hook(self._bwd_pre(name))
        h4 = module.register_full_backward_hook(self._bwd_post(name))
        return (h1, h2, h3, h4)

    def dump_and_reset(self, prefix: str | None = None):
        """Return dict of metrics and clear accumulators."""
        out = {}
        if self.use_cuda_events:
            # SYNCHRONIZE ONCE HERE, before calculating any times.
            torch.cuda.synchronize()
            for name, events in self.event_pairs.items():
                key = f"{prefix}/{name}" if prefix else name
                
                total_fwd_time = sum(s.elapsed_time(e) / 1000.0 for s, e in events["fwd"])
                total_bwd_time = sum(s.elapsed_time(e) / 1000.0 for s, e in events["bwd"])
                cnt = max(1, len(events["fwd"]))

                out[f"{key}/fwd_time_s"] = total_fwd_time
                out[f"{key}/bwd_time_s"] = total_bwd_time
                out[f"{key}/fwd_avg_ms"] = 1000.0 * total_fwd_time / cnt
                out[f"{key}/bwd_avg_ms"] = 1000.0 * total_bwd_time / max(1, len(events["bwd"]))
            self.event_pairs.clear()
        else:
            # Handle wall-clock times
            for name, times in self.wall_times.items():
                key = f"{prefix}/{name}" if prefix else name
                cnt = max(1, times["cnt"])
                out[f"{key}/fwd_time_s"] = times["fwd"]
                out[f"{key}/bwd_time_s"] = times["bwd"]
                out[f"{key}/fwd_avg_ms"]  = 1000.0 * times["fwd"] / cnt
                out[f"{key}/bwd_avg_ms"]  = 1000.0 * times["bwd"] / cnt
            self.wall_times.clear()

        return out