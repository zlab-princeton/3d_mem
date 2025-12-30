# force_resume_counters.py
import math, pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

class ForceResumeCounters(pl.Callback):
    def __init__(self, get_offset_fn): self.get_offset_fn = get_offset_fn
    def on_fit_start(self, trainer, pl_module):
        offset = int(self.get_offset_fn(pl_module) or 0)
        if offset > 0:
            try: trainer.global_step = offset
            except: pass
            try: trainer._global_step = offset
            except: pass
            rank_zero_info(f"[force-resume] set global_step={offset}")