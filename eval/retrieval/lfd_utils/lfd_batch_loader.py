import sys
from pathlib import Path
import torch
import numpy as np

# Adjust this path depending on where your 'load_data' module actually lives relative to this file
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from load_data.interface import LoadData
except ImportError:
    LoadData = None

_SUFFIX = "_q8_v1.8.art"

class LFDBatchLoader:
    """
    Unified loader for LFD tensors.
    """
    def __init__(self, device="cuda"):
        if LoadData is None:
            raise ImportError("Could not import LoadData from load_data.interface")
        self._cache = {}
        self._dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self._ld = LoadData()

    def _prefix(self, feat_dir: Path):
        arts = list(feat_dir.glob(f"*{_SUFFIX}"))
        return None if not arts else str(arts[0]).replace(_SUFFIX, "")

    def get_global_tensors(self):
        """Returns the q8 table and alignment table (always int32)."""
        q8_np, al_np, _, _, _, _ = self._ld.run("")
        return {
            "q8": torch.from_numpy(q8_np).to(torch.int32).to(self._dev).reshape(256, 256),
            "al10": torch.from_numpy(al_np).to(torch.int32).to(self._dev).reshape(60, 20),
        }

    def get(self, feat_dir: Path, compact: bool = True):
        """
        Loads LFD features.
        Args:
            compact (bool): If True, loads features as uint8 (optimized for retrieval).
                            If False, loads as int32 (legacy/math compatible).
        """
        prefix = self._prefix(feat_dir)
        if prefix is None:
            return None
        if prefix in self._cache:
            return self._cache[prefix]

        q8, al, art, fd, cir, ecc = self._ld.run(prefix)
        
        dtype = torch.uint8 if compact else torch.int32

        tensors = {
            "q8":   torch.from_numpy(q8).to(torch.int32).to(self._dev).reshape(256, 256),
            "al10": torch.from_numpy(al).to(torch.int32).to(self._dev).reshape(60, 20),
            "art":  torch.from_numpy(art[None]).to(dtype).to(self._dev).reshape(1, 10, 10, 35),
            "fd":   torch.from_numpy(fd[None]).to(dtype).to(self._dev).reshape(1, 10, 10, 10),
            "cir":  torch.from_numpy(cir[None]).to(dtype).to(self._dev).reshape(1, 10, 10),
            "ecc":  torch.from_numpy(ecc[None]).to(dtype).to(self._dev).reshape(1, 10, 10),
        }
        
        self._cache[prefix] = tensors
        return tensors