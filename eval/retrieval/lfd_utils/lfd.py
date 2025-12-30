import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from .loader import LFDBatchLoader


@torch.no_grad()
def _lut_take(q8_lut_flat: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    # Indexing: idx = (src << 8) | tgt. src/tgt are uint8, returns int32 dists
    idx = (src.to(torch.int64) << 8) | tgt.to(torch.int64)
    return torch.take(q8_lut_flat, idx)

@torch.no_grad()
def calculate_lfd_distance_streamed(
    q8_table, align_10,
    src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
    tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8
) -> torch.Tensor:
    """
    Computes min distance between source (1 sample) and targets (N samples).
    Expects inputs as uint8 (compact) or int32.
    """
    q8_lut = q8_table.reshape(-1).contiguous()
    
    # Expand dims for broadcasting:
    # Src: (1, 10, 10, 1, 1, {F})
    # Tgt: (N, 1, 1, 10, 10, {F})
    sA = src_ArtCoeff.unsqueeze(3).unsqueeze(4)
    tA = tgt_ArtCoeff.unsqueeze(1).unsqueeze(2)
    sF = src_FdCoeff_q8.unsqueeze(3).unsqueeze(4)
    tF = tgt_FdCoeff_q8.unsqueeze(1).unsqueeze(2)
    sC = src_CirCoeff_q8.unsqueeze(3).unsqueeze(4)
    tC = tgt_CirCoeff_q8.unsqueeze(1).unsqueeze(2)
    sE = src_EccCoeff_q8.unsqueeze(3).unsqueeze(4)
    tE = tgt_EccCoeff_q8.unsqueeze(1).unsqueeze(2)

    # Accumulate costs (int32)
    art_cost = torch.zeros((tA.shape[0], 10, 10, 10, 10), dtype=torch.int32, device=q8_lut.device)
    for k in range(35): 
        art_cost.add_(_lut_take(q8_lut, sA[..., k], tA[..., k]).to(torch.int32))

    fd_cost = torch.zeros_like(art_cost)
    for k in range(10):
        fd_cost.add_(_lut_take(q8_lut, sF[..., k], tF[..., k]).to(torch.int32))

    cir_cost = _lut_take(q8_lut, sC, tC).to(torch.int32)
    ecc_cost = _lut_take(q8_lut, sE, tE).to(torch.int32)

    # Weighted Sum
    cost = art_cost
    cost.add_(fd_cost << 1)   # * 2
    cost.add_(cir_cost << 1)  # * 2
    cost.add_(ecc_cost)

    # Minimize over alignment
    Lbatch = cost.shape[0]
    best = torch.full((Lbatch,), 2**31 - 1, dtype=torch.int32, device=cost.device)

    for k in range(60):
        src_idx = align_10[k, :10].to(torch.long)
        selected = cost.index_select(dim=2, index=src_idx)
        diag_sum = selected.diagonal(dim1=2, dim2=4).sum(dim=-1) # (L, 10, 10)
        k_min = diag_sum.view(Lbatch, -1).min(dim=1).values
        best = torch.minimum(best, k_min)

    return best

@torch.no_grad()
def min_topk_over_targets(
    q8_table, align_10,
    src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
    tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8,
    top_k: int = 1, tgt_chunk: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming Top-K implementation."""
    M = tgt_ArtCoeff.shape[0]
    device = tgt_ArtCoeff.device
    
    best_val = torch.full((top_k,), 2**31 - 1, dtype=torch.int32, device=device)
    best_idx = torch.full((top_k,), -1, dtype=torch.int32, device=device)

    start = 0
    while start < M:
        end = min(start + tgt_chunk, M)
        d = calculate_lfd_distance_streamed(
            q8_table, align_10,
            src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
            tgt_ArtCoeff[start:end], tgt_FdCoeff_q8[start:end],
            tgt_CirCoeff_q8[start:end], tgt_EccCoeff_q8[start:end]
        )

        # Standard TopK merge logic
        v, i = torch.topk(d, k=min(top_k, d.numel()), largest=False)
        combined_v = torch.cat([best_val, v])
        combined_i = torch.cat([best_idx, (start + i).to(best_idx.dtype)])
        
        sel = torch.topk(combined_v, k=top_k, largest=False)
        best_val = combined_v[sel.indices]
        best_idx = combined_i[sel.indices]
        
        start = end

    return best_val, best_idx


class FastLFDMetric:
    """Wrapper that manages DB pointers and computes distances."""
    def __init__(self, precomputed_db_paths: Dict[str, Path], device="cuda", tgt_chunk: int = 256):
        self.device = torch.device(device)
        self.tgt_chunk = int(tgt_chunk)
        self.query_loader = LFDBatchLoader(self.device)

        # Load Reference DB (compact uint8)
        print(f"Loading DB from {precomputed_db_paths['art'].parent}...")
        self.ref_art = torch.load(precomputed_db_paths["art"]).to(torch.uint8).to(self.device)
        self.ref_fd  = torch.load(precomputed_db_paths["fd"]).to(torch.uint8).to(self.device)
        self.ref_cir = torch.load(precomputed_db_paths["cir"]).to(torch.uint8).to(self.device)
        self.ref_ecc = torch.load(precomputed_db_paths["ecc"]).to(torch.uint8).to(self.device)
        
        with open(precomputed_db_paths["meta"], "r") as f:
            self.ref_meta = json.load(f)

        # Cache Globals (int32)
        g = self.query_loader.get_global_tensors()
        self.glob_q8 = g["q8"]
        self.glob_al = g["al10"]

    def get_global_tensors(self):
        return {"q8": self.glob_q8, "al10": self.glob_al}

    def distance_from_tensors(self, query_tensors: Dict, top_k: int) -> List[Dict]:
        """
        Computes distance using pre-loaded query tensors.
        """
        # Ensure query is compact or cast it
        qt_art = query_tensors["art"].to(torch.uint8)
        qt_fd  = query_tensors["fd"].to(torch.uint8)
        qt_cir = query_tensors["cir"].to(torch.uint8)
        qt_ecc = query_tensors["ecc"].to(torch.uint8)
        
        # Globals
        q8   = query_tensors.get("q8", self.glob_q8)
        al10 = query_tensors.get("al10", self.glob_al)

        vals, idxs = min_topk_over_targets(
            q8, al10,
            qt_art, qt_fd, qt_cir, qt_ecc,
            self.ref_art, self.ref_fd, self.ref_cir, self.ref_ecc,
            top_k=top_k, tgt_chunk=self.tgt_chunk
        )
        return self._format_results(vals, idxs, top_k)

    def _format_results(self, vals: torch.Tensor, idxs: torch.Tensor, top_k: int) -> List[Dict]:
        vals = vals.cpu().tolist()
        idxs = idxs.cpu().tolist()
        if top_k == 1:
            vals, idxs = [vals], [idxs]
            
        out = []
        for i, v in zip(idxs, vals):
            if i < 0: continue # uninitialized
            meta = self.ref_meta[int(i)]
            out.append({
                "model_id":    meta.get("model_id"),
                "category_id": meta.get("category_id"),
                "distance":    float(v),
            })
        return out