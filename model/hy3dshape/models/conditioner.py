import torch
import torch.nn as nn
from typing import List, Union, Optional

from hy3dshape.models.embedders import ClassEmbedder, TextEmbedder, ImageEmbedder

def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensure embeddings are (B, T, D)."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(x.shape)}")

class UnconditionalEmbedder(nn.Module):
    """Placeholder for unconditional training."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.output_key = None
    def forward(self, **kwargs):
        return {}

class MultiConditioner(nn.Module):
    def __init__(
        self, 
        type: str,
        num_classes: int = None,
        text_embed_dim: int = None,
        images_cls_only: bool = False,
        l2_normalized: bool = False,
        p_uncond: float = 0.0
    ):
        super().__init__()
        self.type = type
        self.p_uncond = float(p_uncond)
        self.output_key = None
        
        # Default to 512 (CLIP ViT-B/16 dim) if not specified
        self.embed_dim = text_embed_dim if text_embed_dim is not None else 512  

        if type == "uncond":
            self.embedder = UnconditionalEmbedder()
            self.output_key = None
            
        elif type == "class":
            if num_classes is None or text_embed_dim is None:
                raise ValueError("`num_classes` and `text_embed_dim` required for class conditioning.")
            self.null_id = num_classes
            # Reserve one extra slot for null_id (uncond)
            self.embedder = ClassEmbedder(num_classes + 1, embed_dim=text_embed_dim)
            self.output_key = 'main'
            
        elif type == "text":
            self.embedder = TextEmbedder()
            self.output_key = 'main'
            
        elif type == "image":
            self.embedder = ImageEmbedder(return_cls_only=images_cls_only, l2_normalize=l2_normalized)
            self.output_key = 'main'
            
        else:
            raise ValueError(f"Unknown conditioner type: {type}")

    def unconditional_embedding(self, batch_size: int, device=None):
        """Returns the 'null' embedding used for Classifier-Free Guidance."""
        if self.type == "uncond":
            return {}
        
        device = device if device is not None else next(self.embedder.parameters()).device
        B, D = int(batch_size), int(self.embed_dim)

        if self.type == "class":
            # Use the reserved null_id
            idx = torch.full((B,), self.null_id, dtype=torch.long, device=device)
            return {'main': _ensure_3d(self.embedder(idx))}
            
        if self.type == "text":
            # Use empty string
            return {'main': _ensure_3d(self.embedder([""] * B))}
            
        if self.type == "image":
            # Use Zero tensor
            T = getattr(self.embedder, 'token_count', 1)
            dtype = next(self.embedder.parameters()).dtype
            z = torch.zeros((B, T, D), dtype=dtype, device=device)
            return {'main': z}
            
        raise RuntimeError("Invalid conditioner type")

    def forward(self, conditioning_data, **kwargs):
        if self.output_key is None:
            return {}

        if self.type == "class":
            labels = conditioning_data
            dev = self.embedder.embedding.weight.device
            
            if isinstance(labels, (list, tuple)):
                labels = torch.tensor(labels, dtype=torch.long, device=dev)
            elif torch.is_tensor(labels):
                labels = labels.to(dev).long()
            
            labels = labels.view(-1) # Flatten


            if self.training and self.p_uncond > 0.0:
                drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.p_uncond
                labels = torch.where(drop_mask, torch.full_like(labels, self.null_id), labels)

            return {'main': _ensure_3d(self.embedder(labels))}

        if self.type == "text":
            if isinstance(conditioning_data, str):
                conditioning_data = [conditioning_data]
            
            if self.training and self.p_uncond > 0.0:
                mask = torch.rand(len(conditioning_data)) < self.p_uncond
                conditioning_data = [("" if m else s) for m, s in zip(mask.tolist(), conditioning_data)]
                
            return {'main': _ensure_3d(self.embedder(conditioning_data))}
        
        if self.type == "image":
            emb = self.embedder(conditioning_data)
            emb = _ensure_3d(emb)

            if self.training and self.p_uncond > 0.0:
                drop = torch.rand(emb.size(0), device=emb.device) < self.p_uncond
                if drop.any():
                    emb = emb.clone()
                    emb[drop] = 0.0
            
            return {'main': emb}

        raise RuntimeError("Invalid conditioner type")