import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional
from PIL import Image

# Requires: pip install git+https://github.com/openai/CLIP.git
import clip

class ClassEmbedder(nn.Module):
    """
    Simple lookup table for class conditioning.
    Output: (B, 1, D)
    """
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, class_labels):
        # class_labels: (B,)
        return self.embedding(class_labels).unsqueeze(1)

class TextEmbedder(nn.Module):
    """
    Frozen CLIP Text Encoder (ViT-B/16).
    Returns full sequence embeddings (B, 77, D) instead of just pooled.
    """
    def __init__(self, version="ViT-B/16", freeze_text_encoder=True, device="cuda"):
        super().__init__()
        # Note: CLIP downloads weights to ~/.cache/clip by default.
        self.clip_model, _ = clip.load(version, device=device)
        
        if freeze_text_encoder:
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
    
    @property
    def embed_dim(self):
        return self.clip_model.token_embedding.embedding_dim
    
    @property
    def token_count(self): 
        return 77

    @torch.no_grad()
    def forward(self, text_prompts: List[str]):
        device = next(self.clip_model.parameters()).device
        tokens = clip.tokenize(text_prompts, context_length=77, truncate=True).to(device)
        
        # Manual forward pass to get sequence features (B, 77, D)
        # Standard clip.encode_text() pools the output, which we don't want.
        x = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        return x

class ImageEmbedder(nn.Module):
    """
    Frozen CLIP Image Encoder (ViT-B/16).
    Can return full spatial tokens (B, 197, D) or just CLS (B, 1, D).
    """
    def __init__(
        self,
        version="ViT-B/16",
        device: Union[str, torch.device] = "cuda",
        return_cls_only: bool = False,
        l2_normalize: bool = False,
    ):
        super().__init__()
        self.clip_model, self._preprocess = clip.load(version, device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.return_cls_only = return_cls_only
        self._l2_normalize = l2_normalize

        # Register normalization constants for Tensor input path
        dev = next(self.clip_model.parameters()).device
        self.register_buffer("_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1))

    @property
    def embed_dim(self) -> int:
        return int(getattr(self.clip_model.visual, "output_dim", 512))

    @property
    def token_count(self):
        return 1 if self.return_cls_only else 197

    def _preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        dev = next(self.clip_model.parameters()).device
        x = x.to(dev, dtype=torch.float32)

        # Normalize range to [0, 1] if needed
        if x.min() < -0.05 or x.max() > 1.05:
            x = (x.clamp(-1, 1) + 1.0) * 0.5
        else:
            x = x.clamp(0.0, 1.0)

        # Resize to CLIP standard resolution (224x224)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        
        # Normalize with CLIP mean/std
        x = (x - self._mean) / self._std
        return x

    def _prep_batch(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        # Handle Tensor input
        if isinstance(images, torch.Tensor):
            if images.dim() == 3: images = images.unsqueeze(0)
            return self._preprocess_tensor(images)

        # Handle List of PIL Images
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            dev = next(self.clip_model.parameters()).device
            return torch.stack([self._preprocess(img.convert("RGB")) for img in images], dim=0).to(dev)

        # Handle Single PIL
        if isinstance(images, Image.Image):
            dev = next(self.clip_model.parameters()).device
            return self._preprocess(images.convert("RGB")).unsqueeze(0).to(dev)

        raise TypeError(f"Unsupported input type: {type(images)}")

    @torch.no_grad()
    def _encode_tokens(self, batch_224: torch.Tensor) -> torch.Tensor:
        """
        Manual ViT forward pass to extract all patch tokens + CLS.
        """
        visual = self.clip_model.visual
        B = batch_224.shape[0]
        
        # Verify this is a ViT model
        if not hasattr(visual, "conv1") or not hasattr(visual, "class_embedding"):
             raise RuntimeError("ImageEmbedder requires a ViT-based CLIP model (e.g., ViT-B/16).")

        # Stem
        x = batch_224.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)  # (B, width, grid, grid)
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1) # (B, grid*grid, width)

        # Add CLS token
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B, 1, cls.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) 
        x = x + visual.positional_embedding.to(x.dtype)

        # Transformer
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x)

        # Projection
        if hasattr(visual, "proj"):
            x = x @ visual.proj

        return x

    @torch.no_grad()
    def forward(self, images) -> torch.Tensor:
        batch = self._prep_batch(images)
        tokens = self._encode_tokens(batch)
        
        if self.return_cls_only:
            tokens = tokens[:, :1, :] # (B, 1, D)
            
        if self._l2_normalize:
            tokens = F.normalize(tokens, p=2, dim=-1)
            
        return tokens.to(next(self.clip_model.parameters()).dtype)