from __future__ import annotations
import numpy as np
import torch
from torch import nn
import timm

from .point_encoder import PointcloudEncoder

class Uni3D(nn.Module):
    def __init__(self, point_encoder: nn.Module):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder
        # expose embed_dim for adapter
        self.embed_dim = getattr(point_encoder, "embed_dim", 1024)

    @torch.inference_mode()
    def encode_pc(self, pc: torch.Tensor) -> torch.Tensor:
        xyz = pc[:, :, :3].contiguous()
        rgb = pc[:, :, 3:].contiguous()
        return self.point_encoder(xyz, rgb)

def create_uni3d(args) -> Uni3D:
    # timm backbone for point tokens
    pt = timm.create_model(
        args.pc_model,
        checkpoint_path=getattr(args, "pretrained_pc", ""),
        drop_path_rate=getattr(args, "drop_path_rate", 0.0),
        pretrained=False,   # using explicit checkpoint_path if any
    )

    point_encoder = PointcloudEncoder(point_transformer=pt, args=args)
    model = Uni3D(point_encoder=point_encoder)
    return model
