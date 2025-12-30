from __future__ import annotations
import torch
import torch.nn as nn

try:
    from pointnet2_ops import pointnet2_utils
    def fps(xyz: torch.Tensor, number: int) -> torch.Tensor:
        # xyz: (B,N,3) -> (B,number,3)
        idx = pointnet2_utils.furthest_point_sample(xyz, number)  # (B,number)
        sel = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
        return sel
    _HAS_PN2 = True
except Exception:
    _HAS_PN2 = False
    @torch.no_grad()
    def fps(xyz: torch.Tensor, number: int) -> torch.Tensor:
        B, N, _ = xyz.shape
        device = xyz.device
        centroids = torch.zeros(B, number, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), device=device)
        batch_indices = torch.arange(B, device=device)
        for i in range(number):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return xyz[batch_indices[:, None], centroids]  # (B,number,3)

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist

def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    dist = square_distance(new_xyz, xyz)
    _, idx = torch.topk(dist, nsample, dim=-1, largest=False, sorted=False)
    return idx

class PatchDropout(nn.Module):
    def __init__(self, prob: float, exclude_first_token: bool = True):
        super().__init__()
        assert 0.0 <= prob < 1.0
        self.prob = float(prob)
        self.exclude_first_token = bool(exclude_first_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = x[:, :1]
        B, T, C = x.shape
        keep = max(1, int(T * (1 - self.prob)))
        idx = torch.randn(B, T, device=x.device).topk(keep, dim=-1).indices
        x = x[torch.arange(B, device=x.device)[:, None], idx]
        if self.exclude_first_token:
            x = torch.cat([cls_tokens, x], dim=1)
        return x

class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)

    def forward(self, xyz: torch.Tensor, color: torch.Tensor):
        """
        xyz: (B,N,3), color: (B,N,3)
        returns: neighborhood(B,G,M,3), centers(B,G,3), features(B,G,M,6)
        """
        B, N, _ = xyz.shape
        # centers by FPS
        centers = fps(xyz, self.num_group)  # (B,G,3)
        # KNN indices
        idx = knn_point(self.group_size, xyz, centers)  # (B,G,M)
        assert idx.size(1) == self.num_group and idx.size(2) == self.group_size
        base = torch.arange(B, device=xyz.device).view(-1, 1, 1) * N
        idx_flat = (idx + base).reshape(-1)
        neigh = xyz.reshape(B * N, -1)[idx_flat, :].reshape(B, self.num_group, self.group_size, 3).contiguous()
        neigh_c = color.reshape(B * N, -1)[idx_flat, :].reshape(B, self.num_group, self.group_size, 3).contiguous()
        neigh = neigh - centers.unsqueeze(2)
        feats = torch.cat([neigh, neigh_c], dim=-1)  # (B,G,M,6)
        return neigh, centers, feats

class Encoder(nn.Module):
    def __init__(self, encoder_channel: int):
        super().__init__()
        C = int(encoder_channel)
        self.first = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, C, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        # point_groups: (B,G,M,6) -> (B,G,C)
        B, G, M, _ = point_groups.shape
        x = point_groups.reshape(B * G, M, 6).transpose(1, 2)
        x = self.first(x)                                   
        g = torch.max(x, dim=2, keepdim=True)[0]          
        x = torch.cat([g.expand(-1, -1, M), x], dim=1)      
        x = self.second(x)                                 
        g = torch.max(x, dim=2, keepdim=False)[0]         
        return g.reshape(B, G, -1)

class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer: nn.Module, args):
        super().__init__()
        self.trans_dim = int(args.pc_feat_dim)
        self.embed_dim = int(args.embed_dim)
        self.group_size = int(args.group_size)
        self.num_group = int(args.num_group)
        self.encoder_dim = int(args.pc_encoder_dim)
        self.patch_dropout = PatchDropout(getattr(args, "patch_dropout", 0.0)) if getattr(args, "patch_dropout", 0.0) > 0 else nn.Identity()

        self.group_divider = Group(self.num_group, self.group_size)
        self.encoder = Encoder(self.encoder_dim)
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos   = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))

        self.visual = point_transformer 

    def forward(self, pts: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        _, centers, feats = self.group_divider(pts, colors)          
        tokens = self.encoder(feats)                                 
        tokens = self.encoder2trans(tokens)                          
        B, G, T = tokens.shape
        cls = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos = self.pos_embed(centers)                                  
        x = torch.cat([cls, tokens], dim=1)                             
        pos = torch.cat([cls_pos, pos], dim=1)                           
        x = x + pos
        x = self.patch_dropout(x)

        x = getattr(self.visual, "pos_drop", nn.Identity())(x)
        for blk in self.visual.blocks:
            x = blk(x)
        x = self.visual.norm(x[:, 0, :])
        if getattr(self.visual, "fc_norm", None) is not None:
            x = self.visual.fc_norm(x)
        x = self.trans2embed(x)                                        
        return x
