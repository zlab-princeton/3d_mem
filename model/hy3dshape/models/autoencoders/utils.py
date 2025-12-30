import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm
from skimage import measure
from typing import Callable, Union, Tuple, List

from flash_attn import flash_attn_kvpacked_func
from torch_cluster import fps


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def zero_module(module):
    """ Zero out the parameters of a module and return it. """
    for p in module.parameters():
        p.detach().zero_()
    return module

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.endpoint = endpoint
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / max_positions) ** freqs
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        x_emb = x[:, None] * self.freqs[None, :]
        return torch.cat([x_emb.cos(), x_emb.sin()], dim=1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # self.norm = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-3)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)
    
class DiffusionGEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class DiffusionFeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else DiffusionGEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.dropout = dropout

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, window_size=-1):

        input_dtype = x.dtype
        context = default(context, x)
        
        q = self.to_q(x)
        kv = self.to_kv(context)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        kv = rearrange(kv, 'b n (p h d) -> b n p h d', h=self.heads, p=2)
        
        # Use bfloat16 for speed and pass dropout directly to the kernel
        out = flash_attn_kvpacked_func(
            q.to(torch.bfloat16), 
            kv.to(torch.bfloat16), 
            dropout_p=self.dropout if self.training else 0.0
        )
        
        out = out.to(input_dtype)
        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum('bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))
        return embed

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        return self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        if other is None:
            return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2])
        else:
            return 0.5 * torch.mean(
                torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3])

def subsample(pc, N, M):
    B, N0, D = pc.shape
    assert N == N0
    flattened = pc.view(B * N, D)
    batch = torch.arange(B, device=pc.device).repeat_interleave(N)
    ratio = float(M) / N
    idx = fps(flattened.to(torch.float32), batch, ratio=ratio)
    sampled_pc = flattened[idx].view(B, M, D)
    return sampled_pc


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        # self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-3)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        return self.layernorm(x) * (1 + scale) + shift
    
    
def center_vertices(vertices):
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center

def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f
        
        
class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = 256,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        def _ddp_info():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    return torch.distributed.get_rank(), torch.distributed.get_world_size()
                except Exception:
                    pass
            return 0, 1
        rank, world = _ddp_info()

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        total_pts = int(xyz_samples.shape[0])
        n_chunks = (total_pts + num_chunks - 1) // num_chunks
        print(f"[rank{rank}/{world}] VolumeDecoding: batch={batch_size}, "
              f"points={total_pts}, octree_res={octree_resolution}, "
              f"chunks={n_chunks}, chunk_size={num_chunks}")

        # 2. latents to 3d volume
        batch_logits = []
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc=f"[rank{rank}] Volume Decoding",
                          disable=(not enable_pbar) or (rank != 0 and world > 1)):
            chunk_queries = xyz_samples[start: start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits
    
    
class SurfaceExtractor:
    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        """
        Compute grid size, bounding box minimum coordinates, and bounding box size based on input 
        bounds and resolution.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single 
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].
            octree_resolution (int): Resolution of the octree grid.

        Returns:
            grid_size (List[int]): Grid size along each axis (x, y, z), each equal to octree_resolution + 1.
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        """
        Process a batch of grid logits to extract surface meshes.

        Args:
            grid_logits (torch.Tensor): Batch of grid logits with shape (batch_size, ...).
            **kwargs: Additional keyword arguments passed to the `run` method.

        Returns:
            List[Optional[Latent2MeshOutput]]: List of mesh outputs for each grid in the batch.
                If extraction fails for a grid, None is appended at that position.
        """
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        # grid_logit: torch.Tensor of shape (Nx, Ny, Nz)
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        grid = grid_logit.detach().to(torch.float32).cpu().numpy()

        # spacing maps array indices to metric units per axis (axis 0/1/2)
        spacing = (bbox_size / (np.array(grid_size) - 1)).astype(np.float32)  # (3,)

        # skimage returns verts already scaled by `spacing`, but not translated
        # No need to pass method="lewiner" (deprecated); default is fine.
        vertices, faces, normals, _ = measure.marching_cubes(
            grid, level=float(mc_level), spacing=tuple(spacing)
        )

        vertices = vertices + bbox_min  # translate into the bbox frame
        return vertices.astype(np.float32), np.ascontiguousarray(faces)
    
    
class DMCSurfaceExtractor(SurfaceExtractor):
    def __init__(self):
        super().__init__()
        self._dmc = None

    def _lazy_init(self, device):
        if self._dmc is None:
            try:
                from diso import DiffDMC
            except Exception as e:
                raise ImportError("Please `pip install diso` to use DMCSurfaceExtractor") from e
            self._dmc = DiffDMC(dtype=torch.float32).to(device)

    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Match MCSurfaceExtractor geometry frame:
          - iso-surface at (grid - mc_level) = 0
          - scale by voxel spacing
          - translate by bbox_min
        """
        device = grid_logit.device
        self._lazy_init(device)

        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        spacing = torch.from_numpy(bbox_size / (np.array(grid_size) - 1)).to(device=device, dtype=torch.float32)

        # Build SDF in the same units as grid values; if your grid is "logit-ish",
        # using (grid - mc_level) is usually the right 0-level set
        sdf = (grid_logit.detach().to(torch.float32) - float(mc_level)).contiguous()

        # DMC returns vertices in voxel index coordinates when normalize=False (0..Nx, etc.)
        # If your DiffDMC only supports normalize=True, set normalize=True and adapt scale accordingly.
        verts_vox, faces = self._dmc(sdf, deform=None, return_quads=False, normalize=False)

        # Map from voxel coords to world coords: multiply per-axis spacing and add bbox_min
        # verts_vox is (V, 3) in (z,y,x) or (x,y,z) depending on impl; most libs are (x,y,z).
        verts = verts_vox * spacing  + torch.from_numpy(bbox_min).to(device, dtype=torch.float32)

        vertices = verts.detach().cpu().numpy().astype(np.float32)
        faces = faces.detach().cpu().numpy()[:, ::-1]  # flip winding to match trimesh/skimage
        return vertices, np.ascontiguousarray(faces)