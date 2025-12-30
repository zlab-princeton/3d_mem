import torch
import torch.nn as nn
from einops import rearrange, repeat
import math


from .utils import PreNorm, CrossAttention, FeedForward, subsample, PointEmbed, VanillaVolumeDecoder, MCSurfaceExtractor, DMCSurfaceExtractor
from .bottleneck import KLBottleneck, NormalizedBottleneck

class VecSetAutoEncoder(nn.Module):
    """ The core AutoEncoder model, largely unchanged from V2 but now configured by the factory. """
    def __init__(
        self,
        *,
        depth=24,
        dim=768,
        num_inputs=8192,
        num_latents=1024,
        dim_head=64,
        query_type='point',
        bottleneck=None,
    ):
        super().__init__()
        self.depth = depth
        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.query_type = query_type
        
        if query_type == 'point':
            pass
        elif query_type == 'learnable':
            self.latents = nn.Embedding(num_latents, dim)
        else:
            raise NotImplementedError(f'Query type {query_type} not implemented')

        self.point_embed = PointEmbed(dim=dim)
        
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, CrossAttention(dim, context_dim=dim, heads=dim // dim_head, dim_head=dim_head, dropout=0.)),
            PreNorm(dim, FeedForward(dim))
        ])
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, heads=dim // dim_head, dim_head=dim_head, dropout=0.)),
                PreNorm(dim, FeedForward(dim))
            ]) for _ in range(depth)
        ])
        
        self.decoder_cross_attn = PreNorm(dim, CrossAttention(dim, context_dim=dim, heads=dim // dim_head, dim_head=dim_head, dropout=0.))
        # self.to_outputs = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.to_outputs = nn.Sequential(nn.LayerNorm(dim, eps=1e-3), nn.Linear(dim, 1))
        
        # The bottleneck module is now passed in directly
        self.bottleneck = bottleneck

    def encode(self, pc, sample_posterior=True):
        B, N, _ = pc.shape
        assert N == self.num_inputs, f"Input point cloud size ({N}) does not match expected size ({self.num_inputs})"
        
        if self.query_type == 'point':
            # Use Farthest Point Sampling for queries
            sampled_pc = subsample(pc, N, self.num_latents)
            x = self.point_embed(sampled_pc)
        elif self.query_type == 'learnable':
            # Use learned embeddings for queries
            x = repeat(self.latents.weight, 'n d -> b n d', b=B)
            
        pc_embeddings = self.point_embed(pc)
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context=pc_embeddings) + x
        x = cross_ff(x) + x
        
        # Pass through the pre-bottleneck stage
        bottleneck_output = self.bottleneck.pre(x, sample_posterior=sample_posterior)
        return bottleneck_output

    def learn_and_decode(self, bottleneck_output, queries):        
        # Process the latent vector through the bottleneck and transformer layers.
        x = self.bottleneck.post(bottleneck_output['x'])
        
        if self.query_type == 'learnable':
            x = x + repeat(self.latents.weight, 'n d -> b n d', b=x.shape[0])

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
            
        # Decode the queries in chunks to save memory.
        if queries.shape[1] > 65536: # Use a reasonable chunk size
            chunk_size = 65536
            outputs = []
            for i in range(math.ceil(queries.shape[1] / chunk_size)):
                chunk_queries = queries[:, i*chunk_size:(i+1)*chunk_size, :]
                queries_embeddings = self.point_embed(chunk_queries)
                latents_chunk = self.decoder_cross_attn(queries_embeddings, context=x)
                outputs.append(self.to_outputs(latents_chunk))
            return torch.cat(outputs, dim=1)
        else:
            # If the number of queries is small, process them all at once.
            queries_embeddings = self.point_embed(queries)
            latents = self.decoder_cross_attn(queries_embeddings, context=x)
            return self.to_outputs(latents)

    def forward(self, pc, queries):
        bottleneck_output = self.encode(pc)
        
        # print(f"[LOG] Bottleneck output: {bottleneck_output['x']}")
        
        o = self.learn_and_decode(bottleneck_output, queries)
        
        return {'logits': o.squeeze(-1), **bottleneck_output}



class CustomShapeVAE(nn.Module):
    def __init__(
        self,
        *,
        model_depth: int,
        model_dim: int,
        surface_size: int,
        num_latents: int,
        latent_dim: int,
        query_type: str,
        bottleneck: str,
        surface_extractor: str = 'mc',  # 'mc' or 'dmc'
    ):
        super().__init__()
        
        class Args:
            pass
        
        args = Args()
        args.model_depth = model_depth
        args.model_dim = model_dim
        args.surface_size = surface_size
        args.num_latents = num_latents
        args.latent_dim = latent_dim
        args.query_type = query_type
        args.bottleneck = bottleneck
        
        
        # Build the core autoencoder using your existing factory
        self.autoencoder = create_autoencoder(args)
        
        # Instantiate the generic decoder and extractor tools from utils.py
        self.volume_decoder = VanillaVolumeDecoder()
        if surface_extractor.lower() == 'dmc':
            self.surface_extractor = DMCSurfaceExtractor()
        else:
            self.surface_extractor = MCSurfaceExtractor()

    def encode(self, surface, sample_posterior=True):
        pc = surface[..., :3]
        
        bottleneck_output = self.autoencoder.encode(pc, sample_posterior=sample_posterior)
        
        # The output of your bottleneck's 'pre' method is the latent
        latents = bottleneck_output['x'] 
        return latents

    def decode(self, latents, queries):
        bottleneck_output = {'x': latents}
        
        out = self.autoencoder.learn_and_decode(bottleneck_output, queries)
        return out.squeeze(-1)

    def latents2mesh(self, latents, **kwargs):
        # The 'geo_decoder' that VanillaVolumeDecoder needs is our implicit 'decode' method
        grid_logits = self.volume_decoder(latents, self.decode, **kwargs)
        
        outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs

    def forward(self, surface, queries, **kwargs):
        """
        A standard forward pass for training the VAE itself.
        """
        return self.autoencoder(surface, queries)


def create_autoencoder(args):
    bottleneck_map = {
        'kl': KLBottleneck,
        'normalized': NormalizedBottleneck,
    }
    bottleneck_cls = bottleneck_map.get(args.bottleneck)
    if bottleneck_cls is None:
        raise ValueError(f"Unknown bottleneck type: {args.bottleneck}")

    bottleneck_args = {
        'dim': args.model_dim,
        'latent_dim': args.latent_dim,
    }
    
    bottleneck_instance = bottleneck_cls(**bottleneck_args)

    model = VecSetAutoEncoder(
        depth=args.model_depth,
        dim=args.model_dim,
        num_inputs=args.surface_size,
        num_latents=args.num_latents,
        query_type=args.query_type,
        bottleneck=bottleneck_instance
    )
    
    return model