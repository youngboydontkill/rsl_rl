import torch
import torch.nn as nn


def build_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    activation=nn.ReLU,
    layer_norm=False,
):
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        if layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(activation())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def pointwise_mlp(mlp, x):
    """Apply an MLP on the last dimension of x, preserving leading dims."""
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    x = mlp(x)
    return x.reshape(*orig_shape[:-1], -1)


class PropsEncoder(nn.Module):
    def __init__(self, d_proprio: int, prop_history: int = 1,proprio_hiddens:list[int]=[256,128],proprio_embed_dim: int = 64):
        super().__init__()
        input_dim = d_proprio*prop_history
        self.proprio_encoder = build_mlp(
            input_dim,
            list(proprio_hiddens) if proprio_hiddens is not None else [],
            proprio_embed_dim,
        )

    def forward(self, proprioception):
        """
        :param proprioception: (B, d) or (B, H, d)
        :return: (B, proprio_embed_dim)
        """
        if proprioception.dim() == 3:
            B, H, d = proprioception.shape
            proprioception = proprioception.reshape(B , H*d)
            proprio_embed = self.proprio_encoder(proprioception)
            return proprio_embed
        else:
            return self.proprio_encoder(proprioception)


class AME2Encoder(nn.Module):
    """
    AME-2 Encoder (attention-based map encoder):
    - CNN for local map features
    - MLP for positional embedding
    - MLP fusion for pointwise local features
    - MLP + MaxPool for global features
    - MLP to build attention query from global + proprio
    - MHA to get weighted local features
    - Concatenate global + weighted local as map embedding
    """

    def __init__(
        self,
        map_channels: int = 3,
        proprio_embed_dim: int = 64,
        attn_dim: int = 96,
        num_heads: int = 8,
        local_cnn_channels=(16, 48),
        pos_embed_dim: int = 16,
        local_mlp_hidden=(96,),
        global_mlp_hidden=(64,),
        global_dim: int = 64,
        query_mlp_hidden=(96,),
        use_batch_norm: bool = True,
        pos_from_map: bool = True,
        remove_xy_channels: bool = True,
    ):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")

        self.map_channels = map_channels
        self.proprio_embed_dim = proprio_embed_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.pos_embed_dim = pos_embed_dim
        self.global_dim = global_dim
        self.pos_from_map = pos_from_map
        self.remove_xy_channels = remove_xy_channels

        # Local CNN (remove x,y channels by default)
        cnn_layers = []
        if remove_xy_channels and map_channels > 2:
            in_ch = map_channels - 2
        else:
            in_ch = map_channels
        for out_ch in local_cnn_channels:
            cnn_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            )
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_ch))
            cnn_layers.append(nn.ReLU())
            in_ch = out_ch
        self.local_cnn = nn.Sequential(*cnn_layers)
        self.local_cnn_out = in_ch

        # Positional embedding MLP
        self.pos_mlp = build_mlp(2, [pos_embed_dim], pos_embed_dim)

        # Pointwise local features
        local_in_dim = self.local_cnn_out + pos_embed_dim
        self.local_mlp = build_mlp(
            local_in_dim,
            list(local_mlp_hidden),
            attn_dim,
        )

        # Global features
        self.global_mlp = build_mlp(
            attn_dim,
            list(global_mlp_hidden),
            global_dim,
        )

        # Query MLP (global + proprio)
        self.query_mlp = build_mlp(
            global_dim + proprio_embed_dim,
            list(query_mlp_hidden),
            attn_dim,
        )

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def _get_positional_features(self, map_obs):
        # map_obs: (B, L, W, C)
        if self.pos_from_map and map_obs.shape[-1] >= 2:
            pos = map_obs[..., :2]
        else:
            B, L, W, _ = map_obs.shape
            y = torch.linspace(-1.0, 1.0, L, device=map_obs.device)
            x = torch.linspace(-1.0, 1.0, W, device=map_obs.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            pos = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos

    def _get_cnn_map(self, map_obs):
        if self.remove_xy_channels and map_obs.shape[-1] > 2:
            return map_obs[..., 2:]
        return map_obs

    def forward(self, map_obs, proprio_embed):
        """
        :param map_obs: (B, L, W, C)
        :param proprio_embed: (B, proprio_embed_dim)
        :return map_embedding: (B, global_dim + attn_dim)
        :return attn_weights: (B, L, W)
        """
        map_obs = torch.where(
            torch.isnan(map_obs),
            torch.zeros_like(map_obs),
            map_obs,
        )

        B, L, W, C = map_obs.shape

        # Local CNN features (x,y removed if configured)
        cnn_map = self._get_cnn_map(map_obs)
        x = cnn_map.permute(0, 3, 1, 2)
        cnn_feat = self.local_cnn(x).permute(0, 2, 3, 1)  # (B, L, W, Cc)

        # Positional embedding
        pos = self._get_positional_features(map_obs)
        pos_emb = pointwise_mlp(self.pos_mlp, pos)  # (B, L, W, pos_dim)

        # Pointwise local features
        local_in = torch.cat([cnn_feat, pos_emb], dim=-1)
        pointwise_local = pointwise_mlp(
            self.local_mlp,
            local_in,
        )  # (B, L, W, attn_dim)
        pointwise_local = pointwise_local.reshape(B, L * W, self.attn_dim)

        # Global features (pointwise MLP + max pool)
        global_points = pointwise_mlp(
            self.global_mlp,
            pointwise_local,
        )  # (B, L*W, global_dim)
        global_feat = global_points.max(dim=1).values  # (B, global_dim)

        # Query from global + proprio
        query = self.query_mlp(
            torch.cat([global_feat, proprio_embed], dim=-1)
        ).unsqueeze(1)

        # Attention over local features
        attn_out, attn_weights = self.mha(
            query=query,
            key=pointwise_local,
            value=pointwise_local,
        )
        weighted_local = attn_out.squeeze(1)  # (B, attn_dim)

        # Map embedding
        map_embedding = torch.cat([global_feat, weighted_local], dim=-1)
        attn_weights = attn_weights.reshape(B, L, W)
        return map_embedding, attn_weights


class AME2MapEncoder(nn.Module):
    """
    Wrapper that encodes proprioception and calls AME2Encoder.
    Supports history dimension H: map_obs (B, H, L, W, C), proprio (B, H, d).
    """

    def __init__(
        self,
        map_channels: int = 3,
        proprio_embed_dim: int = 64,
        attn_dim: int = 96,
        num_heads: int = 8,
        local_cnn_channels=(16, 48),
        pos_embed_dim: int = 16,
        local_mlp_hidden=(96,),
        global_mlp_hidden=(64,),
        global_dim: int = 64,
        query_mlp_hidden=(96,),
        use_batch_norm: bool = True,
        pos_from_map: bool = True,
    ):
        super().__init__()
        self.encoder = AME2Encoder(
            map_channels=map_channels,
            proprio_embed_dim=proprio_embed_dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            local_cnn_channels=local_cnn_channels,
            pos_embed_dim=pos_embed_dim,
            local_mlp_hidden=local_mlp_hidden,
            global_mlp_hidden=global_mlp_hidden,
            global_dim=global_dim,
            query_mlp_hidden=query_mlp_hidden,
            use_batch_norm=use_batch_norm,
            pos_from_map=pos_from_map,
        )

    def forward(self, map_obs, proprio_embed, embedding_only=False):
        """
        :param map_obs: (B, L, W, C) or (B, H, L, W, C) with H fixed to 1
        :param proprio_embed: (B, proprio_embed_dim) or (B, H, proprio_embed_dim)
        :return: (B, global_dim + attn_dim) or (B, map_embed + proprio_embed)
        """
        # H is fixed to 1, return shape [B, ...]
        if map_obs.dim() == 5:
            B, H, L, W, C = map_obs.shape
            map_obs = map_obs.reshape(B * H, L, W, C)
            if proprio_embed.dim() == 3:
                proprio_flat = proprio_embed.reshape(B * H, -1)
            else:
                proprio_flat = proprio_embed
            map_emb, attn = self.encoder(map_obs, proprio_flat)
            attn = attn.reshape(B, H, L, W)
            if H == 1:
                map_emb = map_emb.reshape(B, -1)
                attn = attn.reshape(B, L, W)
                if embedding_only:
                    return map_emb, attn
                if proprio_embed.dim() == 3:
                    proprio_flat = proprio_embed[:, 0, :]
                return torch.cat([map_emb, proprio_flat], dim=-1), attn
            if embedding_only:
                return map_emb, attn
            return (
                torch.cat(
                    [map_emb, proprio_embed.reshape(B, H, -1)],
                    dim=-1,
                ),
                attn,
            )
        else:
            map_emb, attn = self.encoder(map_obs, proprio_embed)
            if embedding_only:
                return map_emb, attn
            return torch.cat([map_emb, proprio_embed], dim=-1), attn
