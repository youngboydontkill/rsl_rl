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
    """
    编码本体感觉的前馈网络（参考 AME 结构）
    """

    def __init__(
        self,
        d_obs,
        prop_history=1,
        hidden_dims=(256, 128),
        embedding_dim=64,
        layer_norm=False,
    ):
        """
        :param d_obs: 本体感知向量的维度(单次观测)
        :param prop_history: 历史帧数（将被展平）
        :param embedding_dim: 编码维度
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.prop_history = prop_history
        input_dim = d_obs * prop_history
        self.proprio_encoder = build_mlp(
            input_dim,
            list(hidden_dims) if hidden_dims is not None else [],
            embedding_dim,
            layer_norm=layer_norm,
        )

    def forward(self, proprioception):
        """
        :param proprioception: (B, d_obs) 或 (B, H, d_obs)
        :return: proprio_embedding: (B, embedding_dim)
        """
        if proprioception.dim() == 3:
            B, H, d = proprioception.shape
            proprioception = proprioception.reshape(B, H * d)
        return self.proprio_encoder(proprioception)


class PerceptionEncoder(nn.Module):
    """
    编码外感的特征提取器（参考 AME 结构）
    """

    def __init__(
        self,
        input_channels=1,
        local_cnn_channels=(32, 64),
        pos_embed_dim=16,
        local_mlp_hidden=(96,),
        local_dim=96,
        global_mlp_hidden=(64,),
        global_dim=64,
        use_batch_norm=True,
        pos_from_map=False,
        remove_xy_channels=False,
    ):
        """
        :param input_channels: 输入通道数
        :param local_dim: 局部特征维度（供注意力使用）
        :param global_dim: 全局特征维度
        """
        super().__init__()
        self.pos_embed_dim = pos_embed_dim
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.pos_from_map = pos_from_map
        self.remove_xy_channels = remove_xy_channels

        # Local CNN
        cnn_layers = []
        if remove_xy_channels and input_channels > 2:
            in_ch = input_channels - 2
        else:
            in_ch = input_channels
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
            local_dim,
        )

        # Global features
        self.global_mlp = build_mlp(
            local_dim,
            list(global_mlp_hidden),
            global_dim,
        )

    def _get_positional_features(self, obs):
        # obs: (B, H, W, C)
        if self.pos_from_map and obs.shape[-1] >= 2:
            pos = obs[..., :2]
        else:
            B, H, W, _ = obs.shape
            y = torch.linspace(-1.0, 1.0, H, device=obs.device)
            x = torch.linspace(-1.0, 1.0, W, device=obs.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            pos = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos

    def _get_cnn_input(self, obs):
        if self.remove_xy_channels and obs.shape[-1] > 2:
            return obs[..., 2:]
        return obs

    def forward(self, perception):
        """
        :param perception: (B, C, H, W)
        :return: local_feat (B, H*W, local_dim), global_feat (B, global_dim)
        """
        if perception.dim() != 4:
            raise ValueError("perception must be (B, C, H, W)")

        B, C, H, W = perception.shape
        obs = perception.permute(0, 2, 3, 1)  # (B, H, W, C)
        obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)

        cnn_in = self._get_cnn_input(obs)
        x = cnn_in.permute(0, 3, 1, 2)  # (B, C', H, W)
        cnn_feat = self.local_cnn(x).permute(0, 2, 3, 1)  # (B, H, W, Cc)

        pos = self._get_positional_features(obs)
        pos_emb = pointwise_mlp(self.pos_mlp, pos)  # (B, H, W, pos_dim)

        local_in = torch.cat([cnn_feat, pos_emb], dim=-1)
        pointwise_local = pointwise_mlp(
            self.local_mlp,
            local_in,
        )  # (B, H, W, local_dim)
        pointwise_local = pointwise_local.reshape(B, H * W, self.local_dim)

        global_points = pointwise_mlp(
            self.global_mlp,
            pointwise_local,
        )  # (B, H*W, global_dim)
        global_feat = global_points.max(dim=1).values  # (B, global_dim)
        return pointwise_local, global_feat


class Props_perc_fuser(nn.Module):
    """
    将本体感知与外感特征融合（使用 MLP）
    global_dim: 输出的latent
    """

    def __init__(
        self,
        local_dim=96,  
        proprio_embed_dim=64,
        global_dim=64,
        attn_dim=96,
        fusion_mlp_hidden=(96,),
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.global_dim = global_dim
        self.fusion_mlp = build_mlp(
            local_dim + global_dim + proprio_embed_dim,
            list(fusion_mlp_hidden),
            global_dim,
        )

    def forward(self, local_feat, global_feat, proprio_embed):
        """
        :param local_feat: (B, N, cnn_dim) 
        :param global_feat: (B, global_dim)
        :param proprio_embed: (B, proprio_embed_dim)
        :return: fused_embedding (B, embed_dim), attn_weights (None)
        """
        fused = self.fusion_mlp(
            torch.cat([local_feat.mean(dim=1), global_feat, proprio_embed], dim=-1)
        )
        return fused, None


class LatentEncoder(nn.Module):
    """
    Latent encoder: PropsEncoder + PerceptionEncoder + Props_perc_fuser
    """

    def __init__(
        self,
        d_obs,
        prop_history=1,
        proprio_embed_dim=64,
        proprio_hidden=(256, 128),
        input_channels=1,
        local_cnn_channels=(32, 64),
        pos_embed_dim=16,
        local_mlp_hidden=(96,),
        local_dim=96,
        global_mlp_hidden=(64,),
        global_dim=64,
        attn_dim=96,
        fusion_mlp_hidden=(96,),
        num_heads=None,
        query_mlp_hidden=None,
        use_batch_norm=True,
        pos_from_map=False,
        remove_xy_channels=False,
    ):
        super().__init__()
        self.props_encoder = PropsEncoder(
            d_obs=d_obs,
            prop_history=prop_history,
            hidden_dims=proprio_hidden,
            embedding_dim=proprio_embed_dim,
        )
        self.perc_encoder = PerceptionEncoder(
            input_channels=input_channels,
            local_cnn_channels=local_cnn_channels,
            pos_embed_dim=pos_embed_dim,
            local_mlp_hidden=local_mlp_hidden,
            local_dim=local_dim,
            global_mlp_hidden=global_mlp_hidden,
            global_dim=global_dim,
            use_batch_norm=use_batch_norm,
            pos_from_map=pos_from_map,
            remove_xy_channels=remove_xy_channels,
        )
        self.fuser = Props_perc_fuser(
            local_dim=local_dim,
            proprio_embed_dim=proprio_embed_dim,
            global_dim=global_dim,
            attn_dim=attn_dim,
            fusion_mlp_hidden=fusion_mlp_hidden
            if fusion_mlp_hidden is not None
            else (query_mlp_hidden or (96,)),
        )

    def forward(
        self,
        perception,
        proprioception,
        embedding_only=False,
        return_intermediate=False,
    ):
        """
        :param perception: (B, C, H, W)
        :param proprioception: (B, d_obs) 或 (B, H, d_obs)
        :param return_intermediate: 是否返回 props/perception embedding
        :return: latent (B, global_dim + attn_dim) 或 concat(latent, proprio)
                 若 return_intermediate=True，额外返回 props_embed, perception_embed, local_feat
        """
        proprio_embed = self.props_encoder(proprioception)
        local_feat, global_feat = self.perc_encoder(perception)
        latent, _ = self.fuser(local_feat, global_feat, proprio_embed)
        if embedding_only:
            out = latent
        else:
            out = torch.cat([latent, proprio_embed], dim=-1)
        if return_intermediate:
            return (out, proprio_embed, global_feat, local_feat)
        return out
