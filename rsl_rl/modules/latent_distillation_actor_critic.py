from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict 
from rsl_rl.networks import MLP, EmpiricalNormalization, LatentEncoder


class LatentDistillationActorCritic(nn.Module):
    is_recurrent = False
    LOAD_POLICY_WEIGHTS = 1
    LOAD_CRITIC_WEIGHTS = 2
    LOAD_ENCODER_WEIGHTS = 4 
    LOAD_NORMALIZER_WEIGHTS = 8
    LOAD_CRITIC_ESTIMATOR_WEIGHTS = 16
    LOAD_PROPS_ENCODER_WEIGHTS = 32

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        attn_embedding_dim=64,
        attn_dim: int | None = None,
        embedding_dim: int | None = None,
        map_channels: int | None = None,
        num_heads: int = 8,
        local_cnn_channels: tuple[int, ...] = (16, 48),
        pos_embed_dim: int = 16,
        local_mlp_hidden: tuple[int, ...] = (96,),
        global_mlp_hidden: tuple[int, ...] = (64,),
        global_dim: int = 64,
        fusion_mlp_hidden: tuple[int, ...] = (96,),
        query_mlp_hidden: tuple[int, ...] = (96,),
        use_batch_norm: bool = True,
        pos_from_map: bool = True,
        remove_xy_channels: bool = True,
        props_embed_dim: int = 64,
        actor_props_encoder_hidden:list[int]=[256,128],
        critic_props_encoder_hidden:list[int] | None = None,
        use2Encoder:bool=False,
        load_mask:int=LOAD_POLICY_WEIGHTS|LOAD_CRITIC_WEIGHTS|LOAD_ENCODER_WEIGHTS|LOAD_NORMALIZER_WEIGHTS|LOAD_CRITIC_ESTIMATOR_WEIGHTS,
        output_attention:bool=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "EncActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.verify = True  # 用于验证某些中间变量的shape
        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0  # obervation dimensions in 1 stamp for the actor
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) > 2, "The EncActorCritic module only supports obs shape [B,H,d,...]. "
            "for IsaacLab, you need to make sure that flatten_history_dim is False."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0 # obervation dimensions in 1 stamp for the critic 
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) > 2, "The EncActorCritic module only supports obs shape [B,H,d,...]. "
            "for IsaacLab, you need to make sure that flatten_history_dim is False."
            num_critic_obs += obs[obs_group].shape[-1]
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.use2Encoder = use2Encoder

        actor_horizon_candidates = []
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) > 2:
                actor_horizon_candidates.append(obs[obs_group].shape[1])
        self.actor_horizon = max(actor_horizon_candidates) if actor_horizon_candidates else 1

        critic_horizon_candidates = []
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) > 2:
                critic_horizon_candidates.append(obs[obs_group].shape[1])
        self.critic_single_frame = any(h == 1 for h in critic_horizon_candidates)
        if self.critic_single_frame:
            self.critic_horizon = 1
        else:
            self.critic_horizon = max(critic_horizon_candidates) if critic_horizon_candidates else 1

        # Encoder :
        # num_perception_obs = 0
        scan_height_shape = []
        for obs_group in obs_groups["perception"]:
            # TODO : 这里需要修改为支持多obs的输入
            # num_perception_obs += obs[obs_group].shape[-1]
            if (obs_group == "perception"):
                scan_height_shape = obs[obs_group].shape # 
        if embedding_dim is not None:
            attn_dim = embedding_dim
        if attn_dim is None:
            attn_dim = attn_embedding_dim
        self.attn_embedding_dim = attn_dim

        self.props_embed_dim = props_embed_dim
        if map_channels is None:
            map_channels = scan_height_shape[-1]

        self.map_channels = map_channels

        if (not self.use2Encoder) and (
            num_actor_obs != num_critic_obs or self.actor_horizon != self.critic_horizon
        ):
            self.use2Encoder = True

        actor_encoder_cfg = dict(
            d_obs=num_actor_obs,
            prop_history=self.actor_horizon,
            proprio_embed_dim=self.props_embed_dim,
            input_channels=map_channels,
            local_cnn_channels=local_cnn_channels,
            pos_embed_dim=pos_embed_dim,
            local_mlp_hidden=local_mlp_hidden,
            local_dim=self.attn_embedding_dim,
            global_mlp_hidden=global_mlp_hidden,
            global_dim=global_dim,
            attn_dim=self.attn_embedding_dim,
            fusion_mlp_hidden=fusion_mlp_hidden
            if fusion_mlp_hidden is not None
            else query_mlp_hidden,
            use_batch_norm=use_batch_norm,
            pos_from_map=pos_from_map,
            remove_xy_channels=remove_xy_channels,
        )
        critic_encoder_cfg = dict(
            d_obs=num_critic_obs,
            prop_history=self.critic_horizon,
            proprio_embed_dim=self.props_embed_dim,
            input_channels=map_channels,
            local_cnn_channels=local_cnn_channels,
            pos_embed_dim=pos_embed_dim,
            local_mlp_hidden=local_mlp_hidden,
            local_dim=self.attn_embedding_dim,
            global_mlp_hidden=global_mlp_hidden,
            global_dim=global_dim,
            attn_dim=self.attn_embedding_dim,
            fusion_mlp_hidden=fusion_mlp_hidden
            if fusion_mlp_hidden is not None
            else query_mlp_hidden,
            use_batch_norm=use_batch_norm,
            pos_from_map=pos_from_map,
            remove_xy_channels=remove_xy_channels,
        )

        # TODO 共用一个 or 分离？先用一个试试
        # 在输入encoder时用无噪的actor_obs，cath的时候再把critic的拼进去
        if self.use2Encoder:
            self.actor_encoder = LatentEncoder(**actor_encoder_cfg)
            self.critic_encoder = LatentEncoder(**critic_encoder_cfg)
            print(f"Actor Latent Encoder : {self.actor_encoder}")
            print(f"Critic Latent Encoder : {self.critic_encoder}")
        else:
            self.encoder = LatentEncoder(**actor_encoder_cfg)
            print(f"Latent Encoder : {self.encoder}")

        self.horizon = self.actor_horizon
        self.high_dim_obs_shape = scan_height_shape # [B,H,L,W,C]
        self.load_mask = load_mask  # 加载参数的mask
        self.output_attention = output_attention  # 是否输出attention 
        # 使用prop encoder之后，嵌入维度固定
        map_embed_dim = global_dim + self.attn_embedding_dim
        embedding_actor_dim = self.props_embed_dim + map_embed_dim  # [B, map_embed + prop_embed]
        embedding_critic_dim = self.props_embed_dim + map_embed_dim  # [B, map_embed + prop_embed]
        self.embedding_actor_dim = embedding_actor_dim
        self.embedding_critic_dim = embedding_critic_dim 
        # 这里需要构造一个从critic到actor obs的mask, 但是当前仍然只支持1d的输入
        # critic_to_actor_mask = torch.zeros((num_critic_obs,), dtype=torch.bool)
        # tensor_idx = 0
        # for i, obs_group in enumerate(obs_groups["critic"]):
        #     if obs_group in obs_groups["policy"]:
        #         critic_to_actor_mask[tensor_idx: tensor_idx + obs[obs_group].shape[-1]] = True
        #     tensor_idx += obs[obs_group].shape[-1]
        # self.critic_to_actor_mask = critic_to_actor_mask
        # convert to [B,H,d] style (if obs shape = (B,H*d)) 
        # if (obs_style=='lab'):
        #     self.critic_to_actor_mask = self._lab_to_gym(critic_to_actor_mask,self.horizon,keep_dim=False)
        # else:
        #     self.critic_to_actor_mask = critic_to_actor_mask.reshape(self.horizon,-1)
        

        # actor
        self.actor = MLP(embedding_actor_dim, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization((self.actor_horizon,num_actor_obs))  # 这里是支持输入[B,H,d]的(self.actor_horizon,num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(embedding_critic_dim, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization((self.critic_horizon,num_critic_obs))  # 是不是(self.critic_horizon,num_critic_obs)会更好?
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, proprio_obs: torch.Tensor, perception_obs: torch.Tensor):
        """
        :param proprio_obs: [B, H, d_obs]
        :param perception_obs: [B, H, L, W, C] or [B, L, W, C] or [B, C, H, W]
        """
        perception_cf = self._prepare_perception(perception_obs)
        # compute embedding
        if self.use2Encoder:
            embedding, _ = self.actor_encoder(
                perception_cf, proprio_obs, embedding_only=False
            )
        else:
            embedding, _ = self.encoder(
                perception_cf, proprio_obs, embedding_only=False
            )
        if self.verify:
            print(f"embedding shape: {embedding.shape}")
            self.verify = False
        embedding_vec = embedding
        # compute mean
        mean = self.actor(embedding_vec)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs:TensorDict, **kwargs):
        low_dim_obs,high_dim_obs = self.get_actor_obs(obs)
        low_dim_obs = self.actor_obs_normalizer(low_dim_obs)  # [B,H,d]
        self.update_distribution(low_dim_obs, high_dim_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        low_dim_obs,high_dim_obs = self.get_actor_obs(obs)  # [B,H,d]
        low_dim_obs = self.actor_obs_normalizer(low_dim_obs) # [B,H,d]
        perception_cf = self._prepare_perception(high_dim_obs)
        # compute embedding
        if self.use2Encoder:
            embedding, attention = self.actor_encoder(
                perception_cf, low_dim_obs, embedding_only=False
            )
        else:
            embedding, attention = self.encoder(
                perception_cf, low_dim_obs, embedding_only=False
            )
        # compute mean
        action = self.actor(embedding)
        if (self.output_attention):
            return action,attention
        else:
            return action

    def evaluate(self, obs, **kwargs):
        low_dim_obs,high_dim_obs = self.get_critic_obs(obs)  # [B,H,d]
        low_dim_obs = self.critic_obs_normalizer(low_dim_obs)
        perception_cf = self._prepare_perception(high_dim_obs)
        if self.use2Encoder:
            embedding, _ = self.critic_encoder(
                perception_cf, low_dim_obs, embedding_only=False
            )
        else:
            embedding, _ = self.encoder(
                perception_cf, low_dim_obs, embedding_only=False
            )
        values = self.critic(embedding)
        return values

    def get_student_distill_params(self):
        """Return parameters to optimize for student-side distillation."""
        if self.use2Encoder:
            return self.actor_encoder.parameters()
        return self.encoder.parameters()

    def freeze_teacher(self):
        """Freeze teacher-side parameters after teacher training."""
        if self.use2Encoder:
            for param in self.critic_encoder.parameters():
                param.requires_grad_(False)
        for param in self.critic.parameters():
            param.requires_grad_(False)

    def get_distill_embeddings(self, obs: TensorDict):
        """
        Return embeddings for distillation.

        :return:
            student_latent, teacher_latent,
            student_props_embed, teacher_props_embed,
            student_perc_embed, teacher_perc_embed
        """
        actor_low, actor_high = self.get_actor_obs(obs)
        critic_low, critic_high = self.get_critic_obs(obs)

        actor_low = self.actor_obs_normalizer(actor_low)
        critic_low = self.critic_obs_normalizer(critic_low)

        actor_perc = self._prepare_perception(actor_high)
        critic_perc = self._prepare_perception(critic_high)

        if self.use2Encoder:
            student_latent, _, student_prop, student_perc = self.actor_encoder(
                actor_perc, actor_low, embedding_only=True, return_intermediate=True
            )
            teacher_latent, _, teacher_prop, teacher_perc = self.critic_encoder(
                critic_perc, critic_low, embedding_only=True, return_intermediate=True
            )
        else:
            student_latent, _, student_prop, student_perc = self.encoder(
                actor_perc, actor_low, embedding_only=True, return_intermediate=True
            )
            teacher_latent, _, teacher_prop, teacher_perc = self.encoder(
                critic_perc, critic_low, embedding_only=True, return_intermediate=True
            )

        return (
            student_latent,
            teacher_latent,
            student_prop,
            teacher_prop,
            student_perc,
            teacher_perc,
        )

    def _prepare_perception(self, perception_obs: torch.Tensor) -> torch.Tensor:
        """
        :param perception_obs: [B,H,L,W,C] or [B,L,W,C] or [B,C,H,W]
        :return: [B,C,H,W]
        """
        if perception_obs.dim() == 5:
            perception_obs = perception_obs[:, -1, ...]
        if perception_obs.dim() != 4:
            raise ValueError("perception_obs must be 4D or 5D")

        if perception_obs.shape[1] == self.map_channels and perception_obs.shape[-1] != self.map_channels:
            return perception_obs
        if perception_obs.shape[-1] == self.map_channels:
            return perception_obs.permute(0, 3, 1, 2)
        raise ValueError("unable to infer channel dimension for perception_obs")
    
    def _gym_to_lab(self,obs:torch.Tensor,horizon:int,keep_dim=False)->torch.Tensor:
        """
        Brief:
            from gym style obs [O_1^1,...,O_1^d,...,O_H^1,...,O_H^d] to 
            lab style obs [O_{1:H}^1,...,O_{1:H}^d]
        Args:
            obs: shape [B, H*d]
            horizon: 时间步数 H
            keep_dim: 是否保持输入维度, False会返回[B,d,H]
        Returns:
            lab_style_obs :  shape [B, d*H]
        """
        B, total_dim = obs.shape
        d = total_dim // horizon
        
        # 检查维度是否可整除
        if total_dim % horizon != 0:
            raise ValueError(f"Total dimension {total_dim} must be divisible by horizon {horizon}")
        # 一步完成转换
        if (keep_dim):
            return obs.view(B, horizon, d).permute(0, 2, 1).reshape(B, -1)
        else:
            return obs.view(B, horizon, d).permute(0, 2, 1)  # [B,d,H]
    
    def _lab_to_gym(self,obs:torch.Tensor,horizon:int,keep_dim=False)->torch.Tensor:
        """
        Brief:
            from lab style obs [O_{1:H}^1,...,O_{1:H}^d] to 
            gym style obs [O_1^1,...,O_1^d,...,O_H^1,...,O_H^d]
        Args:
            obs: shape [B, d*H]
            horizon: 时间步数 H
            keep_dim: 是否保持输入的维度, False会返回[B,H,d]
        Returns:
            gym_style_obs :  shape [B, H*d]
        """
        B, total_dim = obs.shape
        d = total_dim // horizon

        # 检查维度是否可整除
        if total_dim % horizon != 0:
            raise ValueError(f"Total dimension {total_dim} must be divisible by horizon {horizon}")
        if keep_dim:
            return obs.view(B, d, horizon).permute(0, 2, 1).reshape(B, -1)
        else:
            return obs.view(B, d, horizon).permute(0, 2, 1)  # [B,H,d]

    def get_actor_obs(self, obs:TensorDict,style:str='lab')->tuple:
        """
        :param obs: TensorDict, each element shape maybe [B,H*d] or [B,H,d,...]
        :param style : 'lab' or 'gym', for lab style obs the permutation is 
            [O_{1:H}^1,O_{1:H}^2,...,O_{1:H}^n] where n is the index of part/group;
            for gym style obs , the permutation is [O_1^1,...,O_1^n,O_2^1,...,O_2^n,...,O_H^n]
        :return : tuple of TensorDict, each element shape is [B,H,d,...]
        """
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            # 这里假设每个group的历史堆叠形式是gym style的
            # if style == 'lab':
            #     gym_obs = self._lab_to_gym(obs[obs_group], self.horizon,keep_dim=False)  # [B,H,d]
            #     obs_list.append(gym_obs)
            # else:
            #     B = obs[obs_group].shape[0]
            #     obs_list.append(obs[obs_group].reshape(B,self.horizon,-1))  # [B,H,d_i]
            obs_list.append(obs[obs_group]) # [B,H,d_i]
        low_dim_obs = torch.cat(obs_list, dim=-1)  # [B,H,d]
        high_dim_obs_list = []
        for obs_group in self.obs_groups["perception"]:
            high_dim_obs_list.append(obs[obs_group])
        high_dim_obs = torch.cat(high_dim_obs_list, dim=-1)  
        return low_dim_obs,high_dim_obs

    def get_critic_obs(self, obs:TensorDict,style:str='lab')->tuple:
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            # 这里假设每个group的历史堆叠形式是gym style的
            # if style == 'lab':
            #     gym_obs = self._lab_to_gym(obs[obs_group], self.horizon,keep_dim=False)  # [B,H,d]
            #     obs_list.append(gym_obs)
            # else:
            #     B = obs[obs_group].shape[0]
            #     obs_list.append(obs[obs_group].reshape(B,self.horizon,-1))  # [B,H,d_i]
            obs_tensor = obs[obs_group]
            if obs_tensor.dim() > 2 and self.critic_single_frame:
                obs_tensor = obs_tensor[:, -1:, ...]
            obs_list.append(obs_tensor) # [B,H,d_i]
        low_dim_obs = torch.cat(obs_list, dim=-1)  # [B,H,d]
        high_dim_obs_list = []
        for obs_group in self.obs_groups["perception"]:
            high_dim_obs_list.append(obs[obs_group])
        high_dim_obs = torch.cat(high_dim_obs_list, dim=-1) 
        return low_dim_obs,high_dim_obs

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs,_ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs,_ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    # for state_dict :
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        module_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        # 不知道为什么, 不能加下面的
        # if self.actor_obs_normalization:
        #     module_dict[prefix + "actor_obs_normalizer"] = self.actor_obs_normalizer.state_dict()
        # if self.critic_obs_normalization:
        #     module_dict[prefix + "critic_obs_normalizer"] = self.critic_obs_normalizer.state_dict()
        return module_dict
    

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        TODO : 
            because of the encoder is independent of the actor and critic, so we need to load the encoder parameters separately. Besides, 
            in the different training process, the critic's observation space is same, hence we can load the critic's parameters directly.
        """
        # 存在的keys : {'std','encoder.*','actor.*','critic.*'}
        if self.load_mask & self.LOAD_POLICY_WEIGHTS:
            actor_state_dict = {k.replace('actor.', '',1): v for k, v in state_dict.items() if k.startswith('actor.')}
            self.actor.load_state_dict(actor_state_dict, strict=strict)
            print("=== EncActorCritic : Load Actor Weights ===")
            # TODO : 这里需要确认一下, 是否需要加载std的参数
        if self.load_mask & self.LOAD_CRITIC_WEIGHTS:
            critic_state_dict = {k.replace('critic.', '',1): v for k, v in state_dict.items() if k.startswith('critic.')}
            self.critic.load_state_dict(critic_state_dict, strict=strict)
            print("=== EncActorCritic : Load Critic Weights ===")
        if self.load_mask & self.LOAD_ENCODER_WEIGHTS & self.use2Encoder:
            enc_state_dict = {k.replace('encoder.', '',1): v for k, v in state_dict.items() if k.startswith('encoder.')}
            self.actor_encoder.load_state_dict(enc_state_dict, strict=strict)
            self.critic_encoder.load_state_dict(enc_state_dict, strict=strict)
            print("=== EncActorCritic : Load Encoder Weights (Actor/Critic) ===")
        if self.load_mask & self.LOAD_ENCODER_WEIGHTS & (not self.use2Encoder):
            enc_state_dict = {k.replace('encoder.', '',1): v for k, v in state_dict.items() if k.startswith('encoder.')}
            self.encoder.load_state_dict(enc_state_dict, strict=strict)
            print("=== EncActorCritic : Load Encoder Weights (Shared) ===")
        # 这里还需要load normalization的参数
        if (self.load_mask & self.LOAD_NORMALIZER_WEIGHTS):
            # if (self.actor_obs_normalization) and ('actor_obs_normalizer' in state_dict):
            if (self.actor_obs_normalization):
                act_obs_norm_state_dict = {k.replace('actor_obs_normalizer.', '',1): v for k, v in state_dict.items() if k.startswith('actor_obs_normalizer.')}
                self.actor_obs_normalizer.load_state_dict(act_obs_norm_state_dict)
                print("=== EncActorCritic : Load actor normalizer weights ===")
            # if (self.critic_obs_normalization) and ('critic_obs_normalizer' in state_dict):
            if (self.critic_obs_normalization):
                critic_obs_norm_state_dict = {k.replace('critic_obs_normalizer.', '',1): v for k, v in state_dict.items() if k.startswith('critic_obs_normalizer.')}
                self.critic_obs_normalizer.load_state_dict(critic_obs_norm_state_dict)
                print("=== EncActorCritic : Load critic normalizer weights ===")
        # super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes