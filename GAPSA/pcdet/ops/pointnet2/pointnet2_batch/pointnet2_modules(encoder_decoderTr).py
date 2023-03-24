from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from . import pointnet2_utils
from ...pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.config import cfg

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

        self.transformer_decoders = nn.ModuleList()
        encoder_layer = TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.transformer_decoders.append(TransformerEncoder(encoder_layer=encoder_layer, num_layers=1, norm=None))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz=xyz.contiguous(),
                new_xyz=new_xyz,
                features=features.contiguous()
            )  # (BN, \sum(grid_size^3), C)

            #new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            # if self.pool_method == 'max_pool':
            #     new_features = F.max_pool2d(
            #         new_features, kernel_size=[1, new_features.size(3)]
            #     )  # (B, mlp[-1], npoint, 1)
            # elif self.pool_method == 'avg_pool':
            #     new_features = F.avg_pool2d(
            #         new_features, kernel_size=[1, new_features.size(3)]
            #     )  # (B, mlp[-1], npoint, 1)
            # else:
            #     raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
            new_features= torch.cat(new_features_list,dim=1)

        new_features = new_features.squeeze(-1).permute(2, 0, 1).contiguous()  # (N, B, C)
        print(new_features.shape)
        new_features = self.transformer_decoders[0](new_features)
        print(new_features.shape)
        new_features = new_features.permute(1, 2, 0).contiguous()  # (B, C, N)

        return new_xyz, new_features #torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool', grid_size:int, input_channels:int):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            # self.groupers.append(
            #     pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
            #     if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            # )
            self.groupers.append(
                pointnet2_stack_modules.PyramidModuleV1(
                    input_channels=input_channels,
                    nsamples=nsample,
                    radius=radius,
                    grid_sizes=grid_size,
                    num_heads=cfg.MODEL.BACKBONE_3D.SA_CONFIG.NUM_HEADS,
                    head_dims=cfg.MODEL.BACKBONE_3D.SA_CONFIG.HEAD_DIMS,
                    attention_op=cfg.MODEL.BACKBONE_3D.SA_CONFIG.ATTENTION_OP,
                    dp_value=cfg.MODEL.BACKBONE_3D.SA_CONFIG.get('DP_RATIO', 0.1),
                    tr_mode=cfg.MODEL.BACKBONE_3D.SA_CONFIG.get('TR_MODE', 'NoTr'),
                )
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

        self.transformer_decoders=nn.ModuleList()
        encoder_layer = TransformerEncoderLayer(
            d_model=mlp[-1],
            nhead=4,
            dim_feedforward=mlp[-1],
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.transformer_decoders.append(TransformerEncoder(encoder_layer=encoder_layer, num_layers=1, norm=None))

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(-1).permute(2, 0, 1).contiguous()  # (N, B, C)
        new_features = self.transformer_decoders[0](new_features)
        new_features = new_features.permute(1, 2, 0).contiguous()  # (B, C, N)

        return new_features.squeeze(-1)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask = None,
                src_key_padding_mask = None,
                pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == "__main__":
    pass
