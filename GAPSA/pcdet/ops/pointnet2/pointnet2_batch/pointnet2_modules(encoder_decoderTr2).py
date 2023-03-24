from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        return new_xyz, torch.cat(new_features_list, dim=1)


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

        self.pos_proj = nn.Sequential(
            nn.Conv1d(3, mlp[-1], 1, groups=1, bias=False),
            nn.ReLU(),
        )
        self.key_proj = nn.Sequential(
            nn.Conv1d(mlp[-1], mlp[-1], 1, groups=1, bias=False),
            nn.ReLU()
        )
        self.value_proj = nn.Sequential(
            nn.Conv1d(mlp[-1], mlp[-1]/2, 1, groups=1, bias=False),
            nn.BatchNorm1d(mlp[-1]/2),
            nn.ReLU(),
            nn.Conv1d(mlp[-1]/2, mlp[-1], 1, groups=1, bias=False),
            nn.BatchNorm1d(mlp[-1]),
            nn.ReLU(),
        )
        self.attention_proj = nn.Sequential(
            nn.Conv1d(mlp[-1], 4, 1, groups=1, bias=False),
        )
        self.norm_layer = nn.Softmax(dim=-1)
        self.k_coef = nn.Sequential(
            nn.Linear(mlp[-1], 1, bias=False),
            nn.Sigmoid()
        )
        self.q_coef = nn.Sequential(
            nn.Linear(mlp[-1], 1, bias=False),
            nn.Sigmoid()
        )
        self.qk_coef = nn.Sequential(
            nn.Linear(mlp[-1], 1, bias=False),
            nn.Sigmoid()
        )
        self.v_coef = nn.Sequential(
            nn.Linear(mlp[-1], 1, bias=False),
            nn.Sigmoid()
        )
        self.head_dim = int(mlp[-1]/4)
        self.input_dim = mlp[-1]

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
        new_features = self.mlp(new_features).squeeze(-1)
        #print(new_features.shape,unknown.shape) (B,512,256)(B,256,3)

        pos_embedding = self.pos_proj(unknown.permute(0,2,1))
        key_embedding = self.key_proj(new_features)
        value_embedding = self.value_proj(new_features)
        pos_key_embedding = pos_embedding * key_embedding

        v_coef = self.v_coef(pos_embedding.transpose(1,2).contiguous().view(-1, self.input_dim))  # (B*npoints,64)
        q_coef = self.q_coef(pos_embedding.transpose(1,2).contiguous().view(-1, self.input_dim))
        k_coef = self.k_coef(key_embedding.transpose(1,2).contiguous().view(-1, self.input_dim))
        qk_coef = self.qk_coef(pos_key_embedding.transpose(1,2).contiguous().view(-1, self.input_dim))

        value_embedding = value_embedding + pos_embedding * v_coef.squeeze(1).view(pos_embedding.shape[0],1,-1)  # (B*npoints,64,nsample)
        # value_embedding = value_embedding + pos_embedding
        attention_embedding = pos_embedding * q_coef.squeeze(1).view(pos_embedding.shape[0],1,-1) + key_embedding * k_coef.squeeze(1).view(pos_embedding.shape[0],1,-1)  # + pos_key_embedding * qk_coef
        #attention_embedding = pos_embedding + key_embedding #+ pos_key_embedding

        attention_map = self.attention_proj(attention_embedding)
        attention_map = self.norm_layer(attention_map)
        # (N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
        attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dim, 1).reshape(attention_map.shape[0], -1,
                                                                                           attention_map.shape[-1])

        new_features = attention_map * value_embedding

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
