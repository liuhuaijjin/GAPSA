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
            new_features = self.stack_attention[i](new_features,new_xyz)

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
        self.stack_attention = nn.ModuleList()

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

            self.stack_attention.append(StackedAttention())

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

        return new_features.squeeze(-1)


# class SA_Layer(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
#         self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
#         self.q_conv.weight = self.k_conv.weight
#         self.v_conv = nn.Conv1d(channels, channels, 1)
#         self.trans_conv = nn.Conv1d(channels, channels, 1)
#         self.after_norm = nn.BatchNorm1d(channels)
#         self.act = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
#         x_k = self.k_conv(x)  # b, c, n
#         x_v = self.v_conv(x)
#         energy = x_q @ x_k  # b, n, n
#         attention = self.softmax(energy)
#         attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
#         x_r = x_v @ attention  # b, c, n
#         x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
#         x = x + x_r
#         return x

# class StackedAttention(nn.Module):
#     def __init__(self, channels=64):
#         super().__init__()
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
#
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.bn2 = nn.BatchNorm1d(channels)
#
#         self.sa1 = SA_Layer(channels)
#         self.sa2 = SA_Layer(channels)
#         self.sa3 = SA_Layer(channels)
#         self.sa4 = SA_Layer(channels)
#
#         self.relu = nn.ReLU()
#
#         self.conv_fuse = nn.Sequential(nn.Conv1d(320, 64, kernel_size=1, bias=False),
#                                        nn.BatchNorm1d(64),
#                                        #nn.LeakyReLU(negative_slope=0.2),
#                                        nn.ReLU())
#
#     def forward(self, x):
#         #
#         # b, 3, npoint, nsample
#         # conv2d 3 -> 128 channels 1, 1
#         # b * npoint, c, nsample
#         # permute reshape
#         batch_size, _, N = x.size()
#
#         #x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
#         #x = self.relu(self.bn2(self.conv2(x)))
#
#         x1 = self.sa1(x)
#         x2 = self.sa2(x1)
#         x3 = self.sa3(x2)
#         x4 = self.sa4(x3)
#
#         x5 = torch.cat((x1, x2, x3, x4), dim=1)
#
#         x = torch.cat([x5, x], dim=1)
#         x = self.conv_fuse(x)
#
#         return x
class StackedAttention(nn.Module):
    def __init__(self, args, channels=64):
        super(StackedAttention, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pos_xyz = nn.Conv1d(3, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.conv_fuse = nn.Sequential(nn.Conv1d(320, 64, kernel_size=1, bias=False),
                                                nn.BatchNorm1d(64),
                                                nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, xyz):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()
        xyz = xyz.permute(0, 2, 1)
        xyz = self.pos_xyz(xyz)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        x5 = torch.cat((x1, x2, x3, x4), dim=1)

        x = torch.cat([x5, x], dim=1)
        x = self.conv_fuse(x)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xyz):
        # b, n, c
        x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

if __name__ == "__main__":
    pass
