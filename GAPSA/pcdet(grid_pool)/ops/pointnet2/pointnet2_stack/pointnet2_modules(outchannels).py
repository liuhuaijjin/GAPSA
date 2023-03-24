from typing import List
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
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

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features

class StackSAModulePyramid(nn.Module):

    def __init__(self, *, mlps: List[List[int]], nsamples, use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features

class PyramidModuleV1(nn.Module):

    def __init__(self, input_channels, output_channels, nsamples, radius, grid_sizes, num_heads, head_dims, attention_op, dp_value = 0.1, tr_mode = 'Residual'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        """
        super().__init__()
        input_channels += 3
        self.num_pyramid_levels = 1
        self.pos_dims = 3
        self.num_heads = num_heads
        self.radius = radius
        #self.output_dims = self.num_heads * self.head_dims
        self.output_dims = output_channels[-1]
        # self.head_dims = head_dims
        self.head_dims = int(self.output_dims / self.num_heads)
        self.grid_sizes = grid_sizes

        self.tr_mode = tr_mode
        assert self.tr_mode in ['NoTr', 'Normal', 'Residual']

        if self.tr_mode != 'NoTr':
            self.transformer_encoders = nn.ModuleList()
        self.cls_weight_module = nn.ModuleList()
        self.reg_weight_module = nn.ModuleList()

        self.pos_embeddings = []

        self.groupers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.k_coef = nn.ModuleList()
        self.qk_coef = nn.ModuleList()
        self.q_coef = nn.ModuleList()
        self.v_coef = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples
            grid_size = grid_sizes
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramidAttention(nsample))
            # vmh 1
            self.pos_proj.append(nn.Sequential(
                nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU(),
            ))
            self.key_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU()
            ))
            self.value_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
            self.attention_proj.append(nn.Sequential(
                nn.Conv1d(self.output_dims, self.num_heads, 1, groups=1, bias=False),
            ))
            self.norm_layer.append(nn.Softmax(dim=-1))
            self.k_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.q_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.qk_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.v_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))

            num_out_ch = self.output_dims

            if self.tr_mode != 'NoTr':
                encoder_layer = TransformerEncoderLayer(
                    d_model=num_out_ch,
                    nhead=4,
                    dim_feedforward=num_out_ch,
                    dropout=dp_value,
                    activation="relu",
                    normalize_before=True,
                )
                self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = 1, norm = None))
                self.pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((1 ** 3, num_out_ch)).cuda()
                    )
                )

            self.cls_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch//2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))
            self.reg_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch //2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))

        self.attention_op = attention_op
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, new_xyz, features=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size), [B, N x grid_size^3]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        reg_features_list = []
        for i in range(self.num_pyramid_levels):
            xyz = xyz.view(-1,3).contiguous() #(BN,3)
            xyz_batch_cnt = xyz.new_zeros(1).int().fill_(xyz.shape[0]) #(BN)
            features = features.permute(0,2,1)
            features = features.reshape(-1,features.shape[-1]).contiguous() #(BN,C)

            batch_size = new_xyz.shape[0]
            num_rois = new_xyz.shape[1]
            new_xyz = new_xyz.view(-1, 3).contiguous() #(B*npoints,3)
            new_xyz_r = new_xyz.new(new_xyz.shape[0], 1).fill_(self.radius)  # (B*npoints,1)
            new_xyz_batch_cnt = new_xyz.new_zeros(1).int().fill_(new_xyz.shape[0]) #(B*npoints)

            #print(xyz.shape,xyz_batch_cnt,new_xyz.shape,new_xyz_batch_cnt,features.shape)
            grouped_xyz, grouped_features, empty_mask = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            #print(grouped_xyz.shape,grouped_features.shape) #(B*npoints,3,nsample)(B*npoints,C+3,nsample)

            pos_embedding = self.pos_proj[i](grouped_xyz) #(B*npoints,64,nsample)
            key_embedding = self.key_proj[i](grouped_features)
            value_embedding = self.value_proj[i](grouped_features)
            pos_key_embedding = pos_embedding * key_embedding

            v_coef = self.v_coef[i](pos_embedding.mean(2)) #(B*npoints,64)
            q_coef = self.q_coef[i](pos_embedding.mean(2))
            k_coef = self.k_coef[i](key_embedding.mean(2))
            qk_coef = self.qk_coef[i](pos_key_embedding.mean(2))

            value_embedding = value_embedding + pos_embedding * v_coef.unsqueeze(2) #(B*npoints,64,nsample)
            attention_embedding = pos_embedding * q_coef.unsqueeze(2) + key_embedding * k_coef.unsqueeze(2) + pos_key_embedding * qk_coef.unsqueeze(2)
            attention_map = self.attention_proj[i](attention_embedding) #(B*npoints,num_heads,nsample)
            attention_map = self.norm_layer[i](attention_map)
            #(N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
            attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dims, 1).reshape(attention_map.shape[0], -1, attention_map.shape[-1])

            attend_features = (attention_map * value_embedding).sum(-1)

            new_features = attend_features

            num_features = new_features.shape[1]
            new_features = new_features.reshape(batch_size * num_rois, -1, num_features) #(B*npoints,1,64)

            if self.tr_mode == 'NoTr':
                pass
            elif self.tr_mode == 'Normal':
                new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                #pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                #new_features = self.transformer_encoders[i](new_features, pos=pos_emb)
                new_features = self.transformer_encoders[i](new_features)
                new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
            elif self.tr_mode == 'Residual':
                tr_new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                #pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, tr_new_features.shape[1], 1)  # (L, B, C)
                #tr_new_features = self.transformer_encoders[i](tr_new_features, pos=pos_emb)
                tr_new_features = self.transformer_encoders[i](tr_new_features)
                tr_new_features = tr_new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
                new_features = new_features + tr_new_features
            else:
                raise NotImplementedError

            reg_features = new_features.squeeze(1).reshape(batch_size,num_rois,-1).permute(0,2,1)
            #reg_weights = self.reg_weight_module[i](new_features)  # (B, L, 1)
            #reg_features = new_features * reg_weights
            reg_features_list.append(reg_features)

        reg_features = torch.cat(reg_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return reg_features.contiguous()

class PyramidModule(nn.Module):

    def __init__(self, input_channels, nsamples, grid_sizes, num_heads, head_dims, attention_op, dp_value = 0.1, tr_mode = 'Normal'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        """
        super().__init__()
        input_channels += 3
        self.num_pyramid_levels = len(nsamples)
        self.pos_dims = 3
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.output_dims = self.num_heads * self.head_dims
        self.grid_sizes = grid_sizes

        self.tr_mode = tr_mode
        assert self.tr_mode in ['NoTr', 'Normal', 'Residual']

        if self.tr_mode != 'NoTr':
            self.transformer_encoders = nn.ModuleList()
        self.cls_weight_module = nn.ModuleList()
        self.reg_weight_module = nn.ModuleList()

        self.pos_embeddings = []

        self.groupers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.k_coef = nn.ModuleList()
        self.qk_coef = nn.ModuleList()
        self.q_coef = nn.ModuleList()
        self.v_coef = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramidAttention(nsample))
            # vmh 1
            self.pos_proj.append(nn.Sequential(
                nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU(),
            ))
            self.key_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU()
            ))
            self.value_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
            self.attention_proj.append(nn.Sequential(
                nn.Conv1d(self.output_dims, self.num_heads, 1, groups=1, bias=False),
            ))
            self.norm_layer.append(nn.Softmax(dim=-1))
            self.k_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.q_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.qk_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.v_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))

            num_out_ch = self.output_dims

            if self.tr_mode != 'NoTr':
                encoder_layer = TransformerEncoderLayer(
                    d_model=num_out_ch,
                    nhead=4,
                    dim_feedforward=num_out_ch,
                    dropout=dp_value,
                    activation="relu",
                    normalize_before=False,
                )
                self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = 1, norm = None))
                self.pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((grid_size ** 3, num_out_ch)).cuda()
                    )
                )

            self.cls_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch//2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))
            self.reg_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch //2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))

        self.attention_op = attention_op
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list,
                features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size), N x grid_size^3
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        cls_features_list = []
        reg_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()

            grouped_xyz, grouped_features, empty_mask = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )

            pos_embedding = self.pos_proj[i](grouped_xyz)
            key_embedding = self.key_proj[i](grouped_features)
            value_embedding = self.value_proj[i](grouped_features)
            pos_key_embedding = pos_embedding * key_embedding

            v_coef = self.v_coef[i](pos_embedding.mean(2))
            q_coef = self.q_coef[i](pos_embedding.mean(2))
            k_coef = self.k_coef[i](key_embedding.mean(2))
            qk_coef = self.qk_coef[i](pos_key_embedding.mean(2))

            value_embedding = value_embedding + pos_embedding * v_coef.unsqueeze(2)
            attention_embedding = pos_embedding * q_coef.unsqueeze(2) + key_embedding * k_coef.unsqueeze(2) + pos_key_embedding * qk_coef.unsqueeze(2)

            attention_map = self.attention_proj[i](attention_embedding)
            attention_map = self.norm_layer[i](attention_map)
            # (N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
            attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dims, 1).reshape(attention_map.shape[0], -1, attention_map.shape[-1])

            attend_features = (attention_map * value_embedding).sum(-1)

            new_features = attend_features

            num_features = new_features.shape[1]
            new_features = new_features.reshape(batch_size * num_rois, -1, num_features)

            if self.tr_mode == 'NoTr':
                pass
            elif self.tr_mode == 'Normal':
                new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                new_features = self.transformer_encoders[i](new_features, pos=pos_emb)
                new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
            elif self.tr_mode == 'Residual':
                tr_new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, tr_new_features.shape[1], 1)  # (L, B, C)
                tr_new_features = self.transformer_encoders[i](tr_new_features, pos=pos_emb)
                tr_new_features = tr_new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
                new_features = new_features + tr_new_features
            else:
                raise NotImplementedError

            cls_weights = self.cls_weight_module[i](new_features)  # (B, L, 1)
            cls_features = new_features * cls_weights
            cls_features_list.append(cls_features)

            reg_weights = self.reg_weight_module[i](new_features)  # (B, L, 1)
            reg_features = new_features * reg_weights
            reg_features_list.append(reg_features)

        cls_features = torch.cat(cls_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)
        reg_features = torch.cat(reg_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return cls_features.contiguous(), reg_features.contiguous()

class StackPointnetFPModule(nn.Module):
    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
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

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features

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