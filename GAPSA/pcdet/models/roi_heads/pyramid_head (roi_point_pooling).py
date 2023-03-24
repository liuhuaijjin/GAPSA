import torch.nn as nn
import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from .ctrans import build_transformer
import random
import numpy as np

class PyramidRoIHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        # mlps are shared with each grid point
        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        self.num_pyramid_levels = len(mlps)

        self.radius_by_rois = self.model_cfg.ROI_GRID_POOL.RADIUS_BY_ROIS
        self.radii = self.model_cfg.ROI_GRID_POOL.POOL_RADIUS
        self.enlarge_ratios = self.model_cfg.ROI_GRID_POOL.ENLARGE_RATIO
        self.grid_sizes = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.nsamples = self.model_cfg.ROI_GRID_POOL.NSAMPLE

        self.num_heads = self.model_cfg.ROI_GRID_POOL.NUM_HEADS
        self.head_dims = self.model_cfg.ROI_GRID_POOL.HEAD_DIMS
        self.attention_op = self.model_cfg.ROI_GRID_POOL.ATTENTION_OP
        assert len(self.radii) == len(self.enlarge_ratios) == len(self.grid_sizes) == len(self.nsamples) == self.num_pyramid_levels

        self.dp_value = self.model_cfg.ROI_GRID_POOL.get('DP_RATIO', 0.1)
        self.tr_mode = self.model_cfg.ROI_GRID_POOL.get('TR_MODE', 'Normal')

        self.roi_grid_pool_layer = pointnet2_stack_modules.PyramidModule(
            input_channels = input_channels,
            nsamples = self.nsamples,
            grid_sizes = self.grid_sizes,
            num_heads = self.num_heads,
            head_dims = self.head_dims,
            attention_op = self.attention_op,
            dp_value = self.dp_value,
            tr_mode = self.tr_mode,
        )

        #self.query_embed = nn.Embedding(self.model_cfg.Transformer.num_queries,self.model_cfg.Transformer.hidden_dim)
        #self.transformer = build_transformer(self.model_cfg.Transformer)

        pre_channel = 0
        for i in range(self.num_pyramid_levels):
            pre_channel += (self.grid_sizes[i] ** 3) * mlps[i][-1]

        """
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        """
        xyz_mlps = [5] + [128,128]
        shared_mlps = []
        use_bn=False
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)
        c_out = 128
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel+65536, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel+65536,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
        Returns:
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        num_rois = rois.shape[1]
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        new_xyz_list = []
        new_xyz_r_list = []
        new_xyz_batch_cnt_list = []
        for i in range(len(self.grid_sizes)):
            global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_enlarged_roi(
                rois, grid_size = self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i]
            ) # (BN,grid_size^3, 3)
            global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3).contiguous() # (B, N x grid_size^3, 3)
            if self.radius_by_rois:
                roi_grid_radius = self.get_radius_by_enlarged_roi(
                    rois, grid_size= self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i], radius_ratio = self.radii[i]
                )
                roi_grid_radius = roi_grid_radius.view(batch_size, -1, 1).contiguous() # (B, N x grid_size^3, 1)
            else:
                roi_grid_radius = rois.new_zeros(batch_size, num_rois * self.grid_sizes[i] * self.grid_sizes[i] * self.grid_sizes[i], 1).fill_(self.radii[i])
            new_xyz_list.append(global_roi_grid_points)
            new_xyz_r_list.append(roi_grid_radius)
            new_xyz_batch_cnt_list.append(roi_grid_radius.new_zeros(batch_size).int().fill_(roi_grid_radius.shape[1]))

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        cls_features, reg_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz_list=new_xyz_list,
            new_xyz_r_list=new_xyz_r_list,
            new_xyz_batch_cnt_list=new_xyz_batch_cnt_list,
            features=point_features.contiguous(),
            batch_size = batch_size,
            num_rois = num_rois,
        )  # (BN, \sum(grid_size^3), C)

        return cls_features, reg_features

    def get_radius_by_enlarged_roi(self, rois, grid_size, enlarged_ratio, radius_ratio):
        rois = rois.view(-1, rois.shape[-1])

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        roi_grid_radius = (enlarged_rois[:, 3:6] ** 2).sum(dim = 1).sqrt() # base_radius
        roi_grid_radius *= radius_ratio
        roi_grid_radius = roi_grid_radius.view(-1, 1, 1).repeat(1, grid_size ** 3, 1).contiguous() # (BN, grid_size^3, 1)
        return roi_grid_radius

    def get_global_grid_points_of_enlarged_roi(self, rois, grid_size, enlarged_ratio):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        local_roi_grid_points = self.get_dense_grid_points(enlarged_rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), enlarged_rois[:, 6]
        ) #.squeeze(dim=1)
        global_center = enlarged_rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (BN, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def roipool3d_gpu(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1)
        batch_points = point_coords.view(batch_size, -1, 3)
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

        with torch.no_grad():
            # pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
            #    batch_points, batch_point_features, rois
            # )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)
            batch_size = rois.shape[0]
            num_rois = rois.shape[1]
            num_sample = self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS
            pooled_points = rois.new_zeros(batch_size, num_rois, num_sample, 3)
            pooled_features = rois.new_zeros(batch_size, num_rois, num_sample, 130)
            pooled_empty_flag = rois.new_zeros(batch_size, num_rois)
            for bs_idx in range(batch_size):
                cur_points = batch_points[bs_idx]  # (16384,3)
                cur_features = batch_point_features[bs_idx]  # (16384,130)
                cur_batch_boxes = rois[bs_idx]  # (64,7),[x,y,z,h,w,l,ry]

                cur_radiis = torch.sqrt((cur_batch_boxes[:, 4] / 2) ** 2 + (cur_batch_boxes[:, 5] / 2) ** 2) * 1.2
                dis = torch.norm((cur_points[:, :2].unsqueeze(0) - cur_batch_boxes[:, :2].unsqueeze(1).repeat(1,cur_points.shape[0],1)),dim=2)
                point_mask = (dis <= cur_radiis.unsqueeze(-1))

                for roi_box_idx in range(0, num_rois):
                    cur_roi_points = cur_points[point_mask[roi_box_idx]]
                    cur_roi_features = cur_features[point_mask[roi_box_idx]]

                    if cur_roi_points.shape[0] >= num_sample:
                        random.seed(0)
                        index = np.random.randint(cur_roi_points.shape[0], size=num_sample)
                        cur_roi_points_sample = cur_roi_points[index]
                        cur_roi_features_sample = cur_roi_features[index]
                    elif cur_roi_points.shape[0] == 0:
                        cur_roi_points_sample = cur_roi_points.new_zeros(num_sample, 3)
                        cur_roi_features_sample = cur_roi_features.new_zeros(num_sample, 130)
                        pooled_empty_flag[bs_idx, roi_box_idx] = roi_box_idx
                    else:
                        empty_num = num_sample - cur_roi_points.shape[0]
                        add_zeros = cur_roi_points.new_zeros(empty_num, 3)
                        add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                        add_zeros_features = cur_roi_features[0].repeat(empty_num, 1)
                        cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim=0)
                        cur_roi_features_sample = torch.cat([cur_roi_features, add_zeros_features], dim=0)
                    pooled_points[bs_idx, roi_box_idx, :, :] = cur_roi_points_sample  # (B,64,512,3)
                    pooled_features[bs_idx, roi_box_idx, :, :] = cur_roi_features_sample  # (B,64,512,130)
            pooled_features = torch.cat((pooled_points, pooled_features), dim=-1)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

            pooled_features = pooled_features.view(-1, pooled_features.shape[-2],pooled_features.shape[-1])  # (B*64,512,3+130)
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_features

    def forward(self, batch_dict):
        """
        :param:
                points: [BN,5],(idx,x,y,z,intensity)
                gt_boxes: [BM,8]
                point_features: [BN,128]
                point_coords: [BN,4],(idx,x,y,z)
                point_cls_score: [BN]
                point_cls_pred: [BN,1]
                point_box_pred: [BN,7]
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # point_coords = batch_dict['point_coords']
        # src = batch_dict['point_features']
        # src = batch_dict['rois']
        # pos=torch.zeros_like(src)
        # hs = self.transformer(src,self.query_embed.weight, pos)[0]
        # print(hs.shape,2)

        # RoI point pooling
        pooled_features = self.roipool3d_gpu(batch_dict)  #(total_rois,num_sampled_points,3+C)(B*128,512,128+5)
        #batch_size_rcnn = pooled_features.shape[0]
        #pooled_features = pooled_features.reshape(batch_size_rcnn,-1,1) #(Bx128,512*133,1)
        xyz_input = pooled_features[..., 0:5].transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled_features[..., 5:].transpose(1, 2).unsqueeze(dim=3)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)
        batch_size_rcnn = merged_features.shape[0]
        merged_features = merged_features.reshape(batch_size_rcnn,-1,1) #(Bx128,512*128,1)

        # RoI aware pooling
        cls_features, reg_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        batch_size_rcnn = cls_features.shape[0]
        cls_features = cls_features.reshape(batch_size_rcnn, -1, 1) # (Bx128,409*64,1)
        reg_features = reg_features.reshape(batch_size_rcnn, -1, 1)

        cls_features = torch.cat((cls_features,merged_features),dim=1)
        reg_features = torch.cat((reg_features,merged_features),dim=1)

        rcnn_cls = self.cls_layers(cls_features).squeeze(dim=-1).contiguous()  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(reg_features).squeeze(dim=-1).contiguous()  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict

class PyramidRoIHeadV2(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        # mlps are shared with each grid point
        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        self.num_pyramid_levels = len(mlps)

        self.radius_by_rois = self.model_cfg.ROI_GRID_POOL.RADIUS_BY_ROIS
        self.radii = self.model_cfg.ROI_GRID_POOL.POOL_RADIUS
        self.enlarge_ratios = self.model_cfg.ROI_GRID_POOL.ENLARGE_RATIO
        self.grid_sizes = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.nsamples = self.model_cfg.ROI_GRID_POOL.NSAMPLE

        self.num_heads = self.model_cfg.ROI_GRID_POOL.NUM_HEADS
        self.head_dims = self.model_cfg.ROI_GRID_POOL.HEAD_DIMS
        self.attention_op = self.model_cfg.ROI_GRID_POOL.ATTENTION_OP
        assert len(self.radii) == len(self.enlarge_ratios) == len(self.grid_sizes) == len(self.nsamples) == self.num_pyramid_levels

        self.predict_radii = self.model_cfg.ROI_GRID_POOL.PRE_RADII
        self.predict_ns = self.model_cfg.ROI_GRID_POOL.PRE_NS
        self.predict_norm = self.model_cfg.ROI_GRID_POOL.PRE_NORM
        self.use_weights_before = self.model_cfg.ROI_GRID_POOL.USE_WEIGHTS_BEFORE

        self.roi_grid_pool_layer = pointnet2_stack_modules.PyramidModuleV2(
            input_channels = input_channels,
            nsamples = self.nsamples,
            grid_sizes = self.grid_sizes,
            num_heads = self.num_heads,
            head_dims = self.head_dims,
            attention_op = self.attention_op,
            predict_radii = self.predict_radii,
            predict_ns = self.predict_ns,
            norm_factors = self.predict_norm,
            pre_weights = self.use_weights_before,
        )

        pre_channel = 0
        for i in range(self.num_pyramid_levels):
            pre_channel += (self.grid_sizes[i] ** 3) * mlps[i][-1]

        """
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        """

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        num_rois = rois.shape[1]
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        new_xyz_list = []
        new_xyz_r_list = []
        new_xyz_batch_cnt_list = []
        for i in range(len(self.grid_sizes)):
            global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_enlarged_roi(
                rois, grid_size = self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i]
            ) #(BN, grid_size^3, 3)
            global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3).contiguous() # (B, N x grid_size^3, 3)
            if self.radius_by_rois:
                roi_grid_radius = self.get_radius_by_enlarged_roi(
                    rois, grid_size= self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i], radius_ratio = self.radii[i]
                )
                roi_grid_radius = roi_grid_radius.view(batch_size, -1, 1).contiguous() # (B, N x grid_size^3, 1)
            else:
                roi_grid_radius = rois.new_zeros(batch_size, num_rois * self.grid_sizes[i] * self.grid_sizes[i] * self.grid_sizes[i], 1).fill_(self.radii[i])

            new_xyz_list.append(global_roi_grid_points)
            new_xyz_r_list.append(roi_grid_radius)
            new_xyz_batch_cnt_list.append(roi_grid_radius.new_zeros(batch_size).int().fill_(roi_grid_radius.shape[1]))

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        anchor_xyz =  rois[..., :3].contiguous().view(-1, 3) # take center of each roi as anchor points
        anchor_batch_cnt = anchor_xyz.new_zeros(batch_size).fill_(num_rois).int()
        temperature = batch_dict['temperature']

        cls_features, reg_features = self.roi_grid_pool_layer(
            anchor_xyz = anchor_xyz,
            anchor_batch_cnt = anchor_batch_cnt,
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz_list=new_xyz_list,
            new_xyz_r_list=new_xyz_r_list,
            new_xyz_batch_cnt_list=new_xyz_batch_cnt_list,
            features=point_features.contiguous(),
            batch_size = batch_size,
            num_rois = num_rois,
            temperature = temperature,
        )  # (BN, \sum(grid_size^3), C)

        return cls_features, reg_features

    def get_radius_by_enlarged_roi(self, rois, grid_size, enlarged_ratio, radius_ratio):
        rois = rois.view(-1, rois.shape[-1])

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        roi_grid_radius = (enlarged_rois[:, 3:6] ** 2).sum(dim = 1).sqrt() # base_radius
        roi_grid_radius *= radius_ratio
        roi_grid_radius = roi_grid_radius.view(-1, 1, 1).repeat(1, grid_size ** 3, 1).contiguous() # (BN, grid_size^3, 1)
        return roi_grid_radius

    def get_global_grid_points_of_enlarged_roi(self, rois, grid_size, enlarged_ratio):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        local_roi_grid_points = self.get_dense_grid_points(enlarged_rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), enlarged_rois[:, 6]
        ) #.squeeze(dim=1)
        global_center = enlarged_rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (BN, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict, targets_dict):
        """
        :param input_data: input dict
        :return:
        """

        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        """

        # RoI aware pooling
        cls_features, reg_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        batch_size_rcnn = cls_features.shape[0]
        cls_features = cls_features.reshape(batch_size_rcnn, -1, 1)
        reg_features = reg_features.reshape(batch_size_rcnn, -1, 1)

        rcnn_cls = self.cls_layers(cls_features).squeeze(dim=-1).contiguous()  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(reg_features).squeeze(dim=-1).contiguous()  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
