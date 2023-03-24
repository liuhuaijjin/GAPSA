import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils
from pcdet import config as cfg

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)

class CenterNetFocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""
    def __init__(self):
        super(CenterNetFocalLoss, self).__init__()

    def _neg_loss(self, pred, gt):
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
              pred (batch x c x h x w)
              gt_regr (batch x c x h x w)
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def forward(self, out, target):
        return self._neg_loss(out, target)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3)).contiguous()
    feat = _gather_feat(feat, ind)
    return feat.contiguous()

class CenterNetRegLoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(CenterNetRegLoss, self).__init__()

    def _reg_loss(self, regr, gt_regr, mask):
        """ L1 regression loss
            Arguments:
            regr (batch x max_objects x dim)
            gt_regr (batch x max_objects x dim)
            mask (batch x max_objects)
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()
        isnotnan = (~ torch.isnan(gt_regr)).float()
        mask *= isnotnan
        regr = regr * mask
        gt_regr = gt_regr * mask

        loss = torch.abs(regr - gt_regr)
        loss = loss.transpose(2, 0).contiguous()

        loss = torch.sum(loss, dim=2)
        loss = torch.sum(loss, dim=1)

        loss = loss / (num + 1e-4)
        return loss

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = self._reg_loss(pred, target, mask)
        return loss

class CenterNetSmoothRegLoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(CenterNetSmoothRegLoss, self).__init__()

    def _smooth_reg_loss(self, regr, gt_regr, mask, sigma=3):
        """ L1 regression loss
          Arguments:
            regr (batch x max_objects x dim)
            gt_regr (batch x max_objects x dim)
            mask (batch x max_objects)
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()
        isnotnan = (~ torch.isnan(gt_regr)).float()
        mask *= isnotnan
        regr = regr * mask
        gt_regr = gt_regr * mask

        abs_diff = torch.abs(regr - gt_regr)

        abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)

        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + (
                abs_diff - 0.5 / (sigma ** 2)
        ) * (1.0 - abs_diff_lt_1)

        loss = loss.transpose(2, 0).contiguous()

        loss = torch.sum(loss, dim=2)
        loss = torch.sum(loss, dim=1)

        loss = loss / (num + 1e-4)
        return loss

    def forward(self, output, mask, ind, target, sin_loss):
        assert sin_loss is False
        pred = _transpose_and_gather_feat(output, ind)
        loss = self._smooth_reg_loss(pred, target, mask)
        return loss

class BinbasedSmoothRegLoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """
    def __init__(self):
        super(BinbasedSmoothRegLoss, self).__init__()

    def forward(self, cls_score, mask_score, pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                     get_xz_fine = True, get_y_by_bin = False, loc_y_scope = 0.5, loc_y_bin_size = 0.25,
                     get_ry_fine = False,
                     use_cls_score = False, use_mask_score = False,
                     gt_iou_weight = None,
                     use_iou_branch=False,
                     iou_branch_pred=None
                     ):
        """
        Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.
        :param pred_reg: (N, C)
        :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
        :param loc_scope: constant
        :param loc_bin_size: constant
        :param num_head_bin: constant
        :param anchor_size: (N, 3) or (3)
        :param get_xz_fine:
        :param get_y_by_bin:
        :param loc_y_scope:
        :param loc_y_bin_size:
        :param get_ry_fine:
        :return:
        """
        per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
        loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

        reg_loss_dict = { }
        loc_loss = 0

        # xz localization loss
        x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 1], reg_label[:, 2]
        x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
        z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
        x_bin_label = (x_shift / loc_bin_size).floor().long()
        z_bin_label = (z_shift / loc_bin_size).floor().long()

        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r

        loss_x_bin = F.cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
        loss_z_bin = F.cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
        reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
        reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
        loc_loss += loss_x_bin + loss_z_bin

        if get_xz_fine:
            x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
            z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
            start_offset = z_res_r

            x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
            z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
            x_res_norm_label = x_res_label / loc_bin_size
            z_res_norm_label = z_res_label / loc_bin_size

            x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
            x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
            z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
            z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

            loss_x_res = F.smooth_l1_loss((pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim = 1), x_res_norm_label)
            loss_z_res = F.smooth_l1_loss((pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim = 1), z_res_norm_label)
            reg_loss_dict['loss_x_res'] = loss_x_res.item()
            reg_loss_dict['loss_z_res'] = loss_z_res.item()
            loc_loss += loss_x_res + loss_z_res

        # y localization loss
        if get_y_by_bin:
            y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
            y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
            start_offset = y_res_r

            y_shift = torch.clamp(y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 1e-3)
            y_bin_label = (y_shift / loc_y_bin_size).floor().long()
            y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
            y_res_norm_label = y_res_label / loc_y_bin_size

            y_bin_onehot = torch.cuda.FloatTensor(y_bin_label.size(0), loc_y_bin_num).zero_()
            y_bin_onehot.scatter_(1, y_bin_label.view(-1, 1).long(), 1)

            loss_y_bin = F.cross_entropy(pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
            loss_y_res = F.smooth_l1_loss((pred_reg[:, y_res_l: y_res_r] * y_bin_onehot).sum(dim = 1), y_res_norm_label)

            reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
            reg_loss_dict['loss_y_res'] = loss_y_res.item()

            loc_loss += loss_y_bin + loss_y_res
        else:
            y_offset_l, y_offset_r = start_offset, start_offset + 1
            start_offset = y_offset_r

            loss_y_offset = F.smooth_l1_loss(pred_reg[:, y_offset_l: y_offset_r].sum(dim = 1), y_offset_label)
            reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
            loc_loss += loss_y_offset

        # angle loss
        ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
        ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

        ry_label = reg_label[:, 6]

        if get_ry_fine:
            # divide pi/2 into several bins (For RCNN, num_head_bin = 9)
            angle_per_class = (np.pi / 2) / num_head_bin

            ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

            shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min = 1e-3, max = np.pi * 0.5 - 1e-3)  # (0, pi/2)

            # bin center is (5, 10, 15, ..., 85)
            ry_bin_label = (shift_angle / angle_per_class).floor().long()
            ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
            ry_res_norm_label = ry_res_label / (angle_per_class / 2)

        else:
            # divide 2pi into several bins (For RPN, num_head_bin = 12)
            angle_per_class = (2 * np.pi) / num_head_bin
            heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

            shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
            ry_bin_label = (shift_angle / angle_per_class).floor().long()
            ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
            ry_res_norm_label = ry_res_label / (angle_per_class / 2)

        ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
        ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
        loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
        loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim = 1), ry_res_norm_label)

        reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
        reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
        angle_loss = loss_ry_bin + loss_ry_res

        # size loss
        size_res_l, size_res_r = ry_res_r, ry_res_r + 3
        assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1], size_res_r)

        size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
        size_res_norm = pred_reg[:, size_res_l:size_res_r]
        size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)

        pred_x = (pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim = 1) * loc_bin_size
        pred_y = pred_reg[:, y_offset_l: y_offset_r].sum(dim = 1)
        pred_z = (pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim = 1) * loc_bin_size
        pred_size = size_res_norm * anchor_size + anchor_size  # hwl(yzx)

        tar_x, tar_y, tar_z = x_res_label, y_offset_label, z_res_label
        tar_size = reg_label[:, 3:6]

        insect_x = torch.max(torch.min((pred_x + pred_size[:, 2] / 2), (tar_x + tar_size[:, 2] / 2)) - torch.max(
                (pred_x - pred_size[:, 2] / 2), (tar_x - tar_size[:, 2] / 2)),
                             pred_x.new().resize_(pred_x.shape).fill_(1e-3))
        insect_y = torch.max(torch.min((pred_y + pred_size[:, 0] / 2), (tar_y + tar_size[:, 0] / 2)) - torch.max(
                (pred_y - pred_size[:, 0] / 2), (tar_y - tar_size[:, 0] / 2)),
                             pred_x.new().resize_(pred_x.shape).fill_(1e-3))
        insect_z = torch.max(torch.min((pred_z + pred_size[:, 1] / 2), (tar_z + tar_size[:, 1] / 2)) - torch.max(
                (pred_z - pred_size[:, 1] / 2), (tar_z - tar_size[:, 1] / 2)),
                             pred_x.new().resize_(pred_x.shape).fill_(1e-3))


        if cfg.MODEL.POINT_HEAD.IOU_LOSS_TYPE == 'raw':
            # print('USE RAW LOSS')
            #
            insect_area = insect_x * insect_y * insect_z
            pred_area = torch.max(pred_size[:, 0] * pred_size[:, 1] * pred_size[:, 2],
                                  pred_size.new().resize_(pred_size[:, 2].shape).fill_(1e-3))
            tar_area = tar_size[:, 0] * tar_size[:, 1] * tar_size[:, 2]
            iou_tmp = insect_area / (pred_area + tar_area - insect_area)

            if use_iou_branch:
                iou_branch_pred_flat = iou_branch_pred.view(-1)
                iou_branch_pred_flat = torch.clamp(iou_branch_pred_flat, 0.0001, 0.9999)
                iou_tmp_taget = torch.clamp(iou_tmp, 0.0001, 0.9999)
                iou_branch_loss = -(iou_tmp_taget.detach() * torch.log(iou_branch_pred_flat) + (
                            1 - iou_tmp_taget.detach()) * torch.log(1 - iou_branch_pred_flat))
                reg_loss_dict['iou_branch_loss'] = iou_branch_loss.mean()

            if use_cls_score:
                iou_tmp = cls_score * iou_tmp

            if use_mask_score:
                # print('mask_score:', mask_score)
                # iou_tmp = mask_score * iou_tmp
                iou_tmp = iou_tmp
            iou_tmp = torch.max(iou_tmp, iou_tmp.new().resize_(iou_tmp.shape).fill_(1e-4))
            iou_loss = -torch.log(iou_tmp)
            iou_loss = iou_loss.mean()

        elif cfg.MODEL.POINT_HEAD.IOU_LOSS_TYPE == 'cls_mask_with_bin':
            #print('cfg.TRAIN.IOU_LOSS_TYPE')
            pred_x_bin = F.softmax(pred_reg[:, x_bin_l: x_bin_r], 1) # N x num_bin
            pred_z_bin = F.softmax(pred_reg[:, z_bin_l: z_bin_r], 1)

            #
            xz_bin_ind = torch.arange(per_loc_bin_num).float()
            xz_bin_center = xz_bin_ind * loc_bin_size + loc_bin_size / 2 - loc_scope # num_bin
            xz_bin_center = xz_bin_center.to(pred_x_bin.device)

            #
            pred_x_reg = pred_reg[:, x_res_l: x_res_r] * loc_bin_size # N x num_bin
            pred_z_reg = pred_reg[:, z_res_l: z_res_r] * loc_bin_size

            #
            pred_x_abs = xz_bin_center + pred_x_reg
            pred_z_abs = xz_bin_center + pred_z_reg

            pred_x = (pred_x_abs * pred_x_bin).sum(dim=1)
            pred_z = (pred_z_abs * pred_z_bin).sum(dim=1)
            pred_y = pred_reg[:, y_offset_l: y_offset_r].sum(dim=1) # N

            pred_size = size_res_norm * anchor_size + anchor_size # hwl(yzx)

            #
            tar_x, tar_y, tar_z = x_res_label, y_offset_label, z_res_label
            #
            tar_x = xz_bin_center[x_bin_label] + tar_x
            tar_z = xz_bin_center[z_bin_label] + tar_z

            tar_size = reg_label[:, 3:6]

            insect_x = torch.max(torch.min((pred_x + pred_size[:, 2]/2), (tar_x + tar_size[:, 2]/2)) - torch.max((pred_x - pred_size[:, 2]/2), (tar_x - tar_size[:, 2]/2)), pred_x.new().resize_(pred_x.shape).fill_(1e-3))
            insect_y = torch.max(torch.min((pred_y + pred_size[:, 0]/2), (tar_y + tar_size[:, 0]/2)) - torch.max((pred_y - pred_size[:, 0]/2), (tar_y - tar_size[:, 0]/2)), pred_x.new().resize_(pred_x.shape).fill_(1e-3))
            insect_z = torch.max(torch.min((pred_z + pred_size[:, 1]/2), (tar_z + tar_size[:, 1]/2)) - torch.max((pred_z - pred_size[:, 1]/2), (tar_z - tar_size[:, 1]/2)), pred_x.new().resize_(pred_x.shape).fill_(1e-3))

            insect_area = insect_x * insect_y * insect_z
            pred_area = torch.max(pred_size[:, 0] * pred_size[:, 1] * pred_size[:, 2], pred_size.new().resize_(pred_size[:, 2].shape).fill_(1e-3))
            tar_area = tar_size[:, 0] * tar_size[:, 1] * tar_size[:, 2]
            iou_tmp = insect_area/(pred_area+tar_area-insect_area)

            if use_iou_branch:
                iou_branch_pred_flat = iou_branch_pred.view(-1)
                iou_branch_pred_flat = torch.clamp(iou_branch_pred_flat, 0.0001, 0.9999)
                iou_tmp_taget = torch.clamp(iou_tmp, 0.0001, 0.9999)
                iou_branch_loss = -(iou_tmp_taget.detach() * torch.log(iou_branch_pred_flat) + (
                            1 - iou_tmp_taget.detach()) * torch.log(1 - iou_branch_pred_flat))
                reg_loss_dict['iou_branch_loss'] = iou_branch_loss.mean()

            if use_cls_score:
                iou_tmp = cls_score * iou_tmp

            if use_mask_score:
                # print('mask_score:', mask_score)
                # iou_tmp = mask_score * iou_tmp
                iou_tmp = iou_tmp
            iou_tmp = torch.max(iou_tmp, iou_tmp.new().resize_(iou_tmp.shape).fill_(1e-4))
            iou_loss = -torch.log(iou_tmp)

            iou_loss = iou_loss.mean()

        # Total regression loss
        reg_loss_dict['loss_loc'] = loc_loss
        reg_loss_dict['loss_angle'] = angle_loss
        reg_loss_dict['loss_size'] = size_loss
        reg_loss_dict['loss_iou'] = iou_loss


        return loc_loss, angle_loss, size_loss, iou_loss, reg_loss_dict