import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


# 定义CIOU损失函数
class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')

    def forward(self, preds, targets):
        # Extract coordinates
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        b1_x1, b1_y1, b1_x2, b1_y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

        # Calculate intersection over union
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1,
                                                                                     min=0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        # Calculate CIoU specifics
        center_distance = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        c_w = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        c_h = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c_diagonal_distance = c_w ** 2 + c_h ** 2 + 1e-6
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + 1e-6)) - torch.atan(
            (b1_x2 - b1_x1) / (b1_y2 - b1_y1 + 1e-6))), 2)
        alpha = v / (1 - iou + v)

        ciou = iou - (center_distance / c_diagonal_distance + v * alpha)
        return (1 - ciou).mean()


class SmoothL1LossMasked(nn.Module):
    def __init__(self):
        super(SmoothL1LossMasked, self).__init__()

    def forward(self, preds, targets, mask):
        """
        preds: [batch_size, seq_len, 4] 预测的边界框
        targets: [batch_size, seq_len, 4] 真实的边界框
        mask: [batch_size, seq_len] 一个布尔张量，True 表示有效的数据点
        """
        # 应用掩码
        # 扩展 mask 以匹配 preds 和 targets 的最后一维
        mask_expanded = mask.unsqueeze(-1).expand_as(preds)

        # 使用掩码过滤 preds 和 targets
        preds_masked = torch.where(mask_expanded, preds, torch.tensor(0.0).to(preds.device))
        targets_masked = torch.where(mask_expanded, targets, torch.tensor(0.0).to(targets.device))

        # 计算平滑 L1 损失
        loss = f.smooth_l1_loss(preds_masked, targets_masked, reduction='none')

        # 计算掩码位置的平均损失
        # 注意确保分母不为零
        masked_elements = mask_expanded.sum(dim=[0, 1])  # 计算有效损失元素的总数
        loss = loss.sum(dim=[0, 1]) / masked_elements.clamp(min=1)  # 防止除以零

        return loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, mask):
        """
        logits: [batch_size, seq_len, num_classes]
        targets: [batch_size, seq_len]
        mask: [batch_size, seq_len], Boolean tensor where `True` indicates a valid data point
        """
        # 扁平化数据
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        # 计算损失
        losses = self.loss_fn(logits_flat, targets_flat)
        # 应用掩码
        masked_losses = losses * mask_flat.type_as(losses)  # 确保掩码与损失具有相同的数据类型
        # 计算平均损失
        # 防止除以零，确保至少有一个有效的元素
        loss = masked_losses.sum() / mask_flat.float().sum().clamp(min=1.0)

        return loss


# 组合loss
class CombinedLoss(nn.Module):
    def __init__(self, cls_weight=0.5, loc_weight=0.5):
        super(CombinedLoss, self).__init__()
        # self.cls_loss = nn.CrossEntropyLoss()
        # self.loc_loss = nn.SmoothL1Loss()
        self.cls_loss = MaskedCrossEntropyLoss()
        self.loc_loss = SmoothL1LossMasked()
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, cls_preds, loc_preds, cls_targets, loc_targets):
        #  cls_preds shape [bat, seq_len, num_classes]
        #  loc_preds shape [bat, seq_len]
        cls_mask, loc_mask = create_masks(cls_targets, loc_targets)
        cls_loss = self.cls_loss(cls_preds, cls_targets, cls_mask)
        loc_loss = self.loc_loss(loc_preds, loc_targets, loc_mask)
        loss = cls_loss * self.cls_weight + loc_loss * self.loc_weight
        return loss


# acc性能评估
def bbox_iou(box1, box2):
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def localization_accuracy(preds, labels, iou_threshold=0.5):
    """
    计算定位的准确率，基于IoU阈值
    参数:
    - preds: 预测的边界框，形状 [batch_size, seq_len, 4]
    - labels: 真实的边界框，形状 [batch_size, seq_len, 4]
    - iou_threshold: 认为定位正确的IoU阈值
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, mask = create_masks(preds, labels)
    ious = bbox_iou(preds.to(device), labels.to(device))
    # 计算IoU大于阈值的比例
    valid_ious = ious[mask]
    correct_localizations = (valid_ious > iou_threshold).float()
    accuracy = correct_localizations.mean() if correct_localizations.numel() > 0 else torch.tensor(0.0)

    return accuracy * 100  # 返回百分比


# 分类ACC
def classification_accuracy(preds, labels, mask=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds, labels = preds.to(device), labels.to(device)
    if mask is None:
        mask = labels != 0

    probs = f.softmax(preds, dim=-1)
    predicted_classes = probs.argmax(dim=2)
    correct_predictions = (predicted_classes == labels).float() * mask.float()
    total_valid = mask.sum().float()
    accuracy = correct_predictions.sum() / total_valid

    return accuracy * 100


def create_masks(batch_labels_cls, batch_labels_loc):
    """
    Create masking arrays for classification and localization tasks based on label data.

    Args:
    batch_labels_cls (torch.Tensor): The classification labels tensor.
    batch_labels_loc (torch.Tensor): The localization labels tensor.

    Returns:
    tuple: A tuple containing two tensors (mask_cls, mask_loc) where:
        mask_cls (torch.Tensor): Mask for classification tasks. True for valid data points.
        mask_loc (torch.Tensor): Mask for localization tasks. True for rows with any non-zero values.
    """
    # Ensure inputs are tensors
    if not isinstance(batch_labels_cls, torch.Tensor):
        batch_labels_cls = torch.tensor(batch_labels_cls)
    if not isinstance(batch_labels_loc, torch.Tensor):
        batch_labels_loc = torch.tensor(batch_labels_loc)

    # Classification mask: True for non-zero values indicating valid data
    mask_cls = batch_labels_cls != 0

    # Localization mask: True in rows where any element is non-zero
    mask_loc = torch.any(batch_labels_loc != 0, dim=-1)

    return mask_cls, mask_loc


class EarlyStopping:
    """提前停止以防止模型过拟合"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 监控指标没有改进的轮数后停止训练.
            verbose (bool): 如果为True，则打印提前停止的消息
            delta (float): 最小改变量，改变量小于这个值不视为改进
            path (str): 模型最佳状态的保存路径
            trace_func (function): 用于输出信息的函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存模型当验证损失减少时"""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
