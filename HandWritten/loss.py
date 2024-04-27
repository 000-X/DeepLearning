import math

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


# 自定义Label Smoothing Cross Entropy Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')

    def forward(self, preds, labels):
        """
                前向传播计算损失
                参数:
                    preds (Tensor): 预测输出，形状为 [batch_size, seq_len, num_classes], logits未经softmax
                    labels (Tensor): 真实标签，形状为 [batch_size, seq_len]，每个值为类别索引
                """
        preds = preds.to(self.device)
        labels = labels.to(self.device)
        batch_size, seq_len, num_classes = preds.size()

        # 创建平滑的标签
        # true_dist的每一项都是smoothing / (num_classes - 1)
        true_dist = torch.full_like(preds, fill_value=self.smoothing / (num_classes - 1))
        # 将真实标签的对应位置设置为1-smoothing
        # 使用scatter_来更新true_dist中对应标签的置信度
        true_dist.scatter_(2, labels.unsqueeze(2), 1.0 - self.smoothing)

        # 使用log_softmax计算log概率
        log_probs = f.log_softmax(preds, dim=2)

        # 计算标签平滑交叉熵损失
        loss = -torch.sum(log_probs * true_dist, dim=2).mean()  # 计算每个时间步的损失，然后求均值

        return loss


# acc性能评估
def bbox_iou(box1, box2):
    """
    计算两个边界框之间的IoU。
    box1, box2: [batch_size, seq_len, 4]，边界框定义为 [x1, y1, x2, y2]
    """
    # 计算交集的坐标
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    # 计算交集面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算每个边界框的面积
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
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
    ious = bbox_iou(preds, labels)
    # 计算IoU大于阈值的比例
    correct_localizations = (ious > iou_threshold).float()  # 将bool转为float
    accuracy = correct_localizations.mean()
    return accuracy.item() * 100  # 返回百分比


# 分类ACC
def classification_accuracy(preds, labels):
    """
    计算分类准确率，假设preds是logits（未经Softmax处理）
    参数:
    - preds: 模型的输出logits，形状 [batch_size, seq_len, num_classes]
    - labels: 真实的分类标签，形状 [batch_size, seq_len]
    """
    # 应用Softmax函数将logits转换为概率分布
    probs = f.softmax(preds, dim=-1)
    # 获得每个序列位置上最可能的分类
    predicted_classes = probs.argmax(dim=2)
    # 比较预测结果与真实标签
    correct_predictions = (predicted_classes == labels).float()  # 转换为浮点数以便计算平均值
    # 计算整个批次的平均准确率
    accuracy = correct_predictions.mean()
    return accuracy.item() * 100  # 返回百分比
