import json

import torch
import torch.nn as nn

charset_path = 'Generate/txt/Characters.txt'


def read_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        characters = [line.strip() for line in file if line.strip() != '']
    return characters


# 定义函数，用于输出数据内容至文档
def print_val(labels_pre, bbox_pre, cls_real, bbox_real, flag):
    # shape [batch_size * seq_len, num_classes], [batch_size, seq_len, 4]
    out_pre = f'output/pre_json_{flag + 1}.json'
    out_real = f'output/real_json_{flag + 1}.json'
    char_set = read_characters(charset_path)
    idx_to_char = {idx: char for idx, char in enumerate(char_set)}
    cls = labels_pre.view(bbox_pre.shape[0], bbox_pre.shape[1], len(char_set))
    pre = []
    real = []

    # 解码并输出预测值
    max_probs, indices = torch.max(cls, dim=2)  # 获取最大概率的索引
    chars = [[idx_to_char[idx.item()] for idx in indices[i]] for i in range(indices.shape[0])]
    # 将localization输出转换为列表格式
    bboxes = [bbox_pre[i].tolist() for i in range(bbox_pre.shape[0])]
    for i in range(len(chars)):
        image_results = []
        for j in range(len(chars[i])):
            image_results.append({
                'char': chars[i][j],
                'bbox': bboxes[i][j]
            })
        pre.append(image_results)

    # 解码输出真实值
    # [bat_size, 49]
    # [bat_size 49, 4]
    cls = [cls_real[i].tolist() for i in range(cls_real.shape[0])]
    bboxes = [bbox_real[i].tolist() for i in range(bbox_real.shape[0])]
    for i in bboxes:  # 批次
        res = []
        for idx, j in enumerate(i):
            res.append({
                'char': idx_to_char[cls[idx]],
                'bbox': j
            })
        real.append(res)

    # 将解码值输出至文本 sample.txt
    out_to_json(out_pre, out_real, pre, real)


# 解码值输出至 sample.txt, 用于比对数据
def out_to_json(path1, path2, pre, real):
    with open(path1, 'w+', encoding='utf-8') as file:
        json.dump(pre, file, indent=4)  # 使用缩进提高可读性

    with open(path2, 'w+', encoding='utf-8') as file:
        json.dump(real, file, indent=4)  # 使用缩进提高可读性


class HandwritingOCRTrainer:
    def __init__(self, model, device='CPU'):
        self.device = device
        self.model = model
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_localization = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
            {'params': model.cnn.parameters(), 'lr': 1e-5},  # 预训练层
            {'params': model.rnn.parameters(), 'lr': 1e-3},  # 新添加层
            {'params': model.classifier.parameters(), 'lr': 1e-3},
            {'params': model.localizer.parameters(), 'lr': 1e-3}
        ], lr=1e-4)  # 默认学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for images, targets in dataloader:
            self.optimizer.zero_grad()

            labels = (targets['labels'].to(self.device)).view(-1)
            boxes = targets['boxes'].to(self.device)
            classifications, localizations = self.model(images)
            # print(f"classification shape --> {classifications.shape}")
            # print(f"localization shape --> {localizations.shape}")

            loss_classification = self.criterion_classification(classifications, labels)
            loss_localization = self.criterion_localization(localizations, boxes)
            total_loss = loss_classification + loss_localization

            total_loss.backward()
            self.optimizer.step()
            total_loss += total_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Training Loss: {avg_loss}")
        return avg_loss

    def validate(self, dataloader, fa=0, flag=0):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, targets in dataloader:
                labels = (targets['labels'].to(self.device)).view(-1)  # [batch_size * seq_len, num_classes]
                boxes = targets['boxes'].to(self.device)
                classifications, localizations = self.model(images)

                loss_classification = self.criterion_classification(classifications, labels)
                loss_localization = self.criterion_localization(localizations, boxes)
                total_loss += loss_classification.item() + loss_localization.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss}")
        # 打印验证信息
        if fa != 0:
            print_val(classifications, localizations, labels, boxes, flag)
        return avg_loss

    def train(self, train_dataloader, val_dataloader, epochs, early_stopping_threshold=5):
        best_val_loss = float('inf')
        no_improve_epoch = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}:")
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.validate(val_dataloader)
            self.scheduler.step(val_loss)
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {val_loss}")
            print(f"Current learning rate: {current_lr}")
            print(f"Best_val_loss: {best_val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1

            if no_improve_epoch >= early_stopping_threshold:
                print("Early stopping due to no improvement in validation loss.")
                break
