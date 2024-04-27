import json

import torch

from HandWritten import loss

charset_path = 'Generate/txt/Characters.txt'


def read_characters(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            characters = [line.strip() for line in file if line.strip() != '']
        return characters
    except IOError as e:
        print(f"Error reading {file_path}: {e}")
        return []


# 定义函数，用于输出数据内容至文档
def print_val(labels_pre, bbox_pre, cls_real, bbox_real, flag):
    # shape [batch_size * seq_len, num_classes], [batch_size, seq_len, 4]
    out_pre = f'output/pre_json_{flag + 1}.json'
    out_real = f'output/real_json_{flag + 1}.json'
    char_set = read_characters(charset_path)
    idx_to_char = {idx: char for idx, char in enumerate(char_set)}

    # Process predictions and real data
    pre, real = [], []
    for i, (cls_pred, loc_pred, cls_gt, loc_gt) in enumerate(zip(labels_pre, bbox_pre, cls_real, bbox_real)):
        # print((cls_pred, loc_pred, cls_gt, loc_gt))
        pred_chars, gt_chars = [], []
        for idx, (cls_p, loc_p, cls_g, loc_g) in enumerate(zip(cls_pred, loc_pred, cls_gt, loc_gt)):
            # print((cls_p, loc_p, cls_g, loc_g))
            char_pred = idx_to_char[torch.argmax(cls_p).item()]
            char_gt = idx_to_char[cls_g.cpu().item()]
            pred_chars.append({'char': char_pred, 'bbox': loc_p.tolist()})
            gt_chars.append({'char': char_gt, 'bbox': loc_g.tolist()})
        pre.append(pred_chars)
        real.append(gt_chars)

    # Write to JSON
    out_to_json(out_pre, out_real, pre, real)


# 解码值输出至 sample.txt, 用于比对数据
def out_to_json(path1, path2, pre, real):
    try:
        with open(path1, 'w', encoding='utf-8') as f:
            json.dump(pre, f, indent=4)
        with open(path2, 'w', encoding='utf-8') as f:
            json.dump(real, f, indent=4)
    except IOError as e:
        print(f"Error writing to JSON: {e}")


class HandwritingOCRTrainer:
    def __init__(self, model, device='CPU'):
        self.device = device
        self.model = model.to(device)
        # self.criterion_classification = nn.CrossEntropyLoss()
        # self.criterion_localization = nn.SmoothL1Loss()
        self.cls_loss = loss.LabelSmoothingCrossEntropy(smoothing=0.1)
        self.loc_loss = loss.CIoULoss()
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
        total_acc = 0
        num_batches = 0
        for images, targets in dataloader:
            self.optimizer.zero_grad()

            images = images.to(self.device)
            labels = targets['labels'].to(self.device)  # [batch_size, seq_len, num_classes]
            boxes = targets['boxes'].to(self.device)
            classifications, localizations = self.model(images)
            # print(f"classification shape --> {classifications.shape}")
            # print(f"localization shape --> {localizations.shape}")

            loss_cls = self.cls_loss(classifications, labels)
            loss_loc = self.loc_loss(localizations.view(-1, 4), boxes.view(-1, 4))
            total_loss = (loss_loc * 1.0) + (loss_cls * 1.0)
            # print(f"Train loss_cls: {loss_cls}, Train loss_loc: {loss_loc}")
            total_loss.backward()
            self.optimizer.step()

            cls_acc = loss.classification_accuracy(classifications, labels)
            loc_acc = loss.localization_accuracy(localizations, boxes)
            # print(f"Training cls_acc: {cls_acc}; loc_acc: {loc_acc}")

            total_acc += cls_acc + loc_acc
            total_loss += total_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Training Loss: {avg_loss}; Training AVG_ACC: {avg_acc}")
        return avg_loss, avg_acc

    def validate(self, dataloader, fa=0, flag=0):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        flag = 0
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                labels = targets['labels'].to(self.device)  # [batch_size, seq_len, num_classes]
                boxes = targets['boxes'].to(self.device)
                classifications, localizations = self.model(images)

                loss_cls = self.cls_loss(classifications, labels)
                loss_loc = self.loc_loss(localizations.view(-1, 4), boxes.view(-1, 4))
                total_loss += (loss_loc * 0.5) + (loss_cls * 0.5)
                print(f"val loss_cls: {loss_cls}, val loss_loc: {loss_loc}")

                cls_acc = loss.classification_accuracy(classifications, labels)
                loc_acc = loss.localization_accuracy(localizations, boxes)
                print(f"Training cls_acc: {cls_acc}; loc_acc: {loc_acc}")

                total_acc += cls_acc + loc_acc
                num_batches += 1
                if flag // 10 == 0:
                    print_val(classifications, localizations, labels, boxes, flag)
                else:
                    flag += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Training Loss: {avg_loss}; Training AVG_ACC: {avg_acc}")
        return avg_loss, avg_acc

    def train(self, train_dataloader, val_dataloader, epochs, early_stopping_threshold=10):
        best_val_loss = float('inf')
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}:")
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.validate(val_dataloader)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {val_loss}")
            print(f"Epoch {epoch + 1}: Training ACC = {train_acc}, Validation ACC = {val_acc}")
            
        print("Training completed.")
