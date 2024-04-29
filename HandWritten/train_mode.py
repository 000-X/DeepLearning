import json

import torch

import CoderC
import loss


# 定义函数，用于输出数据内容至文档
def print_val(labels_pre, bbox_pre, cls_real, bbox_real, flag):
    # shape [batch_size, seq_len, num_classes], [batch_size, seq_len, 4]
    out_pre = f'output/pre_json_{flag + 1}.json'
    out_real = f'output/real_json_{flag + 1}.json'
    coderc = CoderC.CoderC()

    # Process predictions and real data
    pre, real = [], []
    for i, (cls_pred, loc_pred, cls_gt, loc_gt) in enumerate(zip(labels_pre, bbox_pre, cls_real, bbox_real)):
        # print((cls_pred, loc_pred, cls_gt, loc_gt))
        pred_chars, gt_chars = [], []
        for idx, (cls_p, loc_p, cls_g, loc_g) in enumerate(zip(cls_pred, loc_pred, cls_gt, loc_gt)):
            # print(cls_p)
            # print(torch.argmax(cls_p).item())
            # breakpoint()
            char_pred = coderc.get_char(torch.argmax(cls_p).item())
            char_gt = coderc.get_char(cls_g.item())
            print(f"char_pred --> {char_pred}; char_gt --> {char_gt}")
            pred_chars.append({'char': char_pred, 'bbox': loc_p.tolist()})
            gt_chars.append({'char': char_gt, 'bbox': loc_g.tolist()})
        pre.append(pred_chars)
        real.append(gt_chars)

        # breakpoint()
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
    def __init__(self, model, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        # self.criterion_classification = nn.CrossEntropyLoss()
        # self.criterion_localization = nn.SmoothL1Loss()
        # self.cls_loss = nn.CrossEntropyLoss()
        # self.loc_loss = nn.MSELoss()
        self.loss = loss.CombinedLoss()
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
            # print(images.shape)
            labels = targets['labels'].to(self.device)  # [batch_size, seq_len, num_classes]
            boxes = targets['boxes'].to(self.device)
            classifications, localizations = self.model(images)
            # print(f"classification shape --> {classifications.shape}")
            # print(f"localization shape --> {localizations.shape}")

            loss1 = self.loss(classifications, localizations, labels, boxes)
            loss1.sum().backward()
            self.optimizer.step()
            # print(f"Train loss_cls: {loss_cls}, Train loss_loc: {loss_loc}")

            cls_acc = loss.classification_accuracy(classifications, labels)
            loc_acc = loss.localization_accuracy(localizations, boxes)
            # print(f"Training cls_acc: {cls_acc}; loc_acc: {loc_acc}")

            total_acc += cls_acc + loc_acc
            total_loss += loss1.sum()
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
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                labels = targets['labels'].to(self.device)  # [batch_size, seq_len, num_classes]
                boxes = targets['boxes'].to(self.device)
                classifications, localizations = self.model(images)

                loss1 = self.loss(classifications, localizations, labels, boxes)
                total_loss += loss1.sum()
                # print(f"val loss {loss1}")
                # breakpoint()

                cls_acc = loss.classification_accuracy(classifications, labels)
                loc_acc = loss.localization_accuracy(localizations, boxes)
                # print(f"Training cls_acc: {cls_acc}; loc_acc: {loc_acc}")

                total_acc += cls_acc + loc_acc
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Val Loss: {avg_loss}; Val AVG_ACC: {avg_acc}")
        print(f"Val cls_acc: {cls_acc}; Val loc_acc: {loc_acc}")
        print_val(classifications, localizations, labels, boxes, flag=int(avg_acc))
        return avg_loss, avg_acc

    def train(self, train_dataloader, val_dataloader, epochs):
        # 初始化提前停止器
        early_stopper = loss.EarlyStopping(
            patience=10,
            verbose=True,
            delta=0.001,
            path=r'H:\pro_yzy\HandWritten\PTH\checkpoint.pt',
        )
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}:")
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.validate(val_dataloader)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {val_loss}")
            print(f"Epoch {epoch + 1}: Training ACC = {train_acc}, Validation ACC = {val_acc}")
            early_stopper(val_loss, model=self.model)
            if early_stopper.early_stop:
                print("提前停止训练")
                break
        print("Training completed.")
