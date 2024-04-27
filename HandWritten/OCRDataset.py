import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        characters = [line.strip() for line in file if line.strip() != '']
    return characters


def stand_labels_loc(image_annotations, feat_width, feat_height, image_width, image_height):
    loc = []
    flag = []
    seq_len = feat_width * feat_height
    scale_x = image_width / feat_width
    scale_y = image_height / feat_height
    for char_info in image_annotations['chars']:
        bbox = char_info['bbox']
        # 计算标准化后的边界框并映射到特征图单元
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2

        # 映射到特征图的单元索引
        feature_x = int(x_center / scale_x)
        feature_y = int(y_center / scale_y)

        # 确保索引在有效范围内
        feature_index = feature_y * 7 + feature_x
        if 0 <= feature_index < seq_len:
            # 归一化坐标
            norm_bbox = [
                round(bbox[0] / image_width, 5),
                round(bbox[1] / image_height, 5),
                round(bbox[2] / image_width, 5),
                round(bbox[3] / image_height, 5)
            ]
            loc.append((feature_index, torch.tensor(norm_bbox, dtype=torch.float)))
            flag.append(feature_index)

    # 填充序列化标签
    labels_loc = torch.zeros((seq_len, 4), dtype=torch.float)
    for index, tensor in loc:
        labels_loc[index] = tensor
    # print(flag)
    return labels_loc, flag


class HandwritingOCRDataset(Dataset):
    def __init__(self, font_path, char_set, transform=None):
        self.annotations_files = []
        self.images_dir = []
        self.char_set = char_set
        self.transform = transform or transforms.ToTensor()
        self.char_to_idx = {char: index for index, char in enumerate(char_set)}
        self.num_classes = len(char_set)

        with open(font_path, 'r', encoding='UTF-8') as f:
            for line in f:
                feature_dir = os.path.join(line.strip(), 'feature')
                labels_dir = os.path.join(line.strip(), 'labels')
                if os.path.exists(feature_dir) and os.path.exists(labels_dir):
                    for file_name in os.listdir(labels_dir):
                        if file_name.endswith('.json'):
                            self.annotations_files.append(os.path.join(labels_dir, file_name))
                            self.images_dir.append(feature_dir)

    def __len__(self):
        return len(self.annotations_files)

    def __getitem__(self, idx):
        annotation_path = self.annotations_files[idx]
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        image_path = os.path.join(self.images_dir[idx], annotation['image_name'])
        image = Image.open(image_path).convert('RGB')

        original_width, original_height = image.size
        image = self.transform(image)

        # 处理字符边界伸缩映射情况; flag是对应一维序列数组的索引
        labels_loc, flag = stand_labels_loc(
            image_annotations=annotation,
            feat_width=14,
            feat_height=14,
            image_width=original_width,
            image_height=original_height,
        )
        # 处理文本映射情况，基于序列化
        # 初始化文本标签序列为"空白"类别
        # 初始化文本标签序列
        labels_cls = torch.full((14 * 14,), self.char_to_idx['NULL'], dtype=torch.long)  # 假设None对应无字符的情况
        i = 0
        for char_info in annotation['chars']:
            char_idx = self.char_to_idx.get(char_info['text_char'], self.char_to_idx['NULL'])  # 使用默认索引值防止 None
            labels_cls[flag[i]] = char_idx  # 直接赋值，无需创建新的 tensor
            i += 1

        # print(f"image shape --> {image.shape}")
        # print(f"cls shape --> {labels_cls.shape}")    # 字符集的长度序列
        # print(f"cls --> {labels_cls}")    # 字符集的长度序列
        # print(f"loc shape --> {labels_loc.shape}")    # [feature, 4]
        # print(f"loc --> {labels_loc}")
        return image, labels_cls, labels_loc

# 示例代码，初始化数据集和数据加载器
# dataset = HandwritingOCRDataset(txt_path, char_set, transform=train_transform)
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
