from torch.utils.data import random_split, DataLoader
from torchvision import transforms

import OCRDataset
import collate_fn


class DatasetManager:
    def __init__(self, txt_path, char_set, split_ratio, batch_size):
        self.txt_path = txt_path
        self.char_set = char_set
        self.split_ratio = split_ratio
        self.batch_size = batch_size

        # 数据增强操作
        self.augmented_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomVerticalFlip(),
            transforms.GaussianBlur(3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 基础预处理操作
        self.basic_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_datasets(self):
        # 加载整个数据集，应用基础预处理
        full_dataset = OCRDataset.HandwritingOCRDataset(
            font_path=self.txt_path,
            char_set=self.char_set,
            transform=self.basic_transforms)

        # 根据split_ratio计算训练集和验证集的大小
        train_size = int(len(full_dataset) * self.split_ratio)
        val_size = len(full_dataset) - train_size

        # 划分数据集
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # 修改训练集的transform，不直接修改原始dataset的属性
        train_dataset.dataset = OCRDataset.HandwritingOCRDataset(
            font_path=self.txt_path,
            char_set=self.char_set,
            transform=self.augmented_transforms)

        return train_dataset, val_dataset

    def get_loaders(self, shuffle_train=True):
        train_dataset, val_dataset = self.create_datasets()

        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_fn.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn.collate_fn,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader
