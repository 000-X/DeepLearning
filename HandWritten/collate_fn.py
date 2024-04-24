import torch


def collate_fn(batch):
    images, labels_cls, labels_loc = zip(*batch)

    # 将图像张量堆叠到一起形成一个批次
    images = torch.stack(images, dim=0)

    # 直接将多标签分类向量堆叠到一起，因为它们已经被初始化为固定长度
    labels_cls = torch.stack(labels_cls, dim=0)

    labels_loc = torch.stack(labels_loc, dim=0)
    # print(f"cls and loc shape --> {labels_cls.shape, labels_loc .shape}")
    return images, {'labels': labels_cls, 'boxes': labels_loc}
