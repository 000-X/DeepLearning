import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

import OCRDataset
import Simp_model
import collate_fn
import train_mode


def train(model, dataset, num_splits=5, epochs=10, early_stopping_threshold=5, BATCH_SIZE=10, transform=None):
    kf = KFold(n_splits=num_splits, shuffle=True)
    res = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold + 1}/{num_splits}")
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn.collate_fn,
            num_workers=2
        )
        val_dataloader = DataLoader(
            dataset=val_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn.collate_fn,
            num_workers=2
        )
        if transform:
            train_dataloader.dataset.transforms = transform

        best_val_loss = float('inf')
        no_improve_epoch = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}:")
            train_loss = model.train_epoch(train_dataloader)
            val_loss = model.validate(val_dataloader)
            model.scheduler.step(val_loss)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.model.state_dict()
                no_improve_epoch = 0
                # Optionally reload the best model weights
                model.model.load_state_dict(best_model)
            else:
                no_improve_epoch += 1

            if no_improve_epoch >= early_stopping_threshold:
                print(f"best_val_loss = {best_val_loss}")
                print("Early stopping due to no improvement in validation loss.")
                break
        test_loss = model.validate(val_dataloader, fa=1, flag=fold)
        print(f"test_loss --> {test_loss}")
        res.append(test_loss)
    return res


# basic_transform
basic_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# train_transform
augmented_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

if __name__ == '__main__':
    # 读取字符集
    char_set = OCRDataset.read_characters('Generate/txt/Characters.txt')
    # print(len(char_set))
    # breakpoint()
    # 加载数据集
    dataset = OCRDataset.HandwritingOCRDataset(
        font_path='dataset/font_path.txt',
        char_set=char_set,
        transform=basic_transforms
    )

    print(dataset.__len__())
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")
    model = Simp_model.CustomOCRModel(num_classes=len(char_set), device=device).to(device)
    train_model = train_mode.HandwritingOCRTrainer(model=model, device=device)

    res = train(
        model=train_model,
        dataset=dataset,
        num_splits=5,  # > 2
        epochs=50,
        BATCH_SIZE=20,
        transform=augmented_transforms
    )
    total_loss = sum(res) / 5
    print(f"Total_val_loss --> {total_loss}")
    # 模型保存示例（在训练或验证后）
    save_path = 'PTH/save_model.pth'
    torch.save(train_model.model.state_dict(), save_path)
    print("Over!")
