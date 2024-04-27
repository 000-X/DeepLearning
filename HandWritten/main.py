"""
    训练模型
"""
import torch

from HandWritten import OCRDataset, DataManage, Simp_model, train_mode

if __name__ == '__main__':
    char_set = OCRDataset.read_characters('Generate/txt/Characters.txt')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    # 加载数据集
    data_manage = DataManage.DatasetManager(
        txt_path=f'dataset/font_path.txt',
        char_set=char_set,
        split_ratio=0.8,
        batch_size=20
    )
    train_data, val_data = data_manage.get_loaders()

    # 加载模型
    model = Simp_model.CustomOCRModel(len(char_set)).to(device)
    train = train_mode.HandwritingOCRTrainer(model, device)

    # 开始训练
    train.train(
        train_dataloader=train_data,
        val_dataloader=val_data,
        epochs=10000,
    )
    save_path = 'PTH/save_model.pth'
    torch.save(model.state_dict(), save_path)
    print('Over.')
