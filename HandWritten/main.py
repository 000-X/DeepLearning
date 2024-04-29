"""
    训练模型
"""
import torch

import CoderC
import DataManage
import Simp_model
import train_mode

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    coderc = CoderC.CoderC()

    # 加载数据集
    data_manage = DataManage.DatasetManager(
        txt_path=f'dataset/font_path.txt',
        split_ratio=0.8,
        batch_size=20
    )
    train_data, val_data = data_manage.get_loaders()
    # for img, t in train_data:
    #     print(img.shape)
    #     break

    # 加载模型
    model = Simp_model.CustomOCRModel(coderc.get_set_len()).to(device)
    train = train_mode.HandwritingOCRTrainer(model, coderc.get_set_len())

    # 开始训练
    train.train(
        train_dataloader=train_data,
        val_dataloader=val_data,
        epochs=1000,
    )
    save_path = 'PTH/save_model.pth'
    torch.save(model.state_dict(), save_path)
    print('Over.')
