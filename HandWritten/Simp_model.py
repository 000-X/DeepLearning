import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CustomOCRModel(nn.Module):
    def __init__(self, num_classes, device='CPU'):
        super(CustomOCRModel, self).__init__()
        self.classes = num_classes
        self.device = torch.device(device)  # 确定设备
        self.cnn = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2])

        # 输出通道2048是通过ResNet50的倒数第三层得到的。
        # 该层目的：适配 RNN 层输入特征
        self.conv_adapter = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU()  # 在特征适配层后添加ReLU
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(512, 1)
        # classifier 输出维度为 len(char_set)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),  # 在分类层前添加ReLU
            nn.Linear(512, num_classes)
        )
        self.localizer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),  # 在定位层前添加ReLU
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(self.device)
        features = self.cnn(x)
        # print(f"features shape --> {features.shape}")
        features = self.conv_adapter(features)  # 获得的shape --> [batch_size, 256, 7, 7]
        # shape 设置为：[batch_size, 7*7, 256]
        # -1 表示特征图的空间维度[7, 7]
        features = features.view(features.size(0), -1, 256)  # Adjust shape for RNN
        rnn_out, _ = self.rnn(features)
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        attended_features = rnn_out * attention_weights.expand_as(rnn_out)
        classification = (self.classifier(attended_features)).view(-1, self.classes)
        # print(f"classification shape --> {classification.shape}")
        localization = self.localizer(attended_features)
        # print(f"localization shape --> {localization.shape}")
        return classification, localization
