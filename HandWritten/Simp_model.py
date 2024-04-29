import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CustomOCRModel(nn.Module):
    def __init__(self, num_classes, drop_rate=0.5):
        super(CustomOCRModel, self).__init__()
        self.classes = num_classes
        self.drop_rate = drop_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 更新，采用ResNet50倒数第四层，增加特征图空间大小，feature.shape [14*14]
        self.cnn = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-3])
        # self.cnn = CNN_model.CNN_model().to(self.device)

        # 输出通道2048是通过ResNet50的倒数第三层得到的。
        # 该层目的：适配 RNN 层输入特征
        self.conv_adapter = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Tanh()  # 在特征适配层后添加ReLU
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(512, 1)
        # classifier 输出维度为 len(char_set)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),  # 在分类层前添加ReLU
            nn.Dropout(p=self.drop_rate),
            nn.Linear(512, num_classes)
        )
        self.localizer = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),  # 在定位层前添加ReLU
            nn.Dropout(p=self.drop_rate),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(self.device)
        features = self.cnn(x)
        features = self.conv_adapter(features)
        features = features.view(features.size(0), -1, 256)
        rnn_out, _ = self.rnn(features)
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        attended_features = rnn_out * attention_weights.expand_as(rnn_out)
        classification = self.classifier(attended_features)
        localization = self.localizer(attended_features)

        return classification, localization
