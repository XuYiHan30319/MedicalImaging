import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image


class Data(Dataset):
    def __init__(self):
        rotate = transforms.RandomRotation(90)
        self.x = []
        self.y = []

        pad_size = 92

        # 遍历指定文件夹中的所有文件
        for filename in os.listdir("Data/membrane/train/image"):
            file_path = os.path.join("Data/membrane/train/image", filename)
            # 判断是否是文件以及是否是图片格式
            if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                image = Image.open(file_path)
                image = transforms.ToTensor()(image)
                image2 = rotate(image)
                self.x.append(image)
                self.x.append(image2)

        for filename in os.listdir("Data/membrane/train/label"):
            file_path = os.path.join("Data/membrane/train/label", filename)
            # 判断是否是文件以及是否是图片格式
            if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                image = Image.open(file_path)
                image = transforms.ToTensor()(image)
                image2 = rotate(image)
                self.y.append(image)
                self.y.append(image2)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.__len__()


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )

    def forward(self, x1, x2, x3, x4, x5):
        x1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.net(x1)


class UNet3P(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()
        ch = [64, 128, 256, 512, 1024]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        print(input_channel, ch[0])
        self.down1 = DownBlock(input_channel, ch[0])
        self.down2 = DownBlock(ch[0], ch[1])
        self.down3 = DownBlock(ch[1], ch[2])
        self.down4 = DownBlock(ch[2], ch[3])
        self.down5 = DownBlock(ch[3], ch[4])

        self.up1 = UpBlock(320, 320)
        self.up2 = UpBlock(320, 320)
        self.up3 = UpBlock(320, 320)
        self.up4 = UpBlock(320, 320)

        # 这里再给出20个网络来进行操作
        self.x1tox1 = nn.Conv2d(
            in_channels=ch[0], out_channels=64, kernel_size=3, padding=1
        )
        self.x1tox2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=ch[0], out_channels=64, kernel_size=3, padding=1),
        )
        self.x1tox3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels=ch[0], out_channels=64, kernel_size=3, padding=1),
        )
        self.x1tox4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(in_channels=ch[0], out_channels=64, kernel_size=3, padding=1),
        )

        self.x2tox1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
        )
        self.x2tox2 = nn.Conv2d(
            in_channels=ch[1], out_channels=64, kernel_size=3, padding=1
        )
        self.x2tox3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=ch[1], out_channels=64, kernel_size=3, padding=1),
        )
        self.x2tox4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels=ch[1], out_channels=64, kernel_size=3, padding=1),
        )

        self.x3tox1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
        )
        self.x3tox2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
        )
        self.x3tox3 = nn.Conv2d(
            in_channels=ch[2], out_channels=64, kernel_size=3, padding=1
        )
        self.x3tox4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=ch[2], out_channels=64, kernel_size=3, padding=1),
        )

        self.x4tox1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
        )
        self.x4tox2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
        )
        self.x4tox3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
        )
        self.x4tox4 = nn.Conv2d(
            in_channels=ch[3], out_channels=64, kernel_size=3, padding=1
        )

        self.x5tox1 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=ch[4], out_channels=64, kernel_size=3, padding=1),
        )
        self.x5tox2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=ch[4], out_channels=64, kernel_size=3, padding=1),
        )
        self.x5tox3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=ch[4], out_channels=64, kernel_size=3, padding=1),
        )
        self.x5tox4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=ch[4], out_channels=64, kernel_size=3, padding=1),
        )

        # UNet3 + 深监督部分
        # UNet3 + 全尺寸深监督是每个解码器对应一个侧输出（side
        # output），通过ground
        # truth进行监督。为了实现深度监控，每个解码器的最后一层被送入一个普通的3 × 3
        # 卷积层，然后是一个双线性上采样和一个sigmoid函数。
        self.final5 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=output_channel, kernel_size=3, padding=1
            ),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )
        self.final4 = nn.Sequential(
            nn.Conv2d(
                in_channels=320, out_channels=output_channel, kernel_size=3, padding=1
            ),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(
                in_channels=320, out_channels=output_channel, kernel_size=3, padding=1
            ),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(
                in_channels=320, out_channels=output_channel, kernel_size=3, padding=1
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )
        self.final1 = nn.Sequential(
            nn.Conv2d(
                in_channels=320, out_channels=output_channel, kernel_size=3, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 下降过程
        xe1 = self.down1(x)
        xe2 = self.down2(self.maxpool(xe1))
        xe3 = self.down3(self.maxpool(xe2))
        xe4 = self.down4(self.maxpool(xe3))
        xd5 = self.down5(self.maxpool(xe4))
        # 上升过程
        xd4 = self.up1(
            self.x5tox4(xd5),
            self.x4tox4(xe4),
            self.x3tox4(xe3),
            self.x2tox4(xe2),
            self.x1tox4(xe1),
        )
        xd3 = self.up2(
            self.x5tox3(xd5),
            self.x4tox3(xd4),
            self.x3tox3(xe3),
            self.x2tox3(xe2),
            self.x1tox3(xe1),
        )
        xd2 = self.up3(
            self.x5tox2(xd5),
            self.x4tox2(xd4),
            self.x3tox2(xd3),
            self.x2tox2(xe2),
            self.x1tox2(xe1),
        )
        xd1 = self.up4(
            self.x5tox1(xd5),
            self.x4tox1(xd4),
            self.x3tox1(xd3),
            self.x2tox1(xd2),
            self.x1tox1(xe1),
        )
        return [
            self.final1(xd1),
            self.final2(xd2),
            self.final3(xd3),
            self.final4(xd4),
            self.final5(xd5),
        ]


def train(net, data_loader, loss_fn, optimizer, num_epochs, device):
    print("Start Training...")
    for epoch in range(num_epochs):
        net.train()  # 设置网络为训练模式
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # 梯度清零
            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = 0
            for output in outputs:
                loss += loss_fn(output, targets)
            # 反向传播
            loss /= len(outputs)
            loss.backward()
            # 参数更新
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {average_loss:.4f}")


if __name__ == "__main__":
    data = Data()
    train_loader = DataLoader(data, batch_size=4, shuffle=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = UNet3P(input_channel=1, output_channel=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    train(model, train_loader, loss_fn, optimizer, 10, device)
    torch.save(model.state_dict(), "UNet3+.pth")
