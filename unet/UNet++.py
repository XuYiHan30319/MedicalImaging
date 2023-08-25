import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image


def mirror_pad(image, pad_size):
    return F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode="reflect")


class Data(Dataset):
    def __init__(self):
        # transform = transforms.Compose(
        #     [transforms.Resize((388, 388)), transforms.ToTensor()]
        # )
        rotate = transforms.RandomRotation(90)
        self.x = []
        self.y = []

        pad_size = 94

        # 遍历指定文件夹中的所有文件
        for filename in os.listdir("Data/membrane/train/image"):
            file_path = os.path.join("Data/membrane/train/image", filename)
            # 判断是否是文件以及是否是图片格式
            if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                image = Image.open(file_path)
                # image = transform(image)

                # Apply mirror padding
                # image = mirror_pad(image, pad_size)
                image = transforms.ToTensor()(image)
                image2 = rotate(image)
                self.x.append(image)
                self.x.append(image2)

        for filename in os.listdir("Data/membrane/train/label"):
            file_path = os.path.join("Data/membrane/train/label", filename)
            # 判断是否是文件以及是否是图片格式
            if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                image = Image.open(file_path)
                # image = transform(image)
                image = transforms.ToTensor()(image)
                image2 = rotate(image)
                self.y.append(image)
                self.y.append(image2)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.__len__()


class UnetBlock(nn.Module):
    def __init__(self, inChannel, outChannel) -> None:
        #使用卷积核大小3，步长1，填充1，保持大小不变，从而让神经网络绝对对称
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, 3, padding=1,stride=1),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, 3, padding=1,stride=1),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x



class UNetPP(nn.Module):
    def __init__(self, inChannel,outChannel) -> None:
        super().__init__()
        ch = [32, 64, 128, 256, 512]
        # 一个池化层
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        # 两层加起来要求层数不变而大小相同
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)


        self.x0_0 = UnetBlock(inChannel, ch[0])
        self.x1_0 = UnetBlock(ch[0], ch[1])
        self.x2_0 = UnetBlock(ch[1], ch[2])
        self.x3_0 = UnetBlock(ch[2], ch[3])
        self.x4_0 = UnetBlock(ch[3], ch[4])
        self.x0_1 = UnetBlock(ch[0] + ch[1], ch[0])
        self.x1_1 = UnetBlock(ch[1] + ch[2], ch[1])
        self.x2_1 = UnetBlock(ch[2] + ch[3], ch[2])
        self.x3_1 = UnetBlock(ch[3] + ch[4], ch[3])

        self.x0_2 = UnetBlock(ch[0] + ch[1] + ch[0], ch[0])
        self.x1_2 = UnetBlock(ch[1] + ch[2] + ch[1], ch[1])
        self.x2_2 = UnetBlock(ch[2] + ch[3] + ch[2], ch[2])

        self.x0_3 = UnetBlock(ch[0] + ch[1] + ch[0] + ch[0], ch[0])
        self.x1_3 = UnetBlock(ch[1] + ch[2] + ch[1] + ch[1], ch[1])

        self.x0_4 = UnetBlock(ch[0] + ch[1] + ch[0] + ch[0] + ch[0], ch[0])

        # 最后使用深度监督final层输出结果
        self.final1 = nn.Conv2d(ch[0], outChannel, 1)
        self.final2 = nn.Conv2d(ch[0], outChannel, 1)
        self.final3 = nn.Conv2d(ch[0], outChannel, 1)
        self.final4 = nn.Conv2d(ch[0], outChannel, 1)
    def forward(self, x):
        x0_0 = self.x0_0(x)
        x = self.pool(x0_0)
        x1_0 = self.x1_0(x)
        x = self.pool(x1_0)
        x2_0 = self.x2_0(x)
        x = self.pool(x2_0)
        x3_0 = self.x3_0(x)
        x = self.pool(x3_0)
        x4_0 = self.x4_0(x)
        # 第一轮下降结束
        # 计算后续结果
        x0_1 = self.x0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))#在深度上累加
        x1_1 = self.x1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.x3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.x1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        x0_4 = self.x0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        return [output1, output2, output3, output4]


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
            loss /= 4
            loss.backward()
            # 参数更新
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {average_loss:.4f}")


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available
    else "cpu"
)
# device = torch.device("cpu")
Data = Data()
image = Data[0][0]
Data = DataLoader(Data, batch_size=4, shuffle=True)
Net = UNetPP(inChannel=1,outChannel=1).to(device)
optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
train(Net, Data, loss_fn, optimizer, 100, device)
Net._save_to_state_dict("1.pth")