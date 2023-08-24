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
        transform = transforms.Compose(
            [transforms.Resize((388, 388)), transforms.ToTensor()]
        )
        self.x = []
        self.y = []

        pad_size = 92

        # 遍历指定文件夹中的所有文件
        for filename in os.listdir("unet/Data/membrane/train/image"):
            file_path = os.path.join("unet/Data/membrane/train/image", filename)
            # 判断是否是文件以及是否是图片格式
            if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                image = Image.open(file_path)
                image = transform(image)

                # Apply mirror padding
                image = mirror_pad(image, pad_size)

                self.x.append(image)

        for filename in os.listdir("unet/Data/membrane/train/label"):
            file_path = os.path.join("unet/Data/membrane/train/label", filename)
            # 判断是否是文件以及是否是图片格式
            if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                image = Image.open(file_path)
                image = transform(image)
                self.y.append(image)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.__len__()


class DownBlock(nn.Module):
    def __init__(self, inChannel, outChannel) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, 3, padding=0),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, 3, padding=0),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)

    def forward(self, x):
        x = self.net(x)
        return x, self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, inChannel, outChannel) -> None:
        super().__init__()
        self.ct = nn.ConvTranspose2d(inChannel, outChannel, 2, stride=2)
        
        self.net = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, 3, padding=0),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, 3, padding=0),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )

    def forward(self, x, pre):
        x = self.ct(x)
        #裁剪
        size = x.size()[2]
        preSize = pre.size()[2]
        start_row  = (preSize-size)//2
        end_row = start_row + size
        cropped_tensor = pre[:,:,start_row:end_row, start_row:end_row]    
        x = torch.cat([x, cropped_tensor], dim=1)
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.down1 = DownBlock(1, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=0),
            nn.ReLU(),
        )
        
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.final = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1)

    def forward(self, x):
        pre1,x = self.down1(x)
        pre2,x = self.down2(x)
        pre3,x = self.down3(x)
        pre4,x = self.down4(x)
        x = self.center(x)
        print(x.size())
        x = self.up1(x, pre4)
        x = self.up2(x, pre3)
        x = self.up3(x, pre2)
        x = self.up4(x, pre1)

        x = self.final(x)
        return x

def train(net,data_loader,loss_fn,optimizer,num_epochs,device):
    print("Start Training...")
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        net.train()  # 设置网络为训练模式
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs,targets = inputs.to(device),targets.to(device)
            optimizer.zero_grad()  # 梯度清零
            # 前向传播
            outputs = net(inputs)
            # targets = targets[:,0,:,:]
            # outputs = outputs[:, 0, :, :]            
            # 计算损失
            loss = loss_fn(outputs, targets)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(data_loader)}] Loss: {loss.item():.4f}")
        average_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {average_loss:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else "cpu")
device = torch.device("cpu")
Data = Data()
Data = DataLoader(Data, batch_size=2, shuffle=True)
Net = UNet().to(device)
optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
train(Net,Data,loss_fn,optimizer,10,device)
Net._save_to_state_dict('1.pth')