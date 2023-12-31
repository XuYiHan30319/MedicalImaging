import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image, ImageOps
import math


def mirror_pad(image, pad_size):
    return F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode="reflect")


class Data(Dataset):
    def __init__(self, train=True):
        # transform = transforms.Compose(
        #     [transforms.Resize((388, 388)), transforms.ToTensor()]
        # )
        rotate = transforms.RandomRotation(90)
        self.x = []
        self.y = []
        self.train = train
        pad_size = 94
        if train:
            # 遍历指定文件夹中的所有文件
            for filename in os.listdir("unet/Data/membrane/train/image"):
                file_path = os.path.join("unet/Data/membrane/train/image", filename)
                # 判断是否是文件以及是否是图片格式
                if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                    image = Image.open(file_path)
                    # image = transform(image)

                    # Apply mirror padding
                    # image = mirror_pad(image, pad_size)
                    image = transforms.ToTensor()(image)
                    # image2 = rotate(image)
                    self.x.append(image)
                    # self.x.append(image2)

            for filename in os.listdir("unet/Data/membrane/train/label"):
                file_path = os.path.join("unet/Data/membrane/train/label", filename)
                # 判断是否是文件以及是否是图片格式
                if os.path.isfile(file_path) and filename.lower().endswith((".png")):
                    image = Image.open(file_path)
                    # image = transform(image)
                    image = transforms.ToTensor()(image)
                    # image2 = rotate(image)
                    self.y.append(image)
                    # self.y.append(image2)
        else:
            for filename in os.listdir("unet/Data/membrane/test"):
                file_path = os.path.join("unet/Data/membrane/test", filename)
                # 判断是否是文件以及是否是图片格式
                if os.path.isfile(file_path) and not filename.lower().endswith(
                    ("predict.png")
                ):
                    image = Image.open(file_path)
                    image = transforms.ToTensor()(image)
                    self.x.append(image)
                    self.y.append(filename)
        print(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.__len__()


class UnetBlock(nn.Module):
    def __init__(self, inChannel, outChannel) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, 3, padding=1, stride=1),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, drop=0.2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.q = nn.Linear(input_dim, input_dim)
        self.k = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, input_dim)
        self.msa = nn.MultiheadAttention(input_dim, num_heads)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),  # 添加 Dropout 层
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        x1 = self.norm1(x)
        q = self.q(x1)
        k = self.k(x1)
        v = self.v(x1)
        x1, _ = self.msa(q, k, v)
        x1 = x1 + x

        x2 = self.norm2(x1)
        x2 = self.mlp(x2)
        x2 = x1 + x2
        return x2


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channel, embed_dim) -> None:
        # imagesize是图片大小,pathsize是每个patch的大小,embed_dim是每个patch的维度,维度的计算为inchanel*patchsize*patchsize
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channel, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )
        # self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        # 输入为 B,C,H,W
        # 经过卷积,输出为 B,embed_dim,H/patch_size,W/patch_size,
        # 然后flatten变成 B,embed_dim,H/patch_size*W/patch_size
        # 然后transpose变成 B,H/patch_size*W/size,embed_dim得到想要的结果
        x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_patch_num=2000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # Calculate the positional encodings in advance
        position = torch.arange(0, max_patch_num).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pos_encoding = torch.zeros(1, max_patch_num, embed_dim)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        # Input x: [batch_size, num_patches, embed_dim]
        pos_encoding = self.pos_encoding[:, : x.size(1), :]
        return x + pos_encoding


class TransUnet(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=512):
        super(TransUnet, self).__init__()
        ch = [16, 64, 128, 256, 512]
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax()
        self.position = PositionalEncoding(embed_dim)
        self.embed_dim = embed_dim

        self.down1 = UnetBlock(in_channels, ch[0])
        self.down2 = UnetBlock(ch[0], ch[1])
        self.down3 = UnetBlock(ch[1], ch[2])
        self.down4 = UnetBlock(ch[2], ch[3])
        self.patchEmbedding = PatchEmbedding(64, 2, 256, embed_dim)
        self.Transform = nn.Sequential(
            *[TransformerBlock(embed_dim, embed_dim * 3, 8) for _ in range(12)]
        )
        self.conv1 = nn.Conv2d(embed_dim, 512, kernel_size=3, stride=1, padding=1)

        self.ct1 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.up1 = UnetBlock(512, 256)
        self.ct2 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.up2 = UnetBlock(256, 128)
        self.ct3 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.up3 = UnetBlock(128, 64)
        self.ct4 = nn.ConvTranspose2d(
            64, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.up4 = UnetBlock(64, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)  # 最后的输出层

    def forward(self, x):
        # 假设图片大小为512*512
        batchsize, n, h, w = x.size()
        x1 = self.down1(x)  # 16*H*W
        x2 = self.down2(self.maxpool(x1))  # 64*H/2*W/2
        x3 = self.down3(self.maxpool(x2))  # 128*H/4*W/4
        x4 = self.down4(self.maxpool(x3))  # 256*64*64
        x5 = self.patchEmbedding(x4)  # 得到序列化的结果,大小为16*(16*256*16)
        x5 = self.position(x5)
        out = self.Transform(x5)
        out = (
            torch.reshape(out, (batchsize, h // 16, w // 16, self.embed_dim))
            .transpose(1, 3)
            .transpose(2, 3)
        )
        out = self.conv1(out)
        up1 = self.up1(torch.cat([x4, self.ct1(out)], dim=1))
        up2 = self.up2(torch.cat([x3, self.ct2(up1)], dim=1))
        up3 = self.up3(torch.cat([x2, self.ct3(up2)], dim=1))
        up4 = self.ct4(up3)
        out = self.final(up4)
        return out


def train(net, data_loader, loss_fn, optimizer, num_epochs, device):
    print("Start Training...")
    # loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        net.train()  # 设置网络为训练模式
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # 梯度清零
            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = loss_fn(outputs, targets)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {average_loss:.4f}")


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    Net = TransUnet(1, 1).to(device)

    i = 0
    if os.path.exists("TransNet+.pth"):
        Net.load_state_dict(torch.load("TransNet+.pth"))
        Net.eval()
        data = Data(False)
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for batch_idx, (inputs, name) in enumerate(dataloader):
            inputs = inputs.to(device)
            result = Net(inputs)
            # 将张量转换为图像
            out_np = result.squeeze().cpu().detach().numpy()
            # 将 NumPy 数组转换为 PIL 图像
            image = Image.fromarray(np.uint8(out_np))
            # 保存图像为文件
            print(name[0])
            output_image_path = name[0]
            i += 1
            inverted_image = ImageOps.invert(image)
            inverted_image.save(output_image_path)
    else:
        data = Data()
        dataloader = DataLoader(data, batch_size=5, shuffle=True)
        optimizer = torch.optim.Adam(Net.parameters(), lr=0.002)
        loss_fn = nn.BCEWithLogitsLoss()
        train(Net, dataloader, loss_fn, optimizer, 200, device)
        torch.save(Net.state_dict(), "TransNet+.pth")
