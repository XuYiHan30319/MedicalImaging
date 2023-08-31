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


class ViTBlock(nn.Module):
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


class VIT(nn.Module):
    def __init__(
        self, inchannel, out_channels, imageSize=512, embedDim=256, patchSize=16
    ):
        super(VIT, self).__init__()
        self.embeding = PatchEmbedding(
            image_size=imageSize,
            patch_size=16,
            in_channel=inchannel,
            embed_dim=embedDim,
        )
        self.net = nn.Sequential(
            *[ViTBlock(input_dim=256, hidden_dim=512, num_heads=8) for _ in range(12)]
        )
        self.position = PositionalEncoding(
            embed_dim=embedDim,
        )

    def forward(self, x):
        batch, n, w, h = x.shape
        x = self.embeding(x)
        x = self.position(x)
        x = self.net(x)
        x = torch.reshape(x, (batch, n, w, h))
        return x


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

    Net = VIT(inchannel=1, out_channels=1).to(device)

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
