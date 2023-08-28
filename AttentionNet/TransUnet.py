import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetBlock(nn.Module):
    def __init__(self, inChannel, outChannel) -> None:
        # 使用卷积核大小3，步长1，填充1，保持大小不变，从而让神经网络绝对对称,可以吧下降和上升写在一起
        # 在transunet中只有一层哦
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, 3, padding=1, stride=1),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads) -> None:
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
    def __init__(self, image_size,patch_size,in_channel,embed_dim) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_channel,embed_dim,kernel_size=patch_size,stride=patch_size)
        # self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        # x = self.norm(x)
        return x

class TransUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransUnet, self).__init__()
        ch = [16, 64, 128, 256, 512]
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)

        self.down1 = UnetBlock(in_channels, ch[0])
        self.down2 = UnetBlock(ch[0], ch[1])
        self.down3 = UnetBlock(ch[1], ch[2])
        self.down4 = UnetBlock(ch[2], ch[3])



        self.up1 = UnetBlock(512 * 2, 256)
        self.up2 = UnetBlock(256 * 2, 128)
        self.up3 = UnetBlock(128 * 2, 64)
        self.up4 = UnetBlock(64 * 2, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)  # 最后的输出层

    def forward(self, x):
        # 假设图片大小为512*512
        x1 = self.down1(x)  # 16*H*W
        x2 = self.down2(self.maxpool(x1))  # 64*H/2*W/2
        x3 = self.down3(self.maxpool(x2))  # 128*H/4*W/4
        x4 = self.down4(self.maxpool(x3))  # 256*64*64
        # 然后吧图片分为4*4个小块，每个小块为16*16*256
        """
            这里暂时跳过了多头注意力机制
            首先把x4的输出转变为多头注意力的输入格式:(H/8*W/8)*256
        """
        out = 1

        up1 = self.up1(torch.cat([x4, self.up(out)], dim=1))
        up2 = self.up2(torch.cat([x3, self.up(up1)], dim=1))
        up3 = self.up3(torch.cat([x2, self.up(up2)], dim=1))
        up4 = self.up4(torch.cat([x1, self.up(up3)], dim=1))
        out = self.final(up4)
        return self.softmax(out)


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = TransUnet(3, 1).to(device)
