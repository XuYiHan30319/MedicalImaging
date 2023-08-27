import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,input,output) -> None:
        super().__init__()
        # 保证大小不变
        self.cov1 = nn.Conv2d(in_channels=input,out_channels=output,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU()
        
        self.cov2 = nn.Conv2d(in_channels=output,out_channels=output,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(output)
        
        
    def forward(self, x):
        out = self.cov1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.cov2(out)
        out = self.bn2(out)
        
        out = out + x
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
        
    def forward(self, x):
        pass


