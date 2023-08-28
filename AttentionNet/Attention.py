import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,din,dout):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = din**-0.5
        self.q = nn.Linear(din,dout)
        self.k = nn.Linear(din,dout)
        self.v = nn.Linear(din,dout)

    def forward(self,x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = torch.matmul(q,k.transpose(-2,-1))*self.scale
        score = self.softmax(score)
        out = torch.matmul(score,v)
        return out

class MultHeadSelfAttention(nn.Module):
    def __init__(self,dim_in,d_model,n_head):
        super(MultHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_in = dim_in
        self.scale = (d_model//n_head)**-0.5

        self.q = nn.Linear(dim_in,d_model)
        self.k = nn.Linear(dim_in,d_model)
        self.v = nn.Linear(dim_in,d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        self.ndim = self.d_model//self.n_head
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)



net = Attention(5,5)
x = torch.randn(2,5,5)
out = net(x)
print(out)