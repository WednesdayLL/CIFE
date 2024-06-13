import torch
from torch import nn
from math import sqrt

####################   transformer   ########################
class MultiHeadSelfAttention(nn.Module):
    """
    MultiHeadSelfAttention module
    """
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in,dim_k,dim_v, num_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, self.dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, self.dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, self.dim_v, bias=False)
        self._norm_fact = 1 / sqrt(self.dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads  # 2
        dk = self.dim_k // nh  # dim_k of each head 1
        dv = self.dim_v // nh  # dim_v of each head 1

        q = self.linear_q(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk) 5.reshape(16,5,2)
        k = self.linear_k(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        v = self.linear_v(x.reshape(batch, dim_in)).reshape(batch, nh, dv)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, self.dim_v)  # batch, n, dim_v

        return att

class Transformer(nn.Module):
    """
    Transformer module
    """
    def __init__(self, n_input):
        super(Transformer, self).__init__()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        hidden_size1 = n_input[0]
        hidden_size2 = n_input[1]
        # hidden_size3 = n_input[2]
        # hidden_size4 = n_input[3]
        self.proj1 = nn.Linear(hidden_size1, hidden_size1)
        self.proj2 = nn.Linear(hidden_size1, hidden_size2)
        # attention + feedforword
        self.attn1 = MultiHeadSelfAttention(dim_in=hidden_size1,dim_k=4, dim_v=hidden_size1)
        self.attn2 = MultiHeadSelfAttention(dim_in=hidden_size2, dim_k=4, dim_v=hidden_size2)


    def encoder(self, x):
        y = self.proj1(x)
        y = self.gelu(y)
        x = x+y
        y = self.attn1(x)
        y = self.gelu(y)
        x = x+y
        y = self.proj2(x)
        x = self.gelu(y)
        # y = self.attn2(x)
        # x = self.gelu(y)
        # x = x + y


        return x


    def forward(self, x):
        z = self.encoder(x)
        return z