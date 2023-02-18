import torch
from torch import nn



multihead = nn.MultiheadAttention(embed_dim=200,
                                  num_heads=4,
                                  kdim=32,
                                  vdim=32,
                                  dropout=0.1,
                                  batch_first=True)

Q = torch.randn((2, 4, 200))
K = torch.randn((2, 18,32)) # n, ?, kdim
V = torch.randn((2, 18, 32)) # n, ?, vdim


x=multihead(Q,K,V)
print(x[0].shape)