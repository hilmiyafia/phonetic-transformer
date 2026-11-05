import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.split = (in_dim, 1)
    self.weight = nn.Parameter(torch.empty(out_dim, sum(self.split)))
    nn.init.xavier_normal_(self.weight.data)

  def forward(self, x):
    w, b = self.weight.split(self.split, 1)
    return F.linear(x, w, b.flatten())

class Attention(nn.Module):

  def __init__(self, dim, head, hidden_dim):
    super().__init__()
    self.split = (head, 1, 1)
    self.expand = Linear(dim, sum(self.split) * hidden_dim)
    self.unflatten = nn.Unflatten(-1, (sum(self.split), hidden_dim))
    self.project = Linear(head * hidden_dim, dim)

  def forward(self, x):
    y = x * x.square().mean(-1, True).rsqrt()
    y = self.expand(y)
    y = self.unflatten(y)
    y = y.permute(0, 2, 1, 3)
    q, k, v = y.split(self.split, 1)
    y = F.scaled_dot_product_attention(
      query=q, 
      key=k, 
      value=v, 
      is_causal=True, 
      enable_gqa=True)
    y = y.permute(0, 2, 1, 3)
    y = y.flatten(2)
    y = self.project(y)
    return y + x

class Block(nn.Module):

  def __init__(self, dim, head, hidden_dim):
    super().__init__()
    self.attention = Attention(dim, head, hidden_dim)
    self.linear = Linear(dim, 2 * dim)
    self.act = nn.SiLU()

  def forward(self, x):
    y = self.attention(x)
    z = y * y.square().mean(-1, True).rsqrt()
    z = self.linear(z).chunk(2, -1)
    z = self.act(z[0]) * z[1]
    return z + y

class Transformer(nn.Module):

  def __init__(self, token_count, depth, head, dim, hidden_dim):
    super().__init__()
    embedding = torch.pow(100, -torch.arange(dim // 2) / (dim // 2))
    self.embedding = nn.Buffer(embedding[None, None, :])
    self.project = nn.Parameter(torch.randn(token_count, dim))
    self.layers = nn.Sequential(
      *[Block(dim, head, hidden_dim) for _ in range(depth)])

  def forward(self, x):
    t = torch.arange(x.shape[1]).to(x.device)[None, :, None] * self.embedding
    y = self.project[x] + torch.stack((t.cos(), t.sin()), -1).flatten(2, 3)
    y = self.layers(y)
    y = y * y.square().mean(-1, True).rsqrt()
    y = y @ self.project.mT
    return y

  def sample(self, tokens, end_token=None, max_iteration=30):
    with torch.no_grad():
      for _ in range(max_iteration):
        x = torch.as_tensor(tokens).to(self.embedding.device)[None]
        y = self.forward(x)[0, -1]
        tokens.append(y.argmax(-1))
        if tokens[-1] == end_token:
          break
    return tokens
