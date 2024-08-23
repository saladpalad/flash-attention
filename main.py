import torch
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load

flash_attention = load(name='flash_attention', sources=['flash_att.cu'], extra_cuda_cflags=['-O2'])

def naive_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def main():
    M = 64
    N = 64
    d = 12
    Q = torch.randn((N,d), device='cuda', dtype=torch.float32)
    K = torch.randn((N,d), device='cuda', dtype=torch.float32)
    V = torch.randn((N,d), device='cuda', dtype=torch.float32)
    O = torch.zeros((N,d), device='cuda', dtype=torch.float32)
    l = torch.zeros((N,1), device='cuda', dtype=torch.float32)
    m = torch.full((N,1), -float('inf'), device='cuda', dtype=torch.float32)
    
    naive_result = naive_attn(Q, K, V)
    flash_result = flash_attention.forward(Q, K, V, O, l, m, M, N, d)
    print(flash_result)
    print('attn values sanity check:', torch.allclose(flash_result, naive_result, rtol=0, atol=1e-02))

if __name__ == "__main__":
    main()
