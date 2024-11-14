import torch
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load

flash_attention = load(name='flash_attention', sources=['flash_att.cu'], extra_cuda_cflags=['-O3'])

def time_cuda_function(func, *args, num_warmup=0, num_runs=1):
    # Warmup
    for _ in range(num_warmup):
        _ = func(*args)
    
    # Timing
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        result = func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_runs
    return result, elapsed_time


def naive_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0/math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def main():
    B = 16
    H = 12
    N = 64
    d = 32 
    Q = torch.randn((B, H, N, d), device='cuda')
    K = torch.randn((B, H, N, d), device='cuda')
    V = torch.randn((B, H, N, d), device='cuda')
    O = torch.zeros((B, H, N, d), device='cuda')
    l = torch.zeros((B, H, N), device='cuda')
    m = torch.full((B, H, N), -float('inf'), device='cuda')
    
    print('Starting attention computations...')
    
    # Time naive attention
    naive_result, naive_time = time_cuda_function(naive_attn, Q, K, V)
    print(f"Naive attention average time: {naive_time:.4f} ms")
    
    # Time flash attention
    flash_result, flash_time = time_cuda_function(flash_attention.forward, Q, K, V, O, l, m, B, H, N, d)
    print(f"Flash attention average time: {flash_time:.4f} ms")
    
    # Calculate speedup
    speedup = naive_time / flash_time
    print(f"Speedup: {speedup:.2f}x")
    
    print('Attention values sanity check:', torch.allclose(flash_result, naive_result, rtol=0, atol=1))

if __name__ == "__main__":
    main()
