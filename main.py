import torch
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import os

flash_attention = load(name='flash_attention', sources=['flash_att.cu'], extra_cuda_cflags=['-O2'])

def plot_and_save_attention_diff(flash_result, naive_result, filename='attention_diff.png'):
    diff = (flash_result - naive_result).abs()
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(diff.cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar(im)
    plt.title('Absolute Difference between Flash and Naive Attention')
    
    # Save the plot as a PNG file
    current_dir = os.getcwd()
    filepath = os.path.join(current_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    
    print(f"Plot saved as {filepath}")

def naive_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0/math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def main():
    M = 64
    N = 64
    d = 12
    Q = torch.randn((N,d), device='cuda')
    K = torch.randn((N,d), device='cuda')
    V = torch.randn((N,d), device='cuda')
    O = torch.zeros((N,d), device='cuda')
    l = torch.zeros((N,1), device='cuda')
    m = torch.full((N,1), -float('inf'), device='cuda')
    print('hi')
    naive_result = naive_attn(Q, K, V)
    flash_result = flash_attention.forward(Q, K, V, O, l, m, M, N, d)
    plot_and_save_attention_diff(flash_result, naive_result)
    print("First few values of flash: ", flash_result[:])
    print("First few values of naive: ", naive_result[:])
    print('attn values sanity check:', torch.allclose(flash_result, naive_result, rtol=0, atol=1))

if __name__ == "__main__":
    main()
