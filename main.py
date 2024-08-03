import torch
import math
from torch.utils.cpp_extension import load

# Load the CUDA kernel
flash_attention_cuda = load(name="flash_attention",
                            sources=["flash_att.cu"],
                            verbose=True)

def main():
    # Set up dimensions
    M = 1024
    N = 512
    d_size = 64

    # Initialize Q, K, V using PyTorch
    Q = torch.randn((N,d_size), device='cuda', dtype=torch.float32)
    K = torch.randn((N,d_size), device='cuda', dtype=torch.float32)
    V = torch.randn((N,d_size), device='cuda', dtype=torch.float32)
    O = torch.zeros((N,d_size), device='cuda', dtype=torch.float32)
    l = torch.zeros((N,1), device='cuda', dtype=torch.float32)
    m = torch.full((N,1), -float('inf'), device='cuda', dtype=torch.float32)

    # Set up grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid = ((N + threads_per_block[0] - 1) // threads_per_block[0],
                       (N + threads_per_block[1] - 1) // threads_per_block[1])

    # Calculate shared memory size
    b_c = math.ceil(M / 4 * d_size)
    b_r = min(math.ceil(M / 4 * d_size), d_size)
    q_block_size = b_r * d_size
    kv_block_size = b_c * d_size
    s_block_size = b_r * b_c
    
    shared_mem_size = (q_block_size + 2*kv_block_size + q_block_size + 2*b_r + s_block_size + 4*b_r) * 4  # 4 bytes per float

    # Launch the kernel
    flash_attention_cuda.forward_kernel(
        grid=blocks_per_grid,
        block=threads_per_block,
        args=[Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), l.data_ptr(), m.data_ptr(), M, N, d_size],
        shared_mem=shared_mem_size
    )

    # If you need to use the results, you can access them through the PyTorch tensors
    # For example:
    # result = O.cpu().numpy()

if __name__ == "__main__":
    main()
