import torch
import flash_attention

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

    # Call the CUDA function
    O = flash_attention.forward(Q, K, V, O, l, m, M, N, d_size)
    print(O)
    # If you need to use the results, you can access them through the PyTorch tensors
    # For example:
    # result = O.cpu().numpy()

if __name__ == "__main__":
    main()
