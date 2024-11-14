#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <curand.h>
#include <iostream>
#include <cmath>

__global__ void forward_kernel(const float* Q, const float* K, const float* V, float* O, float* l, float* m,
                    const int B, const int H, const int N, const int d, 
                    const int Tc, const int Tr, const int Bc, const int Br, const float innerprod_scale){

    int b = blockIdx.x;    // handles b index
    int h = blockIdx.y;    // handles head index
    int br = threadIdx.x; // thread handles br dimension
    int bc = threadIdx.y;   // thread handles bc dimension
    int bz = gridDim.x;    // batch_size
    int nh = gridDim.y;    // num_head
    
    // allocate SRAM partitions
    extern __shared__ float sram[];
    float* Qi = sram;
    float* Kj = &sram[Br*d];
    float* Vj = &sram[Br*d + Bc*d];
    float* Oi = &sram[Br*d + 2*Bc*d];
    float* li = &sram[2*Br*d + 2*Bc*d];
    float* mi = &sram[2*Br*d + 2*Bc*d + Br];
    float* Sij = &sram[2*Br*d + 2*Bc*d + 2*Br];


    // modify this code to include loops of d instead of bc;

    for(int j = 0; j < Tc; j++) {
        // Load Kj, Vj blocks from HBM to SRAM
        Kj[brc*d + k] = K[b*nh*N*d + h*N*d + j*Bc*d + brc*d + k];
        Vj[brc*d + k] = V[b*nh*N*d + h*N*d + j*Bc*d + brc*d + k];
        __syncthreads();
        
        for(int i = 0; i < Tr; i++) {
            // Load Qi, Oi, li, mi from HBM to SRAM
            Qi[brc*d + k] = Q[b*nh*N*d + h*N*d + i*Br*d + brc*d + k];
            Oi[brc*d + k] = O[b*nh*N*d + h*N*d + i*Br*d + brc*d + k];
            li[brc] = l[i*Br + brc];
            mi[brc] = m[i*Br + brc];
            __syncthreads();
            
            // Sij = QiKjT, mij = rowmax(Sij), mi_new = max(mi,mij)
            float mij = -INFINITY;
            float inner_prod = 0.0f;
            for(int jj = 0; jj < Bc; jj++) {
                inner_prod += Qi[brc*d + kk] * Kj[jj*d + kk];
            }
            Sij[br*Bc+bc] = inner_prod;
            mij = fmaxf(mij, Sij[br*Bc+bc]);


            // Pij = exp(Sij - mij), lij = rowsum(Pij)
            float lij = 0.0f;
            for(int jj = 0; jj < Bc; jj++) {
                float exp_val = __expf(Sij[brc*Bc + jj] - mij);
                Sij[brc*Bc + jj] = exp_val;
                lij += exp_val;
            }
            
            // compute mi_new, li_new
            float mi_new = fmaxf(mi[brc], mij);
            float li_new = __expf(mi[brc] - mi_new) * li[brc] + __expf(mij - mi_new) * lij;
            
            // Write O, l, m to HBM
            float PijVj = 0.0f;
            for(int jj = 0; jj < Bc; jj++) {
                float inner_prod = 0.0f;
                for(int kk = 0; kk < d; kk++) {
                    inner_prod += Sij[brc*Bc + jj] * Vj[jj*d + kk];
                }
                PijVj += inner_prod;
            }
            
            O[b*nh*N*d + h*N*d + i*Br*d + brc*d + k] = 
                (1.0f / li_new) * (li[brc] * __expf(mi[brc] - mi_new) * Oi[brc*d + k] + 
                                  __expf(mij - mi_new) * PijVj);
            
            if(k == 0) {
                l[i*Br + brc] = li_new;
                m[i*Br + brc] = mi_new;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor l, torch::Tensor m, 
        const int bz, const int nh, const int N, const int d) {

    const int Br = 32;
    const int Bc = 32;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    const float dot_prod_scale = 1/sqrt(d);

    dim3 grid_size(bz, nh);
    dim3 block_size(Br, d);

    // Calculate shared memory size
    const int shared_mem_size = (2*Br*d + 2*Bc*d + 2*Br + Br*Bc) * sizeof(float);

    // Check shared memory size
    //int max_sram_size;
    //cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    //printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size, shared_mem_size);

    forward_kernel<<<grid_size, block_size, shared_mem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        bz, nh, N, d, Tc, Tr, Bc, Br, dot_prod_scale
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention forward");
}
