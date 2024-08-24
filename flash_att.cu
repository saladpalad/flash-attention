#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <curand.h>
#include <iostream>
#include <cmath>

#define b_r 32
#define b_c 32 

__global__ void forward_kernel(float* Q, float *K, float *V, float* O, float* l, float* m, int M, int N, int d, const int t_r, const int t_c, const float dot_prod_scale){

	int q_block_size = b_r * d;
	int kv_block_size = b_c * d;

	int tid = threadIdx.x;
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float sram[];
	
	// allocate SRAM  partitions
	float* q_i = sram;
	float* k_j = &sram[q_block_size];
	float* v = &sram[q_block_size + kv_block_size];
	float* o_i = &sram[q_block_size + 2*kv_block_size];
	float* l_i = &sram[q_block_size + 2*kv_block_size + q_block_size];
	float* m_i = &sram[q_block_size + 2*kv_block_size + q_block_size + b_r];
	float* S = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r];

	float tid_max[b_r];
	float tid_sum[b_r];
	float P[b_r * b_c];
	float m_new[b_r];
	float l_new[b_r];

    if (tid < b_r) {
        tid_max[tid] = -INFINITY;
        tid_sum[tid] = 0;
		m_new[tid] = 0;
		l_new[tid] = 0;
    }
    
    for (int i = tid; i < b_r * b_c; i += blockDim.x * gridDim.x) {
        P[i] = 0;
    }
	
	for(int i = 0; i < t_c; i++){
		// Load K_j, V_j blocks from HBM to SRAM
		for(int local_j = tid; local_j < kv_block_size; local_j += blockDim.x){
			int index = i * kv_block_size +	local_j;	
			if(index < N*d){
				k_j[local_j] = K[index];
				v[local_j] = V[index];	
			}	
		}	
		for(int j = 0; j < t_r; j++){
			// Load Q_i, O_i from HBM to SRAM
			for(int local_k = tid; local_k < q_block_size; local_k += blockDim.x){
				int index = j * q_block_size + local_k;	
				if(index < N*d){
					q_i[local_k] = Q[index];
					o_i[local_k] = O[index];
				}	
			}
			// Load l_i, m_i from HBM to SRAM	
			for(int local_k = tid; local_k < b_r; local_k += blockDim.x){
				int index = j * q_block_size + local_k;	
				if(index < N){
					l_i[local_k] = l[index];
					m_i[local_k] = m[index];
				}	
			}
			__syncthreads();

			// S_ij = Q_i*K_j^T, m_ij = tidmax(S_ij)
			if (tid < b_r) {
				float local_max = -INFINITY;
				for(int c = 0; c < b_c; c++){
					float dot_prod = 0;
					for(int k = 0; k < d; k++){
						dot_prod += q_i[tid*d + k] * k_j[c*d + k];
					}
					dot_prod *= dot_prod_scale;
					S[tid*b_c + c] = dot_prod;
					local_max = max(dot_prod, local_max);
				}
				tid_max[tid] = local_max;
			}
			
			// P_ij = exp(S_ij - tid_max), l_ij = tidsum(P_ij)
			if(tid < b_r){
				float local_sum = 0;
				for(int c = 0; c < b_c; c++){
					int idx = tid*b_c + c;
					float S_ij = S[idx];
					float m_ij = tid_max[tid];
					float exp_val = __expf(S_ij - m_ij);

					P[idx] = exp_val;
					local_sum += exp_val;
				}
				tid_sum[tid] = local_sum;
			}
			
			// compute m_new, l_new
			if(tid < b_r){
				float m_i_prev = m_i[tid];
				float m_ij = tid_max[tid];
				float m_i_new = max(m_i_prev, m_ij);
				m_new[tid] = m_i_new;

				float l_i_prev = l_i[tid];
				float l_ij = tid_sum[tid];
				float l_i_new = __expf(m_i_prev - m_i_new) * l_i_prev + __expf(m_ij - m_i_new) * l_ij;
				l_new[tid] = l_i_new;
			}
			
			// Write O, l, m to HBM
			if(tid < b_r) {
				int global_tid = j * b_r + tid;
				if(global_tid < N) {
					// Update l and m in global memory
					l[global_tid] = l_new[tid];
					m[global_tid] = m_new[tid];

					// Update O in global memory
					for(int k = 0; k < d; k++) {
						float new_O_i = 0;
						for(int c = 0; c < b_c; c++) {
						int global_col = i * b_c + c;
							if(global_col < N) {
								float v_j = v[c * d + k];
								float P_ij = P[tid * b_c + c];
								new_O_i += P_ij * v_j;
							}
						}
						O[global_tid * d + k] = (1.0f / l_new[tid]) * (l_i[tid] * __expf(m_i[tid] - m_new[tid]) * o_i[tid * d + k] + __expf(tid_max[tid] - m_new[tid]) * new_O_i);
					}
				}
			}
		}
	}
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, torch::Tensor l, torch::Tensor m, int M, int N, int d) {
    const int t_r = (N + b_r - 1) / b_r; // ceil(N/b_r)
    const int t_c = (N + b_c - 1) / b_c; // ceil(N/b_c)

    const float dot_prod_scale = 1/sqrt(d);

    dim3 grid_size(1); // usually batch_size * num_head 
    dim3 block_size(b_r);  

    // Calculate shared memory size
    int q_block_size = b_r * d;
    int kv_block_size = b_c * d;
    int s_block_size = b_r * b_c;
    const int shared_mem_size = (q_block_size + 2*kv_block_size + q_block_size + 2*b_r + s_block_size) * sizeof(float);

    // Check shared memory size
    //int max_sram_size;
    //cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    //printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size, shared_mem_size);

    // Launch kernel
    forward_kernel<<<grid_size, block_size, shared_mem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        M, N, d, t_r, t_c, dot_prod_scale
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Flash Attention forward");
}
