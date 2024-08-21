#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <curand.h>
#include <iostream>
#include <cmath>

__global__ void forward_kernel(float* Q, float *K, float *V, float* O, float* l, float* m, int M, int N, int d){
	
	// tile dimensions	
	int b_c = ceilf(M/4*d);
	int b_r = min(static_cast<int>(ceilf(M/4*d)), d);
	
	// num of tiles
	int t_r = ceilf(N/b_r);
	int t_c = ceilf(N/b_c);

	int q_block_size = b_r * d;
	int kv_block_size = b_c * d;
	int s_block_size = b_r * b_c;

	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float sram[];
	
	// allocate SRAM  partitions
	float* q_i = sram;
	float* k_j = &sram[q_block_size];
    float* v = &sram[q_block_size + kv_block_size];
	float* o_i = &sram[q_block_size + 2*kv_block_size];
	float* l_i = &sram[q_block_size + 2*kv_block_size + q_block_size];
	float* m_i = &sram[q_block_size + 2*kv_block_size + q_block_size + b_r];
	float* S = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r];

	float row_max[b_r];
	float row_sum[b_r];
	float P[b_r * b_c];
	float m_new[b_r];
	float l_new[b_r];

    if (row < b_r) {
        row_max[row] = -INFINITY;
        row_sum[row] = 0;
		m_new[row] = 0;
		l_new[row] = 0;
    }
    
    for (int i = tid; i < b_r * b_c; i += blockDim.x * gridDim.x) {
        P[i] = 0;
    }
	
	for(int i = 0; i < t_c; i++){
		
		// Load Kj, Vj blocks from HBM to SRAM
		for(int local_j = tid; local_j < kv_block_size; local_j += blockDim.x * blockDim.y){
			int index = i * kv_block_size +	local_j;	
			if(index < N*d){
				k_j[local_j] = K[index];
				v[local_j] = V[index];	
			}	
		}	


		for(int j = 0; j < t_r; j++){

			// Load Qi, Oi from HBM to SRAM
			for(int local_k = tid; local_k < q_block_size; local_k += blockDim.x * blockDim.y){
				int index = j * q_block_size + local_k;	
				if(index < N*d){
					q_i[local_k] = Q[index];
					o_i[local_k] = O[index];
				}	
			}

		
			// Load li, mi from HBM to SRAM	
			for(int local_k = tid; local_k < b_r; local_k += blockDim.x * blockDim.y){
				int index = j * q_block_size + local_k;	
				if(index < N){
					l_i[local_k] = l[index];
					m_i[local_k] = m[index];
				}	
			}
			__syncthreads();

			// S_ij = Q_i*K_j^T, m_ij = rowmax(S_ij)
			if (row < b_r) {
				float local_max = -INFINITY;
				for(int c = 0; c < b_c; c++){
					float dot_prod = 0;
					for(int k = 0; k < d; k++){
						dot_prod += q_i[row*d + k] * k_j[c*d + k];
					}
					S[row*b_c + c] = dot_prod;
					local_max = max(dot_prod, local_max);
				}
				row_max[row] = local_max;
			}
			
			// P_ij = exp(S_ij - row_max), l_ij = rowsum(P_ij)
			if(row < b_r){
				float local_sum = 0;
				for(int c = 0; c < b_c; c++){
					int idx = row*b_c + c;
					float S_ij = S[idx];
					float m_ij = row_max[row];
					float exp_val = __expf(S_ij - m_ij);

					P[idx] = exp_val;
					local_sum += exp_val;
				}
				row_sum[row] = local_sum;
			}
			
			// compute m_new, l_new
			if(row < b_r){
				float m_i_prev = m_i[row];
				float m_ij = row_max[row];
				float m_i_new = max(m_i_prev, m_ij);
				m_new[row] = m_i_new;

				float l_i_prev = l_i[row];
				float l_ij = row_sum[row];
				float l_i_new = __expf(m_i_prev - m_i_new) * l_i_prev + __expf(m_ij - m_i_new) * l_ij;
				l_new[row] = l_i_new;
			}
			
			// Write O_i, l_i, m_i to HBM
			if(row < b_r){
				float l_new_val = l_new[row];
				float m_i_prev = m_i[row];
				float m_ij = row_max[row];
				float m_new_val = m_new[row];
				float l_i_prev = l_i[row];
				float l_ij = row_sum[row];
				for(int c = 0; c < b_c; c++){  
					for(int k = 0; k < d; k++){
						float prev_O_i = o_i[row * d + k];							
						float v_j = v[c * d + k];  
						float P_ij = P[row * b_c + c];
						o_i[row * d + k] = (1.0f / l_new_val) * (l_i_prev * __expf(m_i_prev - m_new_val) * prev_O_i + __expf(m_ij - m_new_val) * l_ij * P_ij * v_j);
					}
				}	
					
				m_i[row] = m_new_val;
				l_i[row] = l_new_val;
		}
	}

}


