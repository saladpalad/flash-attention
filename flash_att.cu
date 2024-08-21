#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
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
	
	extern __shared__ float sram[];
	
	// allocate SRAM  partitions
	float* q_i = sram;
	float* k_j = &sram[q_block_size];
    float* v_j = &sram[q_block_size + kv_block_size];
	float* o_i = &sram[q_block_size + 2*kv_block_size];
	float* l_i = &sram[q_block_size + 2*kv_block_size + q_block_size];
	float* m_i = &sram[q_block_size + 2*kv_block_size + q_block_size + b_r];
	float* S = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r];

	float m_ij[b_r];
	float l_ij[b_r];
	float p_ij[b_r * b_c];
	float m_new[b_r];
	float l_new[b_r];

    if (tid < b_r) {
        m_ij[tid] = -INFINITY;
        l_ij[tid] = 0;
		m_new[tid] = 0;
		l_new[tid] = 0;
    }
    
    for (int i = tid; i < b_r * b_c; i += blockDim.x * gridDim.x) {
        p_ij[i] = 0;
    }
	
	for(int i = 0; i < t_c; i++){
		
		// Load Kj, Vj blocks from HBM to SRAM
		for(int local_j = tid; local_j < kv_block_size; local_j += blockDim.x * blockDim.y){
			int index = i * kv_block_size +	local_j;	
			if(index < N*d){
				k_j[local_j] = K[index];
				v_j[local_j] = V[index];	
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

			
			int row = blockIdx.x * blockDim.x + threadIdx.x;


			// S_ij = Q_i*K_j^T, m_ij = row_max(S_ij)
			if (row < b_r) {
				float row_max = -INFINITY;
				for(int c = 0; c < b_c; c++){
					float dot_prod = 0;
					for(int k = 0; k < d; k++){
						dot_prod += q_i[row*d + k] * k_j[c*d + k];
					}
					S[row*b_c + c] = dot_prod;
					row_max = max(dot_prod, row_max);
				}
				m_ij[row] = row_max;
			}
			
			// P_ij = exp(S_ij - m_ij), l_ij = rowsum(P_ij)
			if(row < b_r){
				float row_sum = 0;
				for(int c = 0; c < b_c; c++){
					int idx = row*b_c + c;
					float S_ij = S[idx];
					float m_ij = m_ij[row];
					float exp_val = __expf(S_ij - m_ij);

					p_ij[idx] = exp_val;
					row_sum += exp_val;
				}
				l_ij[row] = row_sum;
			}
			
			// compute m_new, l_new
			if(row < b_r){
				float m_i_prev = m_i[row];
				float m_ij = m_ij[row];
				float m_i_new = max(m_i_prev, m_ij);
				m_new[row] = m_i_new;

				float l_i_prev = l_i[row];
				float l_ij = l_ij[row];
				float l_i_new = __expf(m_i_prev - m_i_new) * l_i_prev + __expf(m_ij - m_i_new) * l_ij;
				l_new[row] = l_i_new;
			}
			
			// Write O_i, l_i, m_i to HBM
			if(row < b_r){
				float l_new_val = l_new[row];
				float m_i_prev = m_i[row];
				float m_ij_curr = m_ij[row];
				float m_new_val = m_new[row];
				float l_i_prev = l_i[row];
				float l_ij_curr = l_ij[row];

				for(int c = 0; c < d; c++){  // Note: iterating over d, not b_c
					float prev_O_i = o_i[row * d + c];
					float v_j = v_j[row * d + c];  // Assuming v_j is transposed
					o_i[row * d + c] = (1.0f / l_new_val) * 
						(l_i_prev * __expf(m_i_prev - m_new_val) * prev_O_i + 
						__expf(m_ij_curr - m_new_val) * l_ij_curr * v_j);
				}
				
				m_i[row] = m_new_val;
				l_i[row] = l_new_val;
			}
			
		}

	}

}
