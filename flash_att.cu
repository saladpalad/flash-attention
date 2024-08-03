#include <cuda.h>
#include <cuda_runtime.h>
#include <math>
#include <algorithm>

__global__ void forward_kernel(float* Q, float *K, float *V, float* O, float* l, float* m, int M, int N, int d){
	
	// tile dimensions	
	int b_c = math.ceil(M/4*d);
	int b_r = min(math.ceil(M/4*d), d);
	
	// num of tiles
	int t_r = ceil(N/b_r);
	int t_c = ceil(N/b_c);

	int q_block_size = b_r * d;
	int kv_block_size = b_c * d;
	int s_block_size = b_r * b_c;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	
	extern __shared__ float sram[];
	
	// allocate SRAM  partitions
	q_i = sram;
	k_j = &sram[q_block_size];
       	v_j = &sram[q_block_size + kv_block_size];
	o_i = &sram[q_block_size + 2*kv_block_size];
	l_i = &sram[q_block_size + 2*kv_block_size + q_block_size];
	m_i = &sram[q_block_size + 2*kv_block_size + q_block_size + b_r];
	S = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r];
	row_m = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r + s_block_size];
	row_l = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r + s_block_size + b_r];
	new_row_m = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r + s_block_size + 2*b_r]
	new_row_l = &sram[q_block_size + 2*kv_block_size + q_block_size + 2*b_r + s_block_size + 3*b_r]

	
	for(int i = 0; i < t_c; i++){
		
		// Load Kj, Vj blocks from HBM to SRAM
		for(int local_j = tid; local_j < kv_block_size; local_j += blockDim.x * blockDim.y){
			int index = i * kv_block_size +	local_j;	
			if(index < N*d){
				k_j[local_j] = K[index];
				v_j[local_j] = V[index];	
			}	
		}	


		for(int j = 0; j < t_r; ++j){

			// Load Qi, Oi from HBM to SRAM
			for(int local_k = tid; local_k < q_block_size; local_k += blockDim.x * blockDim.y){
				int index = j * q_block_size + local_k;	
				if(index < N*d)l{
					q_i[local_k] = Q[index];
					O_i[local_k] = O[index];
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

			// implement logic of the computations
	
			//initialize row_m	
			for(int r = threadIdx.y; r < b_r; r +=blockDim.y){
				row_m[r] = -INFINITY;	
			}	

			__syncthreads();

			// S = Q*K^T, row_m = row_max(S)	
			for (int idx = blockDim.x * threadIdx.y + threadIdx.x; idx < s_tile_size; idx += blockDim.x * blockDim.y){
				int r = idx/b_c;
				int c = idx % b_c;
				float sum = 0;
				
				for(int k = 0; k < d; k++){
					sum += q_i[r*d + k] * k_j[k*d + c];	
				}
				S[idx] = sum;
				if(sum > row_m[r]) row_m[r] = sum;	
			
			}	
			__syncthreads();
			
			/*
			for(int r =threadIdx.y; r < b_r; r+=blockDim.y){
				row_l[r] = 0;	
			}
			*/		
			
			//fix this row_l should be an array		
			// P = exp(S - r w_m), row_l = rowsum(P)
			float row_l = 0;
			for(int r = threadIdx.y; r < b_r; r += blockDim.y){
				float sum = 0;	
				for(int c = threadIdx.x c < b_c; c += blockDim.x){
					int idx = r * b_c + c;
					float p = __expf(S[idx] - row_m);
					S[idx] = p;
					sum += p;	
				}
				row_l[r] = sum;
			}
			__syncthreads();

			// compute m_new, l_new
			for(int r = threadIdx.y; r < b_r; r += blockDim.y){
				if(threadIdx.x == 0){ // one thread per row
					float m_i = m[blockIdx.y * b_r + r] // global max for row i out of all the blocks processed so far
					float m_ij = row_max[r]; //local max for this block

					new_row_m[r] = max(m_i, m_ij);



					float l_i = l[blockIdx.y * b_r + r] //global running sum
					float l_ij = row_l[r]; // local running sum
					float new_m_i = new_row_m[r];
					

					new_row_l[r] = __expf(m_i - new_m_i) * l_i + __expf(m_ij - new_m_i) * l_ij;
				}
			}
			__syncthreads();
			
			// write O, l, m to HBM 
			
		}
	
	}





}


int main(){
// How to init Q,K,V ???
// do i use memcpy and what not	

}


