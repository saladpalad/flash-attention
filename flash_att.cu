#include <cuda.h>
#include <cuda_runtime.h>
#include <math>
#include <algorithm>

__global__ void forward_kernel(float* Q, float *K, float *V, float* O, float* l, float* m, int M_SIZE, int N, int d){
	int b_c = math.ceil(M_SIZE/4*d);
	int b_r = min(math.ceil(M_SIZE/4*d), d);
	
	// num of blocks we have
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
	row_l


	
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
		



			
			
			//fix this row_l should be an array		
			// P = exp(S - r w_m), row_l = rowsum(P)
			float row_l = 0;
			for(int r = threadIdx.y; r < b_r; r += blockDim.y){
				for(int c = threadIdx.x c < b_c; c += blockDim.x){
					int idx = r * b_c + c;
					S[idx] = __expf(S[idx] - row_m);
					row_l += S[idx];
				
				}
			}

			// compute m_new, l_new
		}
	
	}





}


int main(){
// How to init Q,K,V ???
// do i use memcpy and what not	

}


