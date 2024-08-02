#include <cuda.h>
#include <cuda_runtime.h>
#include <math>
#include <algorithm>

__global__ void forward_kernel(float* Q, float *K, float *V, float* O, float* l, float* m, int M_SIZE, int N, int d_size){
	int b_c = math.ceil(M_SIZE/4*d_size);
	int b_r = min(math.ceil(M_SIZE/4*d_size), d_size);

	int t_r = ceil(N/b_r);
	int t_c = ceil(N/b_c);

	int q_tile_size = b_r * d_size;
	int kv_tile_size = b_c * d_size;


	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	//don't know how to do indexing if QKV is 3d tensor
	int index = row;
	int tid = threadIdx.x;
	extern __shared__ float sram[];

	q_i = sram[q_tile_size];
	k_j = sram[kv_tile_size];
       	v_j = sram[kv_tile_size];
	o_i = sram[q_tile_size];
	l_i = sram[b_r];
	m_i = sram[b_r];

	for(int i = 0; i < t_c; ++i){
		
		// Load Kj, Vj from HBM to SRAM
		for(int j = 0; j < d_Size; ++j){
			k_j[tid] = K[index];
			v_j[tid] = V[index];	
		}	


		for(int j = 0; j < t_r; ++j){
			
			for(int k = 0; k < d; ++k){
				q_i[tid] = Q[index];
				O_i[tid] = O[index];
				l_i[tid] = l[index];
				m_i[tid] = m[index];			
			}
			
			// implement logic of the computations

		}
	
	}





}


int main(){
// How to init Q,K,V ???
// do i use memcpy and what not	

}


