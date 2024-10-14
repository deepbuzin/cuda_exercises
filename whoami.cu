#include <stdio.h>


__global__ void whoami() {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
    int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;
    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    printf("global thread id %d\n", block_offset + thread_offset);
}

int main() {
   int b_x = 2, b_y = 3, b_z = 4;  // block size in threads
   int g_x = 2, g_y = 2, g_z = 2;  // grid size in blocks

   dim3 block_dim(b_x, b_y, b_z);
   dim3 grid_dim(g_x, g_y, g_z);

   whoami<<<grid_dim, block_dim>>>();
   cudaDeviceSynchronize();
}
