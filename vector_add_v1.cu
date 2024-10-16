#include <stdio.h>
#include <stdlib.h>

#define N 32
#define BLOCK_SIZE 16


void cpu_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}


__global__ void cuda_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


void init_vector(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = i;
    }
}


int main() {
    float *h_a, *h_b, *h_c, *h_c_cuda;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*) malloc(size);
    h_b = (float*) malloc(size);
    h_c = (float*) malloc(size);
    h_c_cuda = (float*) malloc(size);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cpu_add(h_a, h_b, h_c, N);
    cuda_add<<<2, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c_cuda, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%.0f ", h_c_cuda[i]);
    }
    
    printf("\n");
}




