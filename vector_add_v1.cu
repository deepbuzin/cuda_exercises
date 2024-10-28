#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define N 1e7
#define BLOCK_SIZE 1024
#define DELTA 1e-5


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
        v[i] = (float) rand() / RAND_MAX;
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
    
    srand(time(NULL));

    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int grid_size = ceil(N / BLOCK_SIZE);

    // 1. warmup
    for (int i = 0; i < 20; i++) {
        cpu_add(h_a, h_b, h_c, N);
        cuda_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // 2. cpu
    double cpu_total_time = 0.0;

    for (int i = 0; i < 20; i++) {
        double start_time = time(NULL);
        cpu_add(h_a, h_b, h_c, N);
        cpu_total_time += time(NULL) - start_time;
    }

    double avg_cpu_time = cpu_total_time /= 20.0;
    printf("Avg CPU time %.4f\n", avg_cpu_time);

    // 3. gpu
    double cuda_total_time = 0.0;

    for (int i = 0; i < 20; i++) {
        double start_time = time(NULL);
        cuda_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        cuda_total_time += time(NULL) - start_time;
    }

    double avg_cuda_time = cuda_total_time /= 20.0;
    printf("Avg CUDA time %.4f\n", avg_cuda_time);

    // 4. compare outputs
    cpu_add(h_a, h_b, h_c, N);
    cuda_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c_cuda, d_c, size, cudaMemcpyDeviceToHost);

    bool is_same = true;

    for (int i = 0; i < N; i++) {
        double diff = abs(h_c[i] - h_c_cuda[i]);
        if (diff > DELTA) {
            is_same = false;
        }
    }

    printf(is_same ? "Got same result" : "Got different result");

    printf("\n");
}




