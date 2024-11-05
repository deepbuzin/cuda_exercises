#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1e7

#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

#define DELTA 1e-5

void cpu_add(float* a, float* b, float* c, int n)
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void cuda_add(float* a, float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void cuda_add_3d(float* a, float* b, float* c, int nx, int ny,
    int nz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Figure out if we're still within data bounds
    if (i < nx && j < ny && k < nz) {
        // Calculate index within 1d vector
        int idx = i + nx * j + ny * nx * k;

        if (idx < nx * ny * nz) {
            c[idx] = a[idx] + b[idx];
        }
    }
}
void init_vector(float* v, int n)
{
    for (int i = 0; i < n; i++) {
        v[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    float *h_a, *h_b, *h_c, *h_c_cuda, *h_c_cuda_3d;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_c_cuda = (float*)malloc(size);
    h_c_cuda_3d = (float*)malloc(size);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    srand(time(NULL));

    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int grid_size_1d = ceil(N / BLOCK_SIZE_1D);

    int nx = 10e2;
    int ny = 10e2;
    int nz = 10e3;

    dim3 block_size_3d = (BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid_size_3d = (ceil(nx / BLOCK_SIZE_X), ceil(ny / BLOCK_SIZE_Y),
        ceil(nz / BLOCK_SIZE_Z));

    // 1. warmup
    for (int i = 0; i < 20; i++) {
        cpu_add(h_a, h_b, h_c, N);
        cuda_add<<<grid_size_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        cuda_add_3d<<<grid_size_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
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
        cuda_add<<<grid_size_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        cuda_total_time += time(NULL) - start_time;
    }

    double avg_cuda_time = cuda_total_time /= 20.0;
    printf("Avg CUDA time %.4f\n", avg_cuda_time);

    // 4. gpu 3d
    double cuda_total_time_3d = 0.0;

    for (int i = 0; i < 20; i++) {
        double start_time = time(NULL);
        cuda_add_3d<<<grid_size_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize();
        cuda_total_time_3d = time(NULL) - start_time;
    }

    double avg_cuda_time_3d = cuda_total_time_3d = 20.0;
    printf("Avg CUDA time %.4f\n", avg_cuda_time_3d);

    // 5. compare outputs
    cpu_add(h_a, h_b, h_c, N);
    cudaDeviceSynchronize();

    cuda_add<<<grid_size_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c_cuda, d_c, size, cudaMemcpyDeviceToHost);

    cuda_add_3d<<<grid_size_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
    cudaMemcpy(h_c_cuda_3d, d_c, size, cudaMemcpyDeviceToHost);

    bool is_same = true;

    for (int i = 0; i < N; i++) {
        double diff = abs(h_c[i] - h_c_cuda[i]);
        if (diff > DELTA) {
            is_same = false;
        }

        double diff_3d = abs(h_c[i] - h_c_cuda_3d[i]);
        if (diff_3d > DELTA) {
            is_same = false;
        }
    }

    printf(is_same ? "Got same result" : "Got different result");

    printf("\n");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cuda);
    free(h_c_cuda_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
