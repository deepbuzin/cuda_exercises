#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
        int tid = i + nx * j + ny * nx * k;

        if (tid < nx * ny * nz) {
            c[tid] = a[tid] + b[tid];
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
    // Define data dims

    int N = 10e8;

    int nx = 10e2;
    int ny = 10e3;
    int nz = 10e3;

    int block_dim = 1024;
    dim3 block_dim_3d(16, 8, 8);

    int grid_dim = ceil(N / block_dim);
    dim3 grid_dim_3d(
        ceil(nx / block_dim_3d.x),
        ceil(ny / block_dim_3d.y),
        ceil(nz / block_dim_3d.z));

    // Define vars and allocate memory

    float *h_a, *h_b, *h_c, *h_c_cuda, *h_c_cuda_3d;
    float *d_a, *d_b, *d_c, *d_c_3d;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_c_cuda = (float*)malloc(size);
    h_c_cuda_3d = (float*)malloc(size);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_c_3d, size);

    // Init and move to GPU

    srand(time(NULL));

    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 1. Warmup

    for (int i = 0; i < 20; i++) {
        cpu_add(h_a, h_b, h_c, N);
        cuda_add<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
        cuda_add_3d<<<grid_dim_3d, block_dim_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // 2. CPU

    double cpu_total_time = 0.0;

    for (int i = 0; i < 20; i++) {
        double start_time = time(NULL);
        cpu_add(h_a, h_b, h_c, N);
        cpu_total_time += time(NULL) - start_time;
    }

    double avg_cpu_time = cpu_total_time /= 20.0;
    printf("Avg CPU time %.4f\n", avg_cpu_time);

    // 3. CUDA

    double cuda_total_time = 0.0;

    for (int i = 0; i < 20; i++) {
        double start_time = time(NULL);
        cuda_add<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        cuda_total_time += time(NULL) - start_time;
    }

    double avg_cuda_time = cuda_total_time /= 20.0;
    printf("Avg CUDA time %.4f\n", avg_cuda_time);

    // 4. CUDA 3d
    
    double cuda_total_time_3d = 0.0;

    for (int i = 0; i < 20; i++) {
        double start_time = time(NULL);
        cuda_add_3d<<<grid_dim_3d, block_dim_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        cuda_total_time_3d = time(NULL) - start_time;
    }

    double avg_cuda_time_3d = cuda_total_time_3d /= 20.0;
    printf("Avg CUDA 3d time %.4f\n", avg_cuda_time_3d);

    // 5. Compare outputs

    cpu_add(h_a, h_b, h_c, N);
    cudaDeviceSynchronize();

    cuda_add<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c_cuda, d_c, size, cudaMemcpyDeviceToHost);

    cuda_add_3d<<<grid_dim_3d, block_dim_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
    cudaMemcpy(h_c_cuda_3d, d_c_3d, size, cudaMemcpyDeviceToHost);

    bool is_same = true;

    for (int i = 0; i < N; i++) {
        double diff = abs(h_c[i] - h_c_cuda[i]);
        if (diff > DELTA) {
            is_same = false;
        }
    }

    printf(is_same ? "Got same result" : "Got different result \n");

    bool is_same_3d = true;

    for (int i = 0; i < N; i++) {
        double diff_3d = abs(h_c[i] - h_c_cuda_3d[i]);
        if (diff_3d > DELTA) {
            is_same_3d = false;
        }
    }

    printf(is_same_3d ? "Got same result 3d" : "Got different result for 3d \n");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cuda);
    free(h_c_cuda_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_3d);
}
