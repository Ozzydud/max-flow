#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

int main() {
    const int N = 20;
    size_t size = N * sizeof(float);

    float *A, *B, *C;          // Host vectors
    float *d_A, *d_B, *d_C;    // Device vectors

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    VecAdd<<<1, N>>>(d_A, d_B, d_C); // 1 block, n threads

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // We copy back to host.
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

     // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    std::cout << "GPU Vector addition completed in " << duration.count() << " milliseconds." << std::endl;
    
    return 0;
}

// Kernel definition
__global__ void VecAdd(float* d_A, float* d_B, float* d_C)
{
    int i = threadIdx.x;
    a, b = threadIdx.y, threadIdx.z
    d_C[i] = d_A[i] + d_B[i];
}