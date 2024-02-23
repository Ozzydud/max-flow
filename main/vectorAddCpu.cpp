#include <iostream>

int main() {
    const int N = 200000000;
    size_t size = N * sizeof(float);

    float *A, *B, *C; // Host vectors

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Vector addition on CPU
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Free host memory
    free(A);
    free(B);
    free(C);

    std::cout << "CPU Vector addition completed in " << duration.count() << " milliseconds." << std::endl;
    
    return 0;
}
