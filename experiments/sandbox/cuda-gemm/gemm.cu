#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error for %s: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(2);
    }
}

void check_cublas(cublasStatus_t status, const char* expression) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error for %s: %d\n", expression,
                     static_cast<int>(status));
        std::exit(3);
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr)

int parse_positive(const char* flag, const char* value) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 1000000) {
        std::fprintf(stderr, "invalid value for %s: %s\n", flag, value);
        std::exit(64);
    }
    return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
    int matrix_size = 16384;
    int warmup = 20;
    int iterations = 400;
    int seed = 0;

    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[index]);
            return 64;
        }
        if (std::strcmp(argv[index], "--matrix-size") == 0) {
            matrix_size = parse_positive(argv[index], argv[index + 1]);
        } else if (std::strcmp(argv[index], "--warmup") == 0) {
            warmup = parse_positive(argv[index], argv[index + 1]);
        } else if (std::strcmp(argv[index], "--iterations") == 0) {
            iterations = parse_positive(argv[index], argv[index + 1]);
        } else if (std::strcmp(argv[index], "--seed") == 0) {
            seed = std::atoi(argv[index + 1]);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
            return 64;
        }
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    CUDA_CHECK(cudaSetDevice(0));

    const size_t elements = static_cast<size_t>(matrix_size) * matrix_size;
    const size_t bytes = elements * sizeof(__half);
    __half* matrix_a = nullptr;
    __half* matrix_b = nullptr;
    __half* matrix_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matrix_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matrix_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matrix_c), bytes));
    CUDA_CHECK(cudaMemset(matrix_a, 0, bytes));
    CUDA_CHECK(cudaMemset(matrix_b, 0, bytes));
    CUDA_CHECK(cudaMemset(matrix_c, 0, bytes));

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    const float alpha = 1.0F;
    const float beta = 0.0F;

    const auto gemm = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size, matrix_size,
            matrix_size, &alpha, matrix_a, CUDA_R_16F, matrix_size, matrix_b,
            CUDA_R_16F, matrix_size, &beta, matrix_c, CUDA_R_16F, matrix_size,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    for (int iteration = 0; iteration < warmup; ++iteration) {
        gemm();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        gemm();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const long double operations =
        2.0L * matrix_size * matrix_size * matrix_size * iterations;
    const long double elapsed_seconds = elapsed_ms / 1000.0L;
    const long double tflops = operations / elapsed_seconds / 1.0e12L;

    std::printf(
        "{\"workload\":\"cublas-fp16-gemm\",\"device\":\"%s\","
        "\"matrix_size\":%d,\"warmup\":%d,\"iterations\":%d,"
        "\"seed\":%d,\"elapsed_ms\":%.6f,\"operations\":%.0Lf,"
        "\"tflops\":%.6Lf}\n",
        properties.name, matrix_size, warmup, iterations, seed, elapsed_ms,
        operations, tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(matrix_a));
    CUDA_CHECK(cudaFree(matrix_b));
    CUDA_CHECK(cudaFree(matrix_c));
    return 0;
}
