#include "bitnet_porter.cuh"
#include "common.cuh"
// External bridge to the highly-optimized dequantizer for the 36-byte I2_S format.
extern void dequantize_row_i2_s_cuda_real_f32(const void * vx, float * y, const int64_t k, cudaStream_t stream);

extern "C" void bitnet_cuda_porter_axon(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    // Instead of dummy copying or bit-rearrangement, use the float dequantizer directly.
    dequantize_row_i2_s_cuda_real_f32(src, (float *)dst, ne, stream);
}

// Activation Quantization Kernel (Symmetric i8)
__global__ void k_quantize_i8(const float * src, int8_t * dst, int64_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(src[i] * 127.0f)));
    }
}

extern "C" void bitnet_mul_mat_ladder_axon(
    const char * src0, const char * src1, float * dst,
    int64_t ne00, int64_t ne01, int64_t ne11, int64_t ne0,
    cudaStream_t stream) {

    // src0: Weights (Ladder transmuted I2_S) -> int8_t*
    // src1: Activations (from llama.cpp, assume float* for now)
    
    if (ne11 == 1) {
        // High-performance bridge for Phase 3.4
        // Logic to be refined as we manage BF16 buffers for output and scales
        // For now, we remain silent and allow the DMMV fallback if specific shapes aren't matched.
    }
}
