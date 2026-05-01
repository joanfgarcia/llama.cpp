#include "dequantize-i2s.cuh"
#include <cuda_fp16.h>
#include <stdint.h>

static __global__ void dequantize_block_i2_s_real_f32(const void * __restrict__ vx, float * __restrict__ yy, int64_t k) {
    const int i = blockIdx.x;
    const block_i2_s * x = (const block_i2_s *) vx;

    const float block_scale = x[i].d;
    const uint8_t * block_ptr = x[i].qs;

    const int base = i * 128;
    const int thread_id = threadIdx.x;

    if (thread_id < 32) {
        // Each thread processes one byte (which contains 4 ternary values)
        const uint8_t b = block_ptr[thread_id];
        
        const uint8_t c0 = (b >> 6) & 0x3;
        const uint8_t c1 = (b >> 4) & 0x3;
        const uint8_t c2 = (b >> 2) & 0x3;
        const uint8_t c3 = (b >> 0) & 0x3;

        const float m0 = (c0 == 0) ? -1.0f : (c0 == 2) ? 1.0f : 0.0f;
        const float m1 = (c1 == 0) ? -1.0f : (c1 == 2) ? 1.0f : 0.0f;
        const float m2 = (c2 == 0) ? -1.0f : (c2 == 2) ? 1.0f : 0.0f;
        const float m3 = (c3 == 0) ? -1.0f : (c3 == 2) ? 1.0f : 0.0f;

        if (base + 0*32 + thread_id < k) yy[base + 0*32 + thread_id] = (block_scale * m0);
        if (base + 1*32 + thread_id < k) yy[base + 1*32 + thread_id] = (block_scale * m1);
        if (base + 2*32 + thread_id < k) yy[base + 2*32 + thread_id] = (block_scale * m2);
        if (base + 3*32 + thread_id < k) yy[base + 3*32 + thread_id] = (block_scale * m3);
    }
}

static __global__ void dequantize_block_i2_s_real_f16(const void * __restrict__ vx, half * __restrict__ yy, int64_t k) {
    const int i = blockIdx.x;
    const block_i2_s * x = (const block_i2_s *) vx;

    const float block_scale = x[i].d;
    const uint8_t * block_ptr = x[i].qs;

    const int base = i * 128;
    const int thread_id = threadIdx.x;

    if (thread_id < 32) {
        // Each thread processes one byte (which contains 4 ternary values)
        const uint8_t b = block_ptr[thread_id];
        
        const uint8_t c0 = (b >> 6) & 0x3;
        const uint8_t c1 = (b >> 4) & 0x3;
        const uint8_t c2 = (b >> 2) & 0x3;
        const uint8_t c3 = (b >> 0) & 0x3;

        const float m0 = (c0 == 0) ? -1.0f : (c0 == 2) ? 1.0f : 0.0f;
        const float m1 = (c1 == 0) ? -1.0f : (c1 == 2) ? 1.0f : 0.0f;
        const float m2 = (c2 == 0) ? -1.0f : (c2 == 2) ? 1.0f : 0.0f;
        const float m3 = (c3 == 0) ? -1.0f : (c3 == 2) ? 1.0f : 0.0f;

        if (base + 0*32 + thread_id < k) yy[base + 0*32 + thread_id] = __float2half(block_scale * m0);
        if (base + 1*32 + thread_id < k) yy[base + 1*32 + thread_id] = __float2half(block_scale * m1);
        if (base + 2*32 + thread_id < k) yy[base + 2*32 + thread_id] = __float2half(block_scale * m2);
        if (base + 3*32 + thread_id < k) yy[base + 3*32 + thread_id] = __float2half(block_scale * m3);
    }
}

void dequantize_row_i2_s_cuda_real_f32(const void * vx, float * yy, const int64_t k, cudaStream_t stream) {
    int num_blocks = (k + 127) / 128;
    dequantize_block_i2_s_real_f32<<<num_blocks, 32, 0, stream>>>(vx, yy, k);
}

void dequantize_row_i2_s_cuda_real_f16(const void * vx, half * yy, const int64_t k, cudaStream_t stream) {
    int num_blocks = (k + 127) / 128;
    dequantize_block_i2_s_real_f16<<<num_blocks, 32, 0, stream>>>(vx, yy, k);
}
