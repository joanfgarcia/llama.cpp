#include "ggml-bitnet-axon-factory.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

const struct ggml_bitnet_axon_interface * ggml_bitnet_axon_find_best(void) {
    // 1. Manual Force Override (for experimentation/Consensus mode)
    const char * force_env = getenv("GGML_BITNET_FORCE_AXON");
    if (force_env) {
#ifdef GGML_USE_VULKAN
        if (strcmp(force_env, "VULKAN") == 0) return &ggml_bitnet_axon_vulkan;
#endif
        if (strcmp(force_env, "CUDA") == 0)   return &ggml_bitnet_axon_cuda;
        if (strcmp(force_env, "ROCM") == 0)   return &ggml_bitnet_axon_cuda;
        if (strcmp(force_env, "NPU") == 0)    return &ggml_bitnet_axon_cuda; // NPU Bridge to GPU
        if (strcmp(force_env, "CPU") == 0)    return &ggml_bitnet_axon_cpu;
        printf("ERROR: No se pudo activar el acelerador forzado.\n"); exit(1);
    }

    // 2. Hardware Autodetection Logic
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIPBLAS)
    return &ggml_bitnet_axon_cuda;
#endif

#ifdef GGML_USE_VULKAN
    return &ggml_bitnet_axon_vulkan;
#endif

    // 3. Absolute Fallback (also used for ROCm/HIP — uses DMMV path natively)
    printf("ERROR: No se pudo activar el acelerador forzado.\n"); exit(1);
}
