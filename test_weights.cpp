#include "ggml.h"
#include "llama.h"
#include <stdio.h>

int main() {
    llama_model_params params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file("/home/joan/Documents/IA/sharing/storage/models/falcon3-10b-instruct-1.58bit-V2.gguf", params);
    if (!model) { printf("Failed to load model\n"); return 1; }
    
    // We want to find a tensor with type GGML_TYPE_I2_S and print its contents
    // Let's use the ggml_context from the model. 
    // Actually llama.h does not expose the internal ggml_context directly for iteration.
    // We can just use the provided C API to get a tensor by name!
    const char * target_name = "blk.0.ffn_down.weight";
    // We need to look up the model tensor.
    // llama_get_model_tensor is available!
    struct ggml_tensor * tensor = llama_get_model_tensor(model, target_name);
    if (tensor) {
        printf("Found tensor %s, type: %d\n", target_name, tensor->type);
        if (tensor->type == GGML_TYPE_I2_S) {
            uint8_t * data = (uint8_t *)tensor->data;
            printf("First 36 bytes:\n");
            for(int i=0; i<36; i++) {
                printf("%02x ", data[i]);
            }
            printf("\n");
        }
    } else {
        printf("Tensor %s not found\n", target_name);
    }
    return 0;
}
