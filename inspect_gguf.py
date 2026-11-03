import sys
import struct

import gguf

# Uso: python inspect_gguf.py <ruta.gguf>
path = sys.argv[1] if len(sys.argv) > 1 else "model.gguf"
reader = gguf.GGUFReader(path)
for tensor in reader.tensors:
    if tensor.tensor_type.name == 'I2_S': 
        print(f"Found I2_S tensor: {tensor.name}")
        data = tensor.data
        print(f"Data length: {len(data)}")
        # Print first block (36 bytes: 32 bytes data + 4 bytes scale)
        block = data[:36]
        qs = block[:32]
        d = struct.unpack('<f', block[32:36])[0]
        print(f"Scale: {d}")
        print(f"QS bytes (first 8): {[hex(b) for b in qs[:8]]}")
        break
