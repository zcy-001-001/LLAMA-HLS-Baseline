#include "forward.h" // 包含修正后的声明和类型定义
#include <cstring>   // For memcpy, memset
#include <cmath>     // For sqrtf, expf, cosf, sinf, powf
// Note: hls_stream.h 通常通过 forward.h 包含



// ============================================================================
// Kernel 1: Initial Embedding Lookup (Outputting to Stream)
// --- FIXED: Added missing parameter list to the definition ---
// ============================================================================
extern "C" void
initial_embedding_lookup(
    const float* token_embedding_table, // Input: Pointer to table in global memory
    int token,                          // Input: Token index
    hls::stream<float>& stream_out_x    // Output: Stream for initial embedding 'x' state
) { // <<< --- 参数列表已补全 ---
    // Interfaces
#pragma HLS INTERFACE m_axi port=token_embedding_table offset=slave bundle=gmem0 depth=vocab_size*dim latency=100 num_read_outstanding=32
#pragma HLS INTERFACE axis port=stream_out_x
#pragma HLS INTERFACE s_axilite port=token
#pragma HLS INTERFACE s_axilite port=return

    // Use compile-time constant 'dim' directly for array size
    float embedding_buffer[dim];
#pragma HLS ARRAY_PARTITION variable=embedding_buffer type=cyclic factor=2

    // Add check for token bounds to prevent potential SIGSEGV in C-sim
    if (token < 0 || token >= vocab_size) { // 现在 token 是已声明的参数
        // Handle error: e.g., stream out zeros or abort simulation
        for (int i = 0; i < dim; ++i) {
        #pragma HLS PIPELINE II=1
            stream_out_x.write(0.0f); // 现在 stream_out_x 是已声明的参数
        }
        return; // Exit early
    }

    // 现在 token_embedding_table 和 token 是已声明的参数
    const float* embedding_start_addr = token_embedding_table + token * dim;

read_embedding:
    std::memcpy(embedding_buffer, embedding_start_addr, dim * sizeof(float));

write_embedding_stream:
    for (int i = 0; i < dim; ++i) { // Use constant dim
#pragma HLS PIPELINE II=1
        stream_out_x.write(embedding_buffer[i]); // 现在 stream_out_x 是已声明的参数
    }
}
