#include "forward.h" // 包含修正后的声明和类型定义
#include <cstring>   // For memcpy, memset
#include <cmath>     // For sqrtf, expf, cosf, sinf, powf
// Note: hls_stream.h 通常通过 forward.h 包含
// ============================================================================
// Kernel 4: Final Norm and Classifier (Inputting from Stream)
// --- UPDATED QuantizedTensor type and quantize call ---
// ============================================================================
extern "C" void
final_norm_classifier(
    hls::stream<float>& stream_in_x,
    float* logits_out,                 // Output logits (modified)
    const ModelWeights_t* w,           // Input weights (const)
    float GS_val                       // Group size value (if needed)
) {
    // ... (接口) ...
#pragma HLS INTERFACE axis port=stream_in_x
#pragma HLS INTERFACE m_axi port=logits_out offset=slave bundle=gmem_out depth=vocab_size latency=100 num_write_outstanding=32
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem_w latency=100 num_read_outstanding=32
#pragma HLS INTERFACE s_axilite port=GS_val
#pragma HLS INTERFACE s_axilite port=return

    // Constants dim, vocab_size available from config.h
    constexpr int UNROLL_FACTOR = 2; // Local constant

    // Static Buffers
    float x_local[dim];
    float x_norm[dim];
    // --- UPDATED QuantizedTensor Definition ---
    QuantizedTensor<dim, GS> xq; // Use global GS

    // --- UPDATED Partitioning for xq.s ---
    // Partition arrays
#pragma HLS ARRAY_PARTITION variable=x_local type=cyclic factor=UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable=x_norm type=cyclic factor=UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable=xq.q cyclic factor=UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable=xq.s type=cyclic factor=UNROLL_FACTOR // Partition scales

    // Read input stream
read_x_stream_final:
    for (int i = 0; i < dim; ++i) {
#pragma HLS PIPELINE II=1
        x_local[i] = stream_in_x.read();
    }

    // --- Final RMSNorm ---
    rmsnorm<dim>(x_norm, x_local, w->rms_final_weight);

    // --- Final Classifier ---
    // --- UPDATED quantize call ---
    quantize<dim, GS>(&xq, x_norm, GS); // Pass dim and GS as template args

    // matmul call should be okay
    matmul<dim, vocab_size>(logits_out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}