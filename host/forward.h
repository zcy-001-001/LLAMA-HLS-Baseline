#ifndef FORWARD_H
#define FORWARD_H

// #include "hls_stream.h" // Needed for hls::stream
#include "config.h"     // Defines constants like dim, GS, etc.
#include "typedefs.h"   // Defines Config, Transformer, QuantizedTensor structs etc. (NOW CORRECTED)
#include <cstring>      // For memcpy, memset
#include <cmath>        // For fabs, round, sqrtf, expf, cosf, sinf, powf


// ----------------------------------------------------------------------------
// Typedef for the specific TransformerWeights instantiation using config.h constants
// --- UPDATED typedef to pass GS ---
// ----------------------------------------------------------------------------
typedef TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> ModelWeights_t;

// ----------------------------------------------------------------------------
// Declarations for the split kernels (No changes needed here)
// ----------------------------------------------------------------------------

extern "C" {

// void initial_embedding_lookup(
//     const float* token_embedding_table, // Note: This kernel directly accesses the float table part
//     int token,
//     hls::stream<float>& stream_out_x
// );

// // --- ADDED: 新的合并了层循环的 Kernel 声明 ---
// void transformer_layer_pipeline(
//     hls::stream<float>& stream_initial_in, // 来自 embedding 的输入流
//     hls::stream<float>& stream_final_out,  // 输出到 final norm 的流
//     const ModelWeights_t* w,               // 权重指针
//     int pos,                               // 当前位置
//     float* key_cache,                      // KV Cache 指针
//     float* value_cache                     // KV Cache 指针
// );


// void final_norm_classifier(
//     hls::stream<float>& stream_in_x,
//     float* logits_out,
//     const ModelWeights_t* w // Uses the corrected typedef
// );
void forward(
    const float* token_embedding_table, // Global memory for embeddings
    int token,                          // Input token index
    const ModelWeights_t* w,            // Global memory for weights (shared)
    int pos,                            // Current position for KV cache
    float* key_cache,                   // Global memory for Key cache
    float* value_cache,                 // Global memory for Value cache
    const ModelWeights_t* w1,           // Input weights (const)
    float* logits_out                   // Global memory for final logits
);
} // extern "C"

// ----------------------------------------------------------------------------
// Template function definitions (kept in header for template instantiation)
// --- UPDATED quantize/dequantize signatures ---
// ----------------------------------------------------------------------------

// --- UPDATED Signature ---
template <int S, int GROUP_SIZE> // Added GROUP_SIZE template parameter
void dequantize(const QuantizedTensor<S, GROUP_SIZE> *qx, float x[S], int group_size) { // Made qx const, Renamed param
    // Check parameter consistency (optional but good)
    if (group_size <= 0 || group_size != GROUP_SIZE) {
        // Handle error: maybe fill x with zeros
        memset(x, 0, S * sizeof(float));
        return;
    }

    for (int i = 0; i < S; i++) {
        // Use group_size parameter (which should match GROUP_SIZE template param)
        int scale_idx = i / group_size;
        x[i] = qx->q[i] * qx->s[scale_idx];
    }
}

typedef union {
    float f32;
    uint32_t u32;
} float_uint32_union;

// 快速转换函数
inline uint32_t float_to_uint32(float a) {
    float_uint32_union u;
    u.f32 = a;
    return u.u32;
}

inline float uint32_to_float(uint32_t a) {
    float_uint32_union u;
    u.u32 = a;
    return u.f32;
}

// --- UPDATED Signature & FIXED hardcoded 64 ---
template <int S, int GROUP_SIZE> // Added GROUP_SIZE template parameter
void quantize(QuantizedTensor<S, GROUP_SIZE> *qx, const float x[S], int group_size) { // made x const, Renamed param

    // Check parameter consistency (optional but good)
    if (group_size <= 0 || group_size != GROUP_SIZE || S % group_size != 0) {
        // Handle error or default behavior
        memset(qx->q, 0, S * sizeof(int8_t));
        int scale_size = (S > 0 && group_size > 0 && S % group_size == 0) ? (S / group_size) : 0;
        if (scale_size > 0) {
            memset(qx->s, 0, scale_size * sizeof(float));
        }
        return;
    }

    constexpr float Q_MAX = 127.0f;
    // Calculate num_groups based on the 'group_size' parameter
    const int num_groups = S / GS; // <<< FIXED: Use parameter

    // Local buffers (Consider stack limits for very large S)
    // Allocate buffer size based on calculated num_groups
    float scale_buffer[num_groups];
    int8_t quantized_buffer[S];

// #pragma HLS ARRAY_PARTITION variable = quantized_buffer type = cyclic factor = 64 // Example factor, adjust as needed
// #pragma HLS ARRAY_PARTITION variable = scale_buffer type = cyclic factor = 16   // Example factor, adjust as needed

    float x_local[S];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=quantized_buffer cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=scale_buffer complete

// Copy input data to local buffer with better memory access pattern
    for (int i = 0; i < S; i++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=4
        x_local[i] = x[i];
    }


main_loop:
    for (int group = 0; group < num_groups; group++) {
#pragma HLS PIPELINE off
        float wmax = 0.0f;
        // Use 'group_size' parameter
        int base_idx = group * group_size;

    // First pass: find maximum absolute value in the group
    max_val_loop:
        for (int i = 0; i < group_size; i += 4) {
#pragma HLS PIPELINE
            // Process 4 elements at a time to reduce loop iterations
            float val1 = (i < group_size) ? fabs(x_local[base_idx + i]) : 0.0f;
            float val2 = (i+1 < group_size) ? fabs(x_local[base_idx + i + 1]) : 0.0f;
            float val3 = (i+2 < group_size) ? fabs(x_local[base_idx + i + 2]) : 0.0f;
            float val4 = (i+3 < group_size) ? fabs(x_local[base_idx + i + 3]) : 0.0f;
            
            // Find max of first pair
            float max_pair1 = (val1 > val2) ? val1 : val2;
            // Find max of second pair
            float max_pair2 = (val3 > val4) ? val3 : val4;
            // Find max of the two pairs
            float max_quad = (max_pair1 > max_pair2) ? max_pair1 : max_pair2;
            
            // Update overall max
            wmax = (max_quad > wmax) ? max_quad : wmax;
        }

        // Calculate scale - add small epsilon to avoid division by zero
        // float scale = (wmax > 1e-10f) ? (wmax / Q_MAX) : 1e-10f;
// #pragma HLS BIND_OP variable=scale op=fdiv impl=fabric latency=30
        float_uint32_union wmu, qmu, su;
        wmu.f32 = wmax;
        qmu.f32 = Q_MAX;
        //su.u32 = wmu.u32 + 0x3F780000 - qmu.u32;
        uint32_t tempu1 = wmu.u32 + 0x3F780000;
        #pragma HLS BIND_OP variable=tempu1 op=add impl=dsp latency=4
        uint32_t resultu1 = tempu1 - qmu.u32;
        #pragma HLS BIND_OP variable=resultu1 op=sub impl=dsp latency=4
        su.u32 = resultu1;

        // Store scale for this group
        // scale_buffer[group] = scale;
        scale_buffer[group] = su.f32;
        
        // Pre-compute reciprocal of scale to replace division with multiplication
        // float inv_scale = 1.0f / scale;
// #pragma HLS BIND_OP variable=inv_scale op=fdiv impl=fabric latency=20

    // Second pass: quantize using the scale factor
    quant_loop:
        for (int i = 0; i < group_size; i += 2) {
#pragma HLS PIPELINE
            // Process 2 elements at a time
            if (i < group_size) {
                float val = x_local[base_idx + i];
                // float quant_value = val * inv_scale;
                // float quant_value = val / scale;
                // #pragma HLS BIND_OP variable=quant_value op=fdiv impl=fabric latency=30
                float_uint32_union xu, qu;
                xu.f32 = val;
                uint32_t tempu2 = xu.u32 + 0x3F780000;
                #pragma HLS BIND_OP variable=tempu2 op=add impl=dsp
                uint32_t resultu2 = tempu2 - su.u32;
                #pragma HLS BIND_OP variable=resultu2 op=sub impl=dsp
                qu.u32 = resultu2;
                float quant_value = qu.f32;

// #pragma HLS BIND_OP variable=quant_value op=fmul impl=dsp latency=4
                quantized_buffer[base_idx + i] = (int8_t)quant_value;
            }
            
            if (i+1 < group_size) {
                float val = x_local[base_idx + i + 1];
                // float quant_value = val * inv_scale;
                // float quant_value = val / scale;
                // #pragma HLS BIND_OP variable=quant_value op=fdiv impl=fabric latency=30
                float_uint32_union xu, qu;
                xu.f32 = val;
                uint32_t tempu2 = xu.u32 + 0x3F780000;
                #pragma HLS BIND_OP variable=tempu2 op=add impl=dsp
                uint32_t resultu2 = tempu2 - su.u32;
                #pragma HLS BIND_OP variable=resultu2 op=sub impl=dsp
                qu.u32 = resultu2;
                float quant_value = qu.f32;

// #pragma HLS BIND_OP variable=quant_value op=fmul impl=dsp latency=4
                quantized_buffer[base_idx + i + 1] = (int8_t)quant_value;
            }
        }
    }

    // Copy results to output struct
    std::memcpy(qx->q, quantized_buffer, S * sizeof(int8_t));
    // Copy correct number of scales
    std::memcpy(qx->s, scale_buffer, num_groups * sizeof(float)); // <<< Correct size based on num_groups
}



// ============================================================================
// Helper Functions (rmsnorm, softmax, matmul)
// --- 说明: 模板函数的定义通常应放在头文件中 (.h) ---
// --- 但我们在此处修正它们以匹配头文件中的声明和类型 ---
// ============================================================================

template <int S>
// const 限定符已在上次修正中添加
void rmsnorm(float o[S], const float x[S], const float weight[S]) {
    constexpr auto array_size = S * sizeof(float);
    float ss = 0.0f;
    float x_buff[S];
    float weight_buff[S];
    float out_buff[S];
// 根据需要调整分区因子
#pragma HLS array_partition variable = x_buff type = cyclic factor = 2
#pragma HLS array_partition variable = weight_buff type = cyclic factor = 2
#pragma HLS array_partition variable = out_buff type = cyclic factor = 2 // 示例因子
    std::memcpy(x_buff, x, array_size);
    std::memcpy(weight_buff, weight, array_size);

sum_of_squares:
    for (int j = 0; j < S; j++) {
#pragma HLS UNROLL factor=2 skip_exit_check // 示例因子
        float x_j = x_buff[j];
        ss += x_j * x_j;
    }
    ss /= S;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

norm_and_scale:
    for (int j = 0; j < S; j++) {
#pragma HLS UNROLL factor=2 // 示例因子
        float weight_j = weight_buff[j];
        float x_j = x_buff[j];
        out_buff[j] = weight_j * (ss * x_j);
    }
    std::memcpy(o, out_buff, array_size);
}

// Softmax 定义保持不变 (它不直接依赖于 QuantizedTensor 的改动)
template <int MAXSIZE>
void softmax(float *x, int size) {
    float buffer[MAXSIZE];
#pragma HLS array_partition variable=buffer type=cyclic factor=2 // Example factor
    if (size <= 0) return;
    float max_val = x[0];
max:
    for (int i = 1; i < size; i++) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2 // Use config constants
        if (x[i] > max_val) { max_val = x[i]; }
    }
    float sum = 0.0f;
exp_sum: // Merged loop from previous example version
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2
#pragma HLS UNROLL factor = 2 // Example factor
        buffer[i] = expf(x[i] - max_val);
        sum += buffer[i];
    }
    const float inv_sum = (sum == 0.0f) ? 0.0f : 1.0f / sum;
norm:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2
#pragma HLS UNROLL factor = 2 // Example factor
        x[i] = buffer[i] * inv_sum;
    }
}

template <int N, int D>
// const 限定符已在上次修正中添加
// 此函数内部逻辑依赖于全局 GS 常量，并处理从 QuantizedTensor 传入的原始指针
// 功能上与修正后的 QuantizedTensor<SIZE, GS> 内存布局兼容
void matmul(float *xout, const int8_t *xq, const float *xs, const int8_t *wq, const float *ws) {
    // W (d,n) @ x (n,) -> xout (d,)

    // 假设 GS 是通过 config.h 可用的全局常量

    int8_t x_buffer[N];
    // 大小基于全局 GS
    float xs_buffer[N / GS];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 2 // Example factor
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 2 // Example factor

x_buff:
    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor = 2 // Example factor
        x_buffer[i] = xq[i];
    }
xs_buff:
     // 加载对应分组的 scale 因子
     for (int j = 0; j < N / GS; j++) { // Loop over groups
 #pragma HLS UNROLL factor = 2 // Example factor
         xs_buffer[j] = xs[j]; // Assumes xs directly corresponds to groups
     }


    for (int i = 0; i < D; i++) { // Loop over output dimension
        float val = 0.0f;
        int8_t w_buffer[N];
        // 大小基于全局 GS
        float ws_buffer[N / GS];
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 2 // Example factor
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 2 // Example factor

        const int in_w = i * N;       // Start index in wq for row i
        const int in_s = i * (N / GS); // Start index in ws for row i

    load_w: // Load weights for current row
        for (int j = 0; j < N; j++) {
#pragma HLS UNROLL factor = 2 // Consider full unroll if N is small enough, or partial
            w_buffer[j] = wq[j + in_w];
        }
    load_ws: // Load scales for current row
        for (int j = 0; j < N / GS; j++) { // Loop over groups
#pragma HLS UNROLL factor = 2 // Consider full unroll if N/GS is small enough, or partial
            ws_buffer[j] = ws[j + in_s]; // Assumes ws directly corresponds to groups
        }

        // --- 使用之前重写的计算逻辑 (似乎更适合 HLS) ---
        // Perform dot product using groups
        int32_t group_sum[N/GS];
#pragma HLS ARRAY_PARTITION variable=group_sum complete // Partition for parallel accumulation

    dot_product_groups:
        for (int j = 0; j < N / GS; ++j) { // Loop over groups
#pragma HLS UNROLL factor = 2// Unroll group calculation
            int32_t ival = 0;
        inner_dot:
            for(int k=0; k<GS; ++k) { // Loop within group
#pragma HLS UNROLL factor = 2// Unroll inner dot product
                // Use static buffers loaded earlier
                ival += ((int16_t)x_buffer[j*GS + k]) * ((int16_t)w_buffer[j*GS + k]);
            }
            group_sum[j] = ival; // Store sum for the group
        }

    final_sum: // Accumulate scaled group results
        for(int j=0; j<N/GS; ++j) { // Loop over groups
#pragma HLS UNROLL factor = 2// Unroll final summation
             // Use loaded scales
            val += ((float)group_sum[j]) * ws_buffer[j] * xs_buffer[j];
        }
        // --------------------------------------------------

        xout[i] = val; // Store final output value
    }
}
#endif