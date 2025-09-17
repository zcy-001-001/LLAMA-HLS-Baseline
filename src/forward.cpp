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


// ============================================================================
// --- 新的顶层 Kernel，逻辑合并版 ---
// ============================================================================
extern "C" void transformer_layer_pipeline(
    hls::stream<float>& stream_initial_in, // 初始嵌入输入流
    hls::stream<float>& stream_final_out,  // 最终输出流 (送往 final_norm)
    const ModelWeights_t* w,               // 权重指针
    const ModelWeights_t* w_ffn,
    int pos,   
    float* key_cache,                      // KV Cache 指针
    float* value_cache                   // KV Cache 指针
) {
    // --- 顶层接口定义 ---
#pragma HLS INTERFACE axis port=stream_initial_in name=s_initial_in
#pragma HLS INTERFACE axis port=stream_final_out name=s_final_out
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem_w  // 估算深度或移除
#pragma HLS INTERFACE m_axi port=w_ffn offset=slave bundle=gmem_w_ffn  
#pragma HLS INTERFACE m_axi port=key_cache offset=slave bundle=gmem_kvc depth=n_layers*seq_len*kv_dim latency=100 num_read_outstanding=32 num_write_outstanding=32
#pragma HLS INTERFACE m_axi port=value_cache offset=slave bundle=gmem_kvc depth=n_layers*seq_len*kv_dim latency=100 num_read_outstanding=32 num_write_outstanding=32
#pragma HLS INTERFACE s_axilite port=pos       
#pragma HLS INTERFACE s_axilite port=return 
constexpr int UNROLL_FACTOR = 2;  
static float current_x[dim];
#pragma HLS ARRAY_PARTITION variable=current_x type=cyclic factor=UNROLL_FACTOR

    // --- 用于层间状态传递的缓冲区 ---
//    static float current_x[dim];
    
    // --- 读取初始输入 ---
    ReadInitialInput:
    for (int i = 0; i < dim; ++i) {
    #pragma HLS PIPELINE II=1
        current_x[i] = stream_initial_in.read();
    }


    // --- 层循环 ---
LayerLoop:
    for (int l = 0; l < n_layers; ++l) {
        #pragma HLS PIPELINE off
        // --- 在循环内部声明所有需要的局部缓冲区 (非 static) ---
        // --- Attention 部分所需 ---
        float xb_attn[dim]; // 重命名以区分 FFN 的 xb
        float q[dim];
        float k[kv_dim]; // 使用 config.h 中的 kv_dim
        float v[kv_dim];
        float att[n_heads * seq_len];
        QuantizedTensor<dim, GS> xq_attn; // 重命名
        float attn_out_unquant[dim];
        float attn_output_no_residual[dim]; // 用于存储 Attention 输出投影结果 (无残差)

        // --- FFN 部分所需 ---
        float xb_ffn[dim]; // FFN 的 RMSNorm 输入
        float hb[hidden_dim];
        float hb2[hidden_dim];
        QuantizedTensor<dim, GS> xq_ffn; // FFN 输入量化结果
        QuantizedTensor<hidden_dim, GS> hq;
        float ffn_output_no_residual[dim]; // FFN 输出 (无残差)

        // --- 常量和 UNROLL 因子 ---
        constexpr int kv_dim_local = kv_dim; // 在此作用域内可能仍需定义
        constexpr int head_size_local = dim / n_heads;
        constexpr int kv_mul_local = n_heads / n_kv_heads;
        constexpr int ATTN_UNROLL = 2; // 示例 Attention 内部 Unroll 因子
        constexpr int FFN_UNROLL = 2;  // 示例 FFN 内部 Unroll 因子
   
        // --- 必要的数组分区 (复制并调整因子) ---
        // ... (将原来两个 Kernel 的所有 ARRAY_PARTITION pragma 复制到这里, 确保变量名对应) ...
        // 例如:
        #pragma HLS ARRAY_PARTITION variable=xb_attn type=cyclic factor=ATTN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=q type=cyclic factor=ATTN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=k type=cyclic factor=ATTN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=v type=cyclic factor=ATTN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=att type=cyclic factor=ATTN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=xq_attn.q type=cyclic factor=ATTN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=xq_attn.s type=cyclic factor=2 // 使用修正后的固定因子
        #pragma HLS ARRAY_PARTITION variable=attn_out_unquant type=cyclic factor=ATTN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=attn_output_no_residual type=cyclic factor=ATTN_UNROLL

        #pragma HLS ARRAY_PARTITION variable=xb_ffn type=cyclic factor=FFN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=hb type=cyclic factor=FFN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=hb2 type=cyclic factor=FFN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=xq_ffn.q type=cyclic factor=FFN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=xq_ffn.s type=cyclic factor=2 // 使用修正后的固定因子
        #pragma HLS ARRAY_PARTITION variable=hq.q type=cyclic factor=FFN_UNROLL
        #pragma HLS ARRAY_PARTITION variable=hq.s type=cyclic factor=2 // 使用修正后的固定因子
        #pragma HLS ARRAY_PARTITION variable=ffn_output_no_residual type=cyclic factor=FFN_UNROLL


        // ===================================
        // ===== ATTENTION BLOCK LOGIC =====
        // ===================================
        // (将 attention_block 的逻辑复制到这里, 使用 current_x 作为输入)

        // --- Attention RMSNorm --- (输入: current_x, 输出: xb_attn)
        rmsnorm<dim>(xb_attn, current_x, w->rms_att_weight + l * dim); // 使用 l

        // --- QKV Calculation --- (输入: xb_attn, 输出: q, k, v, xq_attn)
        quantize<dim, GS>(&xq_attn, xb_attn, GS); // 使用 l
        matmul<dim, dim>(q, xq_attn.q, xq_attn.s, (w->wq + l)->q, (w->wq + l)->s); // 使用 l
        matmul<dim, kv_dim_local>(k, xq_attn.q, xq_attn.s, (w->wk + l)->q, (w->wk + l)->s); // 使用 l
        matmul<dim, kv_dim_local>(v, xq_attn.q, xq_attn.s, (w->wv + l)->q, (w->wv + l)->s); // 使用 l

        // --- RoPE --- (修改 q, k)
        // ... (将 RoPE 逻辑复制到这里, 使用 l 和 hls_math) ...
         rotation1:
             for (int i = 0; i < kv_dim_local; i += 2) {
                 #pragma HLS UNROLL factor = ATTN_UNROLL
                 #pragma HLS PIPELINE II=1
                 int head_dim_rot = i % head_size_local;
                 if (head_size_local == 0) continue;
                 float freq = 1.0f / powf(10000.0f, (float)head_dim_rot / (float)head_size_local);
                 float val = pos * freq;
                 float fcr = cosf(val); float fci = sinf(val);
                 float v0_q = q[i]; float v1_q = q[i + 1]; q[i] = v0_q * fcr - v1_q * fci; q[i+1] = v0_q * fci + v1_q * fcr;
                 float v0_k = k[i]; float v1_k = k[i + 1]; k[i] = v0_k * fcr - v1_k * fci; k[i+1] = v0_k * fci + v1_k * fcr;
             }


        // --- Update KV Cache --- (输入: k, v, 输出: 写入外部 key_cache, value_cache)
        int loff = l * seq_len * kv_dim_local; // 使用 l
        float *key_cache_row = key_cache + loff + pos * kv_dim_local;
        float *value_cache_row = value_cache + loff + pos * kv_dim_local;
        std::memcpy(key_cache_row, k, kv_dim_local * sizeof(float));
        std::memcpy(value_cache_row, v, kv_dim_local * sizeof(float));

        // --- Multi-Head Attention --- (输入: q, 外部 key_cache, value_cache, 输出: attn_out_unquant)
        // ... (将 MHA 逻辑复制到这里, 使用 l 和 hls_math) ...
         multihead_attention:
             for (int h = 0; h < n_heads; h++) {
                 const int q_offset = h * head_size_local; const int att_offset = h * seq_len;
                 const int kv_head_idx = h; const int kv_head_offset = kv_head_idx * head_size_local;
             iterate_scores:
                 for (int t = 0; t <= pos; t++) {
                     #pragma HLS PIPELINE II=1
                     #pragma HLS loop_tripcount min=1 max=seq_len avg=seq_len/2
                     const float* key_cache_ptr = key_cache + loff + t * kv_dim_local + kv_head_offset;
                     float score = 0.0f;
                 dot_product_qk:
                     for (int i = 0; i < head_size_local; i++) {
                         #pragma HLS UNROLL factor = ATTN_UNROLL
                         score += q[i + q_offset] * key_cache_ptr[i];
                     }
                     if (head_size_local > 0) score /= sqrtf((float)head_size_local); else score = 0.0f;
                     att[t + att_offset] = score;
                 }
                 softmax<seq_len>(att + att_offset, pos + 1);
                 const int out_offset = h * head_size_local;
                 memset(attn_out_unquant + out_offset, 0, head_size_local * sizeof(float));
             accumulate_values:
                 for (int t = 0; t <= pos; t++) {
                     #pragma HLS loop_tripcount min=1 max=seq_len avg=seq_len/2
                     #pragma HLS PIPELINE II=1
                     const float* value_cache_ptr = value_cache + loff + t * kv_dim_local + kv_head_offset;
                     float a = att[t + att_offset];
                 weighted_sum_v:
                     for (int i = 0; i < head_size_local; i++) {
                         #pragma HLS UNROLL factor = ATTN_UNROLL
                         attn_out_unquant[i + out_offset] += a * value_cache_ptr[i];
                     }
                 }
             } // End MHA loop


        // --- Output Projection (wo) --- (输入: attn_out_unquant, 输出: attn_output_no_residual, 使用 xq_attn 作为临时)
        quantize<dim, GS>(&xq_attn, attn_out_unquant, GS); // 使用 l
        matmul<dim, dim>(attn_output_no_residual, xq_attn.q, xq_attn.s, (w->wo + l)->q, (w->wo + l)->s); // 使用 l

        // --- 计算 Attention 块的最终输出 (带残差) ---
        float attn_layer_output[dim]; // 存储 Attention 块结果
        #pragma HLS ARRAY_PARTITION variable=attn_layer_output type=cyclic factor=ATTN_UNROLL // 可选分区
    AddResidualAttn:
        for (int i = 0; i < dim; i++) {
            #pragma HLS UNROLL factor=ATTN_UNROLL skip_exit_check // 匹配因子
            attn_layer_output[i] = current_x[i] + attn_output_no_residual[i];
        }

        // ===================================
        // ===== FFN BLOCK LOGIC =====
        // ===================================
        // (将 ffn_block 的逻辑复制到这里, 使用 attn_layer_output 作为输入)

        // --- FFN RMSNorm --- (输入: attn_layer_output, 输出: xb_ffn)
        rmsnorm<dim>(xb_ffn, attn_layer_output, w_ffn->rms_ffn_weight + l * dim); // 使用 l

        // --- FFN Layers (w1, w3) --- (输入: xb_ffn, 输出: hb, hb2, xq_ffn)
        quantize<dim, GS>(&xq_ffn, xb_ffn, GS); // 使用 l
        matmul<dim, hidden_dim>(hb, xq_ffn.q, xq_ffn.s, (w_ffn->w1 + l)->q, (w_ffn->w1 + l)->s); // 使用 l
        matmul<dim, hidden_dim>(hb2, xq_ffn.q, xq_ffn.s, (w_ffn->w3 + l)->q, (w_ffn->w3 + l)->s); // 使用 l

        // --- SwiGLU Activation --- (修改 hb)
        // ... (将 SwiGLU 逻辑复制到这里, 使用 hls_math) ...
         swi_glu:
             for (int i = 0; i < hidden_dim; i++) {
                 #pragma HLS UNROLL factor = FFN_UNROLL
                 #pragma HLS PIPELINE II=1
                 float val = hb[i];
                 val *= (1.0f / (1.0f + expf(-val)));
                 val *= hb2[i];
                 hb[i] = val;
             }

        // --- FFN Layer (w2) --- (输入: hb, 输出: ffn_output_no_residual, 使用 hq 作为临时)
        quantize<hidden_dim, GS>(&hq, hb, GS); // 使用 l
        matmul<hidden_dim, dim>(ffn_output_no_residual, hq.q, hq.s, (w_ffn->w2 + l)->q, (w_ffn->w2 + l)->s); // 使用 l

        // --- 计算 FFN 块最终输出 (带残差) & 更新 current_x 供下一次迭代 ---
    AddResidualFFN_UpdateX:
        for (int i = 0; i < dim; i++) {
            #pragma HLS UNROLL factor=FFN_UNROLL // 匹配因子
            // FFN 的输入是 attn_layer_output
            current_x[i] = attn_layer_output[i] + ffn_output_no_residual[i];
        }

    } // --- 结束层循环 ---

    // --- 将最后一层的输出写入最终输出流 ---
WriteFinalOutput:
    for (int i = 0; i < dim; ++i) {
    #pragma HLS PIPELINE II=1
        stream_final_out.write(current_x[i]);
    }

} // --- 结束 transformer_layer_pipeline 定义 ---

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