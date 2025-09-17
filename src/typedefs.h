#ifndef TYPEDEFS_H // It's good practice to ensure header guards match filename
#define TYPEDEFS_H

#include <stdint.h>
#include <stdio.h>
// Include config.h here if GS is needed globally, or ensure it's included before typedefs.h
#include "config.h" // Provides dim, hidden_dim, ..., GS

// TODO: replace with HLS types if necessary later

//===========================================================================
//  typedefs.h
//===========================================================================
//  @brief: Defines core data structures for the transformer model.

// Configuration structure remains the same
struct Config
{
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size
    int seq_len;    // max sequence length
    int GS;         // group size for quantization
    int kv_dim;     // Dimension of key/value vectors
};

// --- CORRECTED QuantizedTensor definition ---
// Now takes GROUP_SIZE as a template parameter
template <int SIZE, int GROUP_SIZE>
struct QuantizedTensor
{
    // Compile-time check: SIZE must be divisible by GROUP_SIZE
    static_assert(SIZE > 0, "QuantizedTensor SIZE must be positive");
    static_assert(GROUP_SIZE > 0, "QuantizedTensor GROUP_SIZE must be positive");
    static_assert(SIZE % GROUP_SIZE == 0, "QuantizedTensor SIZE must be divisible by GROUP_SIZE");

    int8_t q[SIZE];             // quantized values
    float s[SIZE / GROUP_SIZE]; // scaling factors (one per group) - CORRECTED SIZE
};

// --- UPDATED RunState to use corrected QuantizedTensor ---
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct RunState
{
    // current wave of activations
    float x[dim];         // activation at current time stamp (dim,)
    float xb[dim];        // same, but inside a residual branch (dim,)
    float xb2[dim];       // an additional buffer just for convenience (dim,)
    float hb[hidden_dim]; // buffer for hidden dimension in the ffn (hidden_dim,)
    float hb2[hidden_dim];// buffer for hidden dimension in the ffn (hidden_dim,)

    // Pass GS as the GROUP_SIZE template parameter
    QuantizedTensor<dim, GS> xq;        // quantized x (dim,)
    QuantizedTensor<hidden_dim, GS> hq; // quantized hb (hidden_dim,)

    float q[dim];                     // query (dim,)
    // Use kv_dim directly from config (assuming it's pre-calculated)
    static constexpr int kv_dim_calc = (dim * n_kv_heads) / n_heads; // Ensure consistency
    float k[kv_dim_calc];             // key (kv_dim,)
    float v[kv_dim_calc];             // value (kv_dim,)

    float att[n_heads * seq_len];     // buffer for scores/attention values (n_heads, seq_len)

    // kv cache
    float key_cache[n_layers * seq_len * kv_dim_calc];   // (layer, seq_len, kv_dim)
    float value_cache[n_layers * seq_len * kv_dim_calc]; // (layer, seq_len, kv_dim)
};

// --- UPDATED TransformerWeights to use corrected QuantizedTensor ---
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct TransformerWeights
{
    // Calculate kv_dim locally for template instantiation if needed, or assume it's passed/global
    static constexpr int kv_dim_calc = (dim * n_kv_heads) / n_heads;

    // token embedding table
    // Pass GS as the GROUP_SIZE template parameter
    QuantizedTensor<vocab_size * dim, GS> q_tokens[1]; // (vocab_size, dim) - Size 1 array? Check usage.
    float token_embedding_table[vocab_size * dim];     // same, but dequantized

    // weights for rmsnorms (float, no change needed here)
    float rms_att_weight[n_layers * dim]; // (layer, dim) rmsnorm weights
    float rms_ffn_weight[n_layers * dim]; // (layer, dim)

    // weights for matmuls. Pass GS as GROUP_SIZE template parameter
    // Note: SIZE calculation based on OutputDim * InputDim appears correct.
    QuantizedTensor<dim * dim, GS>          wq[n_layers]; // (layer, dim, dim)
    QuantizedTensor<kv_dim_calc * dim, GS>  wk[n_layers]; // (layer, kv_dim, dim)
    QuantizedTensor<kv_dim_calc * dim, GS>  wv[n_layers]; // (layer, kv_dim, dim)
    QuantizedTensor<dim * dim, GS>          wo[n_layers]; // (layer, dim, dim)

    // weights for ffn. Pass GS as GROUP_SIZE template parameter
    QuantizedTensor<hidden_dim * dim, GS>   w1[n_layers]; // (layer, hidden_dim, dim)
    QuantizedTensor<dim * hidden_dim, GS>   w2[n_layers]; // (layer, dim, hidden_dim)
    QuantizedTensor<hidden_dim * dim, GS>   w3[n_layers]; // (layer, hidden_dim, dim)

    // final rmsnorm (float, no change needed here)
    float rms_final_weight[dim]; // (dim,)

    // classifier weights. Pass GS as GROUP_SIZE template parameter
    QuantizedTensor<vocab_size * dim, GS>   wcls[1]; // Size 1 array? Check usage.
};

// ----------------------------------------------------------------------------
// Transformer model structure (no changes needed here)
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct Transformer
{
    Config config;                                                                                      // hyperparameters
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> weights; // weights
    RunState<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> state;           // activations
    // Removed memory mapping related fields (fd, data, file_size) as they are likely C specific
};

#endif // TYPEDEFS_H