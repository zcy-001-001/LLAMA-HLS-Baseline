#pragma once
#include "typedefs.h"

static constexpr int dim = 4096;
static constexpr int hidden_dim = 11008;
static constexpr int n_layers = 32;
static constexpr int n_heads = 32;
static constexpr int n_kv_heads = 32;
static constexpr int vocab_size = 32000;
static constexpr int seq_len = 128;
static constexpr int GS = 64;
static constexpr int kv_dim = (dim * n_kv_heads) / n_heads;

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
constexpr Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .vocab_size = vocab_size,
    .seq_len = seq_len,
    .GS = GS,
    .kv_dim = kv_dim,
};