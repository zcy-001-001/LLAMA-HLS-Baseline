/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include "typedefs.h"
#include "forward.h"
#include "config.h"
#include <vector>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <string>
#include <cstdlib> // 关键：包含这个头文件以使用 getenv
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Globals

// void malloc_run_state(RunState *s, Config *p)
// {
//   // we calloc instead of malloc to keep valgrind happy
//   int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
//   s->x = (float *)calloc(p->dim, sizeof(float));
//   s->xb = (float *)calloc(p->dim, sizeof(float));
//   s->xb2 = (float *)calloc(p->dim, sizeof(float));
//   s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
//   s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
//   s->xq = (QuantizedTensor){.q = (int8_t *)calloc(p->dim, sizeof(int8_t)), .s = (float *)calloc(p->dim, sizeof(float))};
//   s->hq = (QuantizedTensor){.q = (int8_t *)calloc(p->hidden_dim, sizeof(int8_t)), .s = (float *)calloc(p->hidden_dim, sizeof(float))};
//   s->q = (float *)calloc(p->dim, sizeof(float));
//   s->k = (float *)calloc(kv_dim, sizeof(float));
//   s->v = (float *)calloc(kv_dim, sizeof(float));
//   s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
//   s->logits = (float *)calloc(p->vocab_size, sizeof(float));
//   s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   // ensure all mallocs went fine
//   if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache)
//   {
//     fprintf(stderr, "malloc failed!\n");
//     exit(EXIT_FAILURE);
//   }
// }

void softmax(float *x, int size)
{
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++)
  {
    if (x[i] > max_val)
    {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++)
  {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++)
  {
    x[i] /= sum;
  }
}

// --- CORRECTED init_quantized_tensors ---
// Takes GROUP_SIZE as template param, uses group_size func param internally
template <int SIZE, int GROUP_SIZE> // Added GROUP_SIZE template param
void init_quantized_tensors(void **ptr,
                            QuantizedTensor<SIZE, GROUP_SIZE> *tensor, // Use corrected type
                            int n,           // Number of tensors in the array
                            int size_each,   // Number of elements (like dim*dim) in ONE tensor's q array
                            int group_size)  // Explicit group size parameter
{
    // Basic check
    if (group_size <= 0 || size_each % group_size != 0) {
        fprintf(stderr, "Error in init_quantized_tensors: Invalid group_size %d or size_each %d not divisible by it.\n", group_size, size_each);
        // Optionally handle error, e.g., by setting tensor data to zero or exiting
        // For now, just return to avoid crashing, but data will be uninitialized/wrong.
        return;
    }

    void *p = *ptr;
    const int scale_count_each = size_each / group_size; // Number of scales per tensor

    for (int i = 0; i < n; i++) {
        // Copy quantized values (q)
        std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
        p = (int8_t *)p + size_each; // Advance pointer past q data

        // Copy scale factors (s) - size is scale_count_each
        std::memcpy(tensor[i].s, p, scale_count_each * sizeof(float));
        p = (float *)p + scale_count_each; // Advance pointer past s data
    }
    *ptr = p; // Update the original pointer
}

// --- UPDATED memory_map_weights ---
// Uses distinct template name group_size_gs, passes it to helpers
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // Use distinct template name for GS
bool memory_map_weights(
    // Use the updated TransformerWeights type which expects GS
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                       vocab_size, seq_len, group_size_gs> *w,
    void *ptr, uint8_t shared_classifier)
{
    static_assert(group_size_gs > 0, "Group size must be positive");
    int head_size = dim / n_heads;
    // Ensure kv_dim calculation matches typedefs.h if used implicitly there
    constexpr int kv_dim_calc = (dim * n_kv_heads) / n_heads;

    // Map float weights (RMSNorm)
    float *fptr = (float *)ptr;
    std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float)); fptr += n_layers * dim;
    std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float)); fptr += n_layers * dim;
    std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));        fptr += dim;

    ptr = (void *)fptr; // Update base pointer past float weights

    // Map Quantized Tensors using the corrected init function
    // Pass group_size_gs explicitly
    init_quantized_tensors<vocab_size * dim, group_size_gs>(&ptr, w->q_tokens, 1, vocab_size * dim, group_size_gs);

    // Dequantize token embeddings (call the function defined in forward.h)
    // Pass group_size_gs explicitly
    // Note: q_tokens is array size 1, so access [0]
    dequantize<vocab_size * dim, group_size_gs>(&(w->q_tokens[0]), w->token_embedding_table, group_size_gs);

    // Map Attention weights
    init_quantized_tensors<dim * dim, group_size_gs>         (&ptr, w->wq, n_layers, dim * dim, group_size_gs);
    init_quantized_tensors<kv_dim_calc * dim, group_size_gs> (&ptr, w->wk, n_layers, kv_dim_calc * dim, group_size_gs);
    init_quantized_tensors<kv_dim_calc * dim, group_size_gs> (&ptr, w->wv, n_layers, kv_dim_calc * dim, group_size_gs);
    init_quantized_tensors<dim * dim, group_size_gs>         (&ptr, w->wo, n_layers, dim * dim, group_size_gs);

    // Map FFN weights
    init_quantized_tensors<hidden_dim * dim, group_size_gs>  (&ptr, w->w1, n_layers, hidden_dim * dim, group_size_gs);
    init_quantized_tensors<dim * hidden_dim, group_size_gs>  (&ptr, w->w2, n_layers, dim * hidden_dim, group_size_gs);
    init_quantized_tensors<hidden_dim * dim, group_size_gs>  (&ptr, w->w3, n_layers, hidden_dim * dim, group_size_gs);

    // Map Classifier weights
    if (shared_classifier) {
        // Simply copy the pointer/data structure if weights are shared
        // Ensure the QuantizedTensor struct allows copying if needed, or just copy data.
        std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim, group_size_gs>));
    } else {
        init_quantized_tensors<vocab_size * dim, group_size_gs>(&ptr, w->wcls, 1, vocab_size * dim, group_size_gs);
    }
    return true;
}


// --- UPDATED read_checkpoint ---
// Uses distinct template name group_size_gs
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // Use distinct template name for GS
bool read_checkpoint(
    std::string checkpoint, Config *config_out,
    // Use the updated TransformerWeights type which expects GS
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                       vocab_size, seq_len, group_size_gs> *weights)
{
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str()); return false; }

    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { fclose(file); return false; }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); fclose(file); return false; }

    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { fclose(file); return false; }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); fclose(file); return false; }

    // Read config fields individually to avoid potential struct packing issues
    // Assuming Config struct layout matches file header:
    if (fread(&config_out->dim, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->hidden_dim, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->n_layers, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->n_heads, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->n_kv_heads, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->vocab_size, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->seq_len, sizeof(int), 1, file) != 1) return false;
    // Skip kv_dim, as it should be calculated

    uint8_t shared_classifier;
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { fclose(file); return false; }

    int group_size_read;
    if (fread(&group_size_read, sizeof(int), 1, file) != 1) { fclose(file); return false; }

    // --- CRITICAL: Ensure consistency ---
    // Check if group size read from file matches the compile-time template parameter
    if (group_size_read != group_size_gs) {
        fprintf(stderr, "ERROR: Group size mismatch! Checkpoint file has GS=%d, but compiled with GS=%d\n",
                group_size_read, group_size_gs);
        fclose(file);
        return false;
    }
    // Assign the validated group size to the config struct
    config_out->GS = group_size_gs; // Or group_size_read, they are the same now
    // Calculate kv_dim based on other config values
    config_out->kv_dim = (config_out->dim * config_out->n_kv_heads) / config_out->n_heads;


    // Seek past the rest of the header (assuming header size is fixed or read previously)
    int header_size = 28; // Minimal size: magic, version, 7 ints, shared_flag, group_size = 4+4+7*4+1+4 = 41 bytes? Or is it fixed 256?
                          // Using fixed 256 based on original comment structure, adjust if needed.
    header_size = 256;
    fseek(file, header_size, SEEK_SET); // Seek to end of header

    // Memory map the rest of the file
    long current_pos = ftell(file);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    long weights_size = file_size - current_pos;
    fclose(file); // Close file now, mmap uses fd

    // Use mmap (consider error checking)
    int fd = open(checkpoint.c_str(), O_RDONLY);
    if (fd == -1) { fprintf(stderr, "open failed for mmap!\n"); return false; }
    // Map only the weights part
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0); // Map whole file for simplicity
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); close(fd); return false; }
    close(fd); // Close fd after mmap

    // Point to the start of weights data in the mapped region
    void *weights_ptr = ((char *)data) + header_size;

    // Call memory_map_weights with the correct pointer and validated group size
    if (!memory_map_weights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                           vocab_size, seq_len, group_size_gs>(weights, weights_ptr, shared_classifier))
    {
        munmap(data, file_size); // Unmap on error
        return false;
    }

    // Unmap memory after weights are presumably copied or used directly if pointers are stored
    // Assuming memory_map_weights copies the data structure contents
    munmap(data, file_size);

    return true;
}


// --- UPDATED build_transformer ---
// Passes group_size_gs template parameter correctly
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // Use distinct template name for GS
bool build_transformer(
    // Use updated Transformer type
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                vocab_size, seq_len, group_size_gs> *t,
    std::string checkpoint_path)
{
    // Call read_checkpoint which now also takes group_size_gs
    return read_checkpoint<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                           vocab_size, seq_len, group_size_gs>(checkpoint_path, &t->config, &t->weights);
}


// ----------------------------------------------------------------------------

typedef struct { char *str; int id; } TokenIndex;
typedef struct { char **vocab; float *vocab_scores; TokenIndex *sorted_vocab; int vocab_size; unsigned int max_token_length; unsigned char byte_pieces[512]; } Tokenizer;
int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); }
bool build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size) {/* ... implementation ... */
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // Initialize to NULL for safety
    FILE *file = fopen(tokenizer_path.c_str(), "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str()); free(t->vocab); free(t->vocab_scores); return false; }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); fclose(file); free(t->vocab); free(t->vocab_scores); return false; }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { /* error handling */ fclose(file); /* free */ return false; }
        if (fread(&len, sizeof(int), 1, file) != 1) { /* error handling */ fclose(file); /* free */ return false; }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { /* error handling */ fclose(file); /* free */ return false; }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    t->sorted_vocab = (TokenIndex *)malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; }
    qsort(t->sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    for (int i = 0; i < 256; i++) { t->byte_pieces[i * 2] = (unsigned char)i; t->byte_pieces[i * 2 + 1] = '\0'; }
    return true;
 }
 void free_tokenizer(Tokenizer *t) { if (!t) return; for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); } free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab); }

 char* decode(Tokenizer *t, int prev_token, int token) { if (token < 0 || token >= t->vocab_size) return NULL; char *piece = t->vocab[token]; if (prev_token == 1 && piece[0] == ' ') { piece++; } return piece; }
 void safe_printf(char *piece) { if (piece == NULL || piece[0] == '\0') { return; } if (piece[1] == '\0') { unsigned char byte_val = piece[0]; if (!(isprint(byte_val) || byte_val == '\n')) { printf("?"); } else { printf("%c", byte_val); } } else { printf("%s", piece); } }
 int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) { if (!str || !sorted_vocab) return -1; TokenIndex tok = {.str = str}; TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens); return res != NULL ? res->id : -1; }
 void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) { /* ... implementation ... */
     if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
     size_t text_len = strlen(text);
     int *str_buffer = (int *)malloc((text_len + 1) * sizeof(int)); // Use size_t for strlen result
     if (!str_buffer) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
     int str_len = 0;
     if (bos) tokens[(*n_tokens)++] = 1; // Assuming BOS token ID is 1
     for (size_t i = 0; i < text_len; ++i) { // Use size_t for loop
         str_buffer[str_len++] = (unsigned char)(text[i]); // Store bytes directly
     }
     while (1) {
         float best_score = -1e10; int best_id = -1; int best_idx = -1;
         for (int i=0; i < str_len - 1; i++) {
             char merge_candidate[t->max_token_length * 2 + 1]; // Ensure buffer is safe based on max_token_length
             char* piece1 = (str_buffer[i] < 256) ? (char*)t->byte_pieces + str_buffer[i] * 2 : t->vocab[str_buffer[i]];
             char* piece2 = (str_buffer[i+1] < 256) ? (char*)t->byte_pieces + str_buffer[i+1] * 2 : t->vocab[str_buffer[i+1]];
              // Check lengths before snprintf to prevent buffer overflow
             if (strlen(piece1) + strlen(piece2) < sizeof(merge_candidate)) {
                snprintf(merge_candidate, sizeof(merge_candidate), "%s%s", piece1, piece2);
                 int id = str_lookup(merge_candidate, t->sorted_vocab, t->vocab_size);
                 if (id != -1 && t->vocab_scores[id] > best_score) { best_score = t->vocab_scores[id]; best_id = id; best_idx = i; }
             } // else: handle case where merged token is too long (optional)
         }
         if (best_idx == -1) break;
         str_buffer[best_idx] = best_id;
         for (int i = best_idx+1; i < str_len-1; i++) { str_buffer[i] = str_buffer[i+1]; }
         str_len--;
     }
     for (int i=0; i < str_len; i++) { tokens[(*n_tokens)++] = str_buffer[i]; }
     free(str_buffer);
     if (eos) tokens[(*n_tokens)++] = 2; // Assuming EOS token ID is 2
 }

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

// Sampler struct and functions (assuming no changes needed)
// ... (Sampler code as provided before) ...
typedef struct { float prob; int index; } ProbIndex;
typedef struct { int vocab_size; ProbIndex *probindex; float temperature; float topp; unsigned long long rng_state; } Sampler;
int sample_argmax(float *probabilities, int n) { int max_i = 0; float max_p = probabilities[0]; for (int i = 1; i < n; i++) { if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; } } return max_i; }
int sample_mult(float *probabilities, int n, float coin) { float cdf = 0.0f; for (int i = 0; i < n; i++) { cdf += probabilities[i]; if (coin < cdf) return i; } return n - 1; }
int compare(const void *a, const void *b) { ProbIndex *a_ = (ProbIndex *)a; ProbIndex *b_ = (ProbIndex *)b; if (a_->prob > b_->prob) return -1; if (a_->prob < b_->prob) return 1; return 0; }
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) { int n0 = 0; for (int i = 0; i < n; i++) { if (probabilities[i] > 0) { probindex[n0].index = i; probindex[n0].prob = probabilities[i]; n0++; } } if (n0 == 0) return 0; qsort(probindex, n0, sizeof(ProbIndex), compare); float cumulative_prob = 0.0f; int last_idx = n0 - 1; for (int i = 0; i < n0; i++) { cumulative_prob += probindex[i].prob; if (cumulative_prob > topp) { last_idx = i; break; } } float r = coin * cumulative_prob; float cdf = 0.0f; for (int i = 0; i <= last_idx; i++) { cdf += probindex[i].prob; if (r < cdf) return probindex[i].index; } return probindex[last_idx].index; }
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) { sampler->vocab_size = vocab_size; sampler->temperature = temperature; sampler->topp = topp; sampler->rng_state = rng_seed; sampler->probindex = (ProbIndex *)malloc(vocab_size * sizeof(ProbIndex)); if (!sampler->probindex) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); } }
void free_sampler(Sampler *sampler) { if (sampler) free(sampler->probindex); }
unsigned int random_u32(unsigned long long *state) { *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27; return (*state * 0x2545F4914F6CDD1Dull) >> 32; }
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }
int sample(Sampler *sampler, float *logits) { int next; if (sampler->temperature == 0.0f) { next = sample_argmax(logits, sampler->vocab_size); } else { for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; } softmax(logits, sampler->vocab_size); float coin = random_f32(&sampler->rng_state); if (sampler->topp <= 0 || sampler->topp >= 1) { next = sample_mult(logits, sampler->vocab_size, coin); } else { next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin); } } return next; }

// Utilities: time (assuming no changes needed)
long time_in_ms() { struct timespec time; clock_gettime(CLOCK_REALTIME, &time); return time.tv_sec * 1000 + time.tv_nsec / 1000000; }


// ----------------------------------------------------------------------------
// generation loop
// --- UPDATED generate function ---
// Passes group_size_gs template parameter correctly
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // Use distinct template name for GS
void generate(
    // Use updated Transformer type
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                vocab_size, seq_len, group_size_gs> *transformer,
    Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps,auto kernelpath)
{
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = (char*)empty_prompt; }

    int num_prompt_tokens = 0;
    // Allocate slightly more space potentially needed for BOS/EOS
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    if (!prompt_tokens) { fprintf(stderr, "malloc failed for prompt_tokens\n"); exit(EXIT_FAILURE); }

    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    std::cout << "Encoded prompt into " << num_prompt_tokens << " tokens." << std::endl;
    if (num_prompt_tokens < 1) { fprintf(stderr, "Error: encode() generated no tokens.\n"); free(prompt_tokens); exit(EXIT_FAILURE);} // Exit if encoding fails

    // Allocate token buffer for prompt + generation steps
    int* tokens = (int*)malloc((num_prompt_tokens + steps) * sizeof(int));
    if (!tokens) { fprintf(stderr, "malloc failed for tokens\n"); free(prompt_tokens); exit(EXIT_FAILURE); }
    memcpy(tokens, prompt_tokens, num_prompt_tokens * sizeof(int));
    free(prompt_tokens); // Free temporary prompt buffer

  const char* basePathCStr = getenv("MODEL_BASE_PATH");
  std::string model_base_path(basePathCStr);
  std::string xclbin_path =model_base_path+ "/sw_emu/forward_sw_emu.xclbin"; // Default for CSIM
  
  std::cout << "Loading kernel..." << std::endl;
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(xclbin_path);
  uuid = device.load_xclbin(kernelpath);

  
  
  
  auto initial_embedding_lookup_kernel = xrt::kernel(device, uuid, "initial_embedding_lookup");
  auto transformer_layer_pipeline_kernel = xrt::kernel(device, uuid, "transformer_layer_pipeline"); 
  auto final_norm_classifier_kernel = xrt::kernel(device, uuid, "final_norm_classifier");


  std::cout << "Allocating  buffer" << std::endl;
 
  auto token_embedding_table=xrt::bo(device, sizeof(transformer->weights.token_embedding_table), initial_embedding_lookup_kernel.group_id(0));

  
  size_t cache_dim = n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
  auto w_buffer = xrt::bo(device, sizeof(transformer->weights), transformer_layer_pipeline_kernel.group_id(2));
  auto w1_buffer = xrt::bo(device, sizeof(transformer->weights), transformer_layer_pipeline_kernel.group_id(3));
  auto key_buffer = xrt::bo(device, cache_dim * sizeof(float), transformer_layer_pipeline_kernel.group_id(5));
  auto value_buffer = xrt::bo(device, cache_dim * sizeof(float), transformer_layer_pipeline_kernel.group_id(6));


  auto logits_out = xrt::bo(device, vocab_size * sizeof(float), final_norm_classifier_kernel.group_id(1));
  auto w2_buffer  = xrt::bo(device, sizeof(transformer->weights), final_norm_classifier_kernel.group_id(2));

  std::cout << "Allocating  buffer completing"<< std::endl;

  std::cout << "Copying data to buffer" << std::endl;
  std::cout << "编码表大小: " << sizeof(transformer->weights.token_embedding_table) << std::endl;
  std::cout << "权重大小: " << sizeof(transformer->weights) << std::endl;
  std::cout << "kvcache大小: " << cache_dim * sizeof(float) << std::endl;

  
  std::cout << "写入数据&&开始同步" << std::endl;
  w_buffer.write(&transformer->weights, sizeof(transformer->weights), 0);
  w1_buffer.write(&transformer->weights, sizeof(transformer->weights), 0);
  w2_buffer.write(&transformer->weights, sizeof(transformer->weights), 0);
  token_embedding_table.write(&transformer->weights.token_embedding_table, sizeof(transformer->weights.token_embedding_table), 0);
  
  w_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  w1_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  w2_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  token_embedding_table.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "写入结束&&同步结束" << std::endl;
 
  int pos = 0;                  // position in the sequence
  int next_token; // 用于存储第一个生成步骤所需的 token

  // first run
  for (int t = 0; t < num_prompt_tokens; ++t) {
    int current_token = tokens[t];
    std::cout << "current_token: " << current_token << std::endl;
    float *logits = (float *)malloc(vocab_size * sizeof(float));
    std::cout << "Processing prompt token " << t << "/" << num_prompt_tokens << " (pos=" << pos << ")" << std::endl; // Debug
    //auto run = initial_embedding_lookup_kernel(token_embedding_table, current_token);
    auto run1 = xrt::run(initial_embedding_lookup_kernel);
    run1.set_arg(0, token_embedding_table);
    run1.set_arg(1, current_token);
    
    //std::cout << "第一个kernal开始执行" << std::endl;
    // run1.wait();
    // std::cout << "第一个kernal执行结束" << std::endl;

    //auto run1 = transformer_layer_pipeline_kernel(w_buffer ,w1_buffer,pos, key_buffer, value_buffer);
    auto run2 = xrt::run(transformer_layer_pipeline_kernel);
    run2.set_arg(2, w_buffer);
    run2.set_arg(3, w1_buffer);
    run2.set_arg(4, pos);
    run2.set_arg(5, key_buffer);
    run2.set_arg(6, value_buffer);
    
    //std::cout << "第二个kernal开始执行" << std::endl;
    // run2.wait();
    // std::cout << "第二个kernal执行结束" << std::endl;
   // auto run2 = final_norm_classifier_kernel(logits_out,w2_buffer);
   auto run3 = xrt::run(final_norm_classifier_kernel);
   run3.set_arg(1, logits_out);
   run3.set_arg(2, w2_buffer);

   run1.start();
   run2.start();
   run3.start();
   run3.wait();
   run2.wait();
   run1.wait();
   logits_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
   logits_out.read(logits, vocab_size * sizeof(float), 0);
   if (t == num_prompt_tokens - 1) {
        next_token = sample(sampler, logits);
   }
    pos++; // 处理完一个 token，位置加一
} // End of prompt processing loop
long start = time_in_ms(); // Start timer after prompt processing
std::cout << "\nPrompt: " <<prompt<< std::endl;
std::cout << "Starting generation..." << std::endl;
std::cout << "Answers: " << std::endl;


// --- Generate New Tokens (同样使用新的 Kernel 调用流程) ---
for (int t = 0; t < steps; ++t) {

    // 使用上一步采样的 token 作为当前输入
    int current_token = next_token;
    tokens[num_prompt_tokens + t] = current_token; // Store generated token
    // --- 解码和打印 ---
    int prev_token = (num_prompt_tokens + t > 0) ? tokens[num_prompt_tokens + t - 1] : 1; // BOS if first token
    char* piece = decode(tokenizer, prev_token, current_token);
    safe_printf(piece);
    fflush(stdout);
    // --- 如果生成 EOS token 则停止 ---
    if (current_token == 2) { // Assuming EOS token ID is 2
         printf("\n[EOS]");
         break;
    }
    // --- 检查序列长度是否超出限制 ---
    if (pos >= seq_len) {
        printf("\n[SEQUENCE LENGTH LIMIT REACHED]\n");
        break;
    }
    float *logits = (float *)malloc(vocab_size * sizeof(float));
    auto run1 = xrt::run(initial_embedding_lookup_kernel);
    run1.set_arg(0, token_embedding_table);
    run1.set_arg(1, current_token);

    auto run2 = xrt::run(transformer_layer_pipeline_kernel);
    run2.set_arg(2, w_buffer);
    run2.set_arg(3, w1_buffer);
    run2.set_arg(4, pos);
    run2.set_arg(5, key_buffer);
    run2.set_arg(6, value_buffer);

    auto run3 = xrt::run(final_norm_classifier_kernel);
    run3.set_arg(1, logits_out);
    run3.set_arg(2, w2_buffer);
 

    run1.start();
    run2.start();
    run3.start();
    run3.wait();;
    run2.wait();
    run1.wait();
    logits_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    logits_out.read(logits, vocab_size * sizeof(float), 0);
    // --- 采样下一个 token ---
    next_token = sample(sampler, logits); // 使用修正后的调用
    pos++; // 位置加一，为处理下一个生成的 token 做准备
} // End of generation loop
    printf("\n");

    long end = time_in_ms();
    if (pos > num_prompt_tokens) { // Only report if actual generation happened
        long elapsed_ms = end - start;
        int generated_tokens = pos - num_prompt_tokens;
         if (generated_tokens > 0) {
             float tokens_per_sec = (float)generated_tokens / (elapsed_ms / 1000.0f);
             std::cout << "--------------------------------\n";
             std::cout << "Generated " << generated_tokens << " tokens in " << elapsed_ms << " ms" << std::endl;
             std::cout << "Tokens per second: " << tokens_per_sec << std::endl;
         }
    } else {
         std::cout << "No new tokens generated.\n";
    }


    // // Cleanup
    free(prompt_tokens);
    // free(tokens);
    // free(logits_out);
    // free(key_buffer);
    // free(value_buffer);
    std::cout << "Generation completed successfully." << std::endl;
}


// --- Utilities: read_stdin (assuming no changes needed) ---
void read_stdin(const char *guide, char *buffer, size_t bufsize) { printf("%s", guide); if (fgets(buffer, bufsize, stdin) != NULL) { size_t len = strlen(buffer); if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; } } }


// ----------------------------------------------------------------------------
// CLI, include only if not testing
void error_usage()
{
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    std::cout << "Start - Testbench for Split Kernels (Quantized)" << std::endl;

    const char* basePathCStr = getenv("MODEL_BASE_PATH");
    std::string model_base_path(basePathCStr);
    std::string checkpoint_path =model_base_path+ "/weights.bin"; // Default for CSIM
    std::string tokenizer_path = model_base_path+"/tokenizer.bin"; // Default for CSIM
    float temperature = 0.0f; // Default to argmax (deterministic) for testing
    float topp = 0.9f;      // Top-p (not used if temp=0)
    int steps = seq_len;    // Default steps = max sequence length
    char *prompt = NULL;    // Default prompt is NULL (empty -> starts with BOS)
    unsigned long long rng_seed = 1234; // Fixed seed for deterministic testing
    const char *mode = "generate"; // Default mode
    char *system_prompt = NULL; // Default system prompt
    std::string kernelpath = "";


    // --- Argument Parsing (CSIM Friendly) ---
     if (argc >= 2) {
         checkpoint_path = argv[1];
         std::cout << "INFO: Using checkpoint path from argv[1]: " << checkpoint_path << std::endl;
         for (int i = 2; i < argc; i += 2) {
             if (i + 1 >= argc) { std::cerr << "WARN: Option " << argv[i] << " requires a value." << std::endl; break; }
             if (argv[i][0] != '-' || strlen(argv[i]) != 2) { std::cerr << "WARN: Invalid option format '" << argv[i] << "'. Skipping." << std::endl; continue; }
             switch (argv[i][1]) {
                 case 't': temperature = atof(argv[i + 1]); break;
                 case 'p': topp = atof(argv[i + 1]); break;
                 case 's': rng_seed = strtoull(argv[i + 1], NULL, 10); break;
                 case 'n': steps = atoi(argv[i + 1]); break;
                 case 'i': prompt = argv[i + 1]; break;
                 case 'z': tokenizer_path = argv[i + 1]; break;
                 case 'm': mode = argv[i + 1]; break;
                 case 'y': system_prompt = argv[i + 1]; break;
                 case 'k': kernelpath = argv[i + 1];break;
                 default: std::cerr << "WARN: Unknown option '" << argv[i] << "'. Ignoring." << std::endl; break;
             }
         }
     } else {
         std::cout << "INFO: Not enough command line arguments provided (argc=" << argc << "). Using default paths/params for CSIM." << std::endl;
     }


    // --- Parameter Validation/Setup ---
    if (rng_seed == 0) { rng_seed = (unsigned long long)time(NULL); }
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps <= 0 || steps > seq_len) steps = seq_len; // Ensure steps is valid

    // --- Build Transformer ---
    // Instantiate using constants from config.h (including GS)
    static Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                       vocab_size, seq_len, GS> // Use global GS from config.h
        transformer;

    // Call build_transformer which uses the global GS via template arg
    bool weights_loaded = build_transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                                            vocab_size, seq_len, GS>(&transformer, checkpoint_path);
    if (!weights_loaded) {
        fprintf(stderr, "ERROR: failed to load checkpoint weights from %s\n", checkpoint_path.c_str());
        return 1;
    }
     // Check if loaded config GS matches compile-time GS
     if (transformer.config.GS != GS) {
          fprintf(stderr, "ERROR: Config group size (%d) does not match compiled group size (%d)!\n", transformer.config.GS, GS);
          return 1;
     }


    // --- Build Tokenizer ---
    Tokenizer tokenizer;
    bool tokenizer_loaded = build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    if (!tokenizer_loaded) {
        fprintf(stderr, "ERROR: failed to load tokenizer from %s\n", tokenizer_path.c_str());
        return 1;
    }

    // --- Build Sampler ---
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // --- Run Generation ---
    if (strcmp(mode, "generate") == 0) {
        // Call generate which uses the global GS via template arg
        generate(&transformer, &tokenizer, &sampler, prompt, steps, kernelpath);
    } else {
        fprintf(stderr, "ERROR: unknown mode: %s\n", mode);
        free_sampler(&sampler);
        free_tokenizer(&tokenizer);
        return 1;
    }

    // --- Cleanup ---
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);

    std::cout << "INFO: Testbench main finished successfully." << std::endl;
    return 0;
}
