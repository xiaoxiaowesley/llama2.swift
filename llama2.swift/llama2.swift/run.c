//
//  run.c
//  llama2.swift
//
//  Created by xiaoxiang's m1 mbp on 2024/4/29.
//

#include "run.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
//
float * temp_x = NULL;
int dim_sizeof_x = 0;


int* tmp_loff;

float** tmp_xb;

//
float** tmp_wq; // (layer, dim, n_heads * head_size)
float** tmp_wk; // (layer, dim, n_kv_heads * head_size)
float** tmp_wv; // (layer, dim, n_kv_heads * head_size)
float** tmp_wo; // (layer, n_heads * head_size, dim)


float** tmp_q;
float** tmp_k;
float** tmp_v;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;// 🟡
    ptr += n_layers * p->dim;
    w->wq = ptr;// 🟡
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;// 🟡
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;// 🟡
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;// 🟡
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;// 🟡
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// 矩阵 w （维度为 d x n） 和 向量 x (维度为 n x 1) 的乘积
// 出的 xout 是乘积的结果，其维度为 d x 1，即一个 d 行、1 列的列向量
// n 表示向量 x 的长度和矩阵 w 的列数，d 表示矩阵 w 的行数。
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    //打印
//    printf("c[xiaoxiao]dim:%d\n",dim);
//    printf("c[xiaoxiao]kv_dim:%d\n",kv_dim);
//    printf("c[xiaoxiao]kv_mul:%d\n",kv_mul);
//    printf("c[xiaoxiao]hidden_dim:%d\n",hidden_dim);
//    printf("c[xiaoxiao]head_size:%d\n",head_size);



    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    
    // int float_x_size = sizeof(float);
    // int ptr_x_size = sizeof(*x);
    // int dim_ptr_x_size =dim*sizeof(*x);
    
    memcpy(x, content_row, dim*sizeof(*x));

    dim_sizeof_x = dim*sizeof(*x);
    temp_x = malloc(dim*sizeof(*x));
    memcpy(temp_x, x, dim*sizeof(*x));    


    tmp_q = malloc(p->n_layers * sizeof(float*));
    tmp_k = malloc(p->n_layers * sizeof(float*));
    tmp_v = malloc(p->n_layers * sizeof(float*));
    tmp_xb = malloc(p->n_layers * sizeof(float*));

    tmp_wq = malloc(p->n_layers * p->dim * (p->n_heads * head_size)); // (layer, dim, n_heads * head_size)
    tmp_wk = malloc(p->n_layers * p->dim * (p->n_kv_heads * head_size)); // (layer, dim, n_kv_heads * head_size)
    tmp_wv = malloc(p->n_layers * p->dim * (p->n_kv_heads * head_size)); // (layer, dim, n_kv_heads * head_size)
    tmp_wo = malloc(p->n_layers * (p->n_heads * head_size) * p->dim); // (layer, n_heads * head_size, dim)

    
    tmp_loff = malloc(p->n_layers * sizeof(int));    

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        
        /// [1]🟡 ✅
        tmp_xb[l] = malloc(dim*sizeof(float));
        memcpy(tmp_xb[l], s->xb, dim*sizeof(float));
        /// 👈

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        /// [2]🟡 ✅
        tmp_loff[l] = loff;
        /// 👈
        
        /// [3]🟡
        tmp_wq[l] = malloc(p->dim * (p->n_heads * head_size)); // (layer, dim, n_heads * head_size)
        tmp_wk[l] = malloc(p->dim * (p->n_kv_heads * head_size)); // (layer, dim, n_kv_heads * head_size)
        tmp_wv[l] = malloc(p->dim * (p->n_kv_heads * head_size)); // (layer, dim, n_kv_heads * head_size)

        memcpy(tmp_wq[l], w->wq + l*dim*dim, dim*dim*sizeof(float));
        memcpy(tmp_wk[l], w->wk + l*dim*kv_dim, dim*kv_dim*sizeof(float));
        memcpy(tmp_wv[l], w->wv + l*dim*kv_dim, dim*kv_dim*sizeof(float));
        /// 👈
        

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        //打印 w->wq[24576]
        printf("c[xiaoxiao]w->wq[24575]:%f\n",w->wq[24575]);
        printf("c[xiaoxiao]w->wq[24576]:%f\n",w->wq[24576]);
        printf("c[xiaoxiao]w->wq[24577]:%f\n",w->wq[24577]);
        
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        
       
        /// [4]🟡
        tmp_q[l] = malloc(dim*sizeof(float));
        tmp_k[l] = malloc(kv_dim*sizeof(float));
        tmp_v[l] = malloc(kv_dim*sizeof(float));
        memcpy(tmp_q[l], s->q, dim*sizeof(float));
        memcpy(tmp_k[l], s->k, kv_dim*sizeof(float));
        memcpy(tmp_v[l], s->v, kv_dim*sizeof(float));
        /// 👈

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}


