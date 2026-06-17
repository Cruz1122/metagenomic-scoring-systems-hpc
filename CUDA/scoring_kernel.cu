/**
 * scoring_kernel.cu — Búsqueda de pesos óptimos en GPU con CUDA C.
 *
 * Estrategias: random, grid, hybrid
 * Modos: full (P=profiles@W, scores=A@P), precompute (scores=B@W, B=A@profiles)
 * Precisión: float32 por defecto, float64 con -DUSE_DOUBLE
 *
 * Compilar:
 *   nvcc -O3 -std=c++17 -arch=sm_75 scoring_kernel.cu -o scoring_cuda
 *
 * Ejecutar:
 *   ./scoring_cuda --k 10000 --seed 42 --search random --data-dir data/npy
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

#define MAX_SAMPLES 1024
#define MAX_ITEMS 1000000

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(2); \
    } \
} while(0)

/* =================================================================== */
/*  NPY Reader                                                          */
/* =================================================================== */

static void read_npy_float32(const char* path, std::vector<float>& data, int& d0, int& d1) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Error: cannot open %s\n", path); exit(1); }

    char magic[6]; f.read(magic, 6);
    if (magic[0]!=(char)0x93||magic[1]!='N'||magic[2]!='U'||magic[3]!='M'||magic[4]!='P'||magic[5]!='Y')
        { fprintf(stderr, "Error: %s not a .npy file\n", path); exit(1); }

    uint8_t vmaj, vmin; f.read((char*)&vmaj,1); f.read((char*)&vmin,1);
    int64_t hdr_len;
    if (vmaj==1) { uint16_t hl; f.read((char*)&hl,2); hdr_len=hl; }
    else if (vmaj==2) { uint32_t hl; f.read((char*)&hl,4); hdr_len=hl; }
    else { fprintf(stderr,"Unsupported npy version %d.%d\n",vmaj,vmin); exit(1); }

    std::string header((size_t)hdr_len,'\0'); f.read(&header[0],hdr_len);

    // Find shape
    std::vector<int64_t> shape;
    auto find_key = [&](const std::string& key) -> std::string {
        size_t pos = header.find("'"+key+"'"); if (pos==std::string::npos) pos=header.find("\""+key+"\"");
        if (pos==std::string::npos) return "";
        size_t colon = header.find(':',pos+key.size()+2); if (colon==std::string::npos) return "";
        size_t start = header.find_first_not_of(" \"'",colon+1); if (start==std::string::npos) return "";
        size_t end = header.find_first_of(",}\n",start);
        return header.substr(start,end-start);
    };
    std::string descr = find_key("descr");
    if ((descr[0]=='\''||descr[0]=='\"')&&descr.size()>=2) descr=descr.substr(1,descr.size()-2);

    // Parse shape from header
    size_t sp = header.find("'shape'"); if (sp==std::string::npos) sp=header.find("\"shape\"");
    if (sp!=std::string::npos) {
        size_t paren = header.find('(',sp); if (paren==std::string::npos) paren=header.find('[',sp);
        if (paren!=std::string::npos) {
            size_t eparen = header.find(')',paren); if (eparen==std::string::npos) eparen=header.find(']',paren);
            std::string shstr = header.substr(paren+1, eparen-paren-1);
            std::stringstream ss(shstr); std::string tok;
            while (std::getline(ss, tok, ',')) {
                char* end=nullptr; int64_t v=strtoll(tok.c_str(),&end,10);
                if (end!=tok.c_str()) shape.push_back(v);
            }
        }
    }
    int64_t nelem=1; for (auto s:shape) nelem*=s;
    if (shape.empty()) { d0=0; d1=0; return; }

    int64_t offset = 6+1+1+(vmaj==1?2:4)+hdr_len;
    f.seekg(offset);

    bool is_double = (descr=="<f8"||descr=="|f8");
    if (is_double) {
        std::vector<double> tmp(nelem); f.read((char*)tmp.data(),nelem*sizeof(double));
        data.resize(nelem); for (int64_t i=0;i<nelem;i++) data[i]=(float)tmp[i];
    } else {
        data.resize(nelem); f.read((char*)data.data(),nelem*sizeof(float));
    }

    if (shape.size()==1) { d0=(int)shape[0]; d1=1; }
    else if (shape.size()==2) { d0=(int)shape[0]; d1=(int)shape[1]; }
    else { d0=(int)nelem; d1=1; }
}

static void read_npy_int32(const char* path, std::vector<int>& data, int& size) {
    std::ifstream f(path,std::ios::binary);
    if (!f.is_open()) { fprintf(stderr,"Error: cannot open %s\n",path); exit(1); }

    char magic[6]; f.read(magic,6);
    if (magic[0]!=(char)0x93||magic[1]!='N'||magic[2]!='U'||magic[3]!='M'||magic[4]!='P'||magic[5]!='Y')
        { fprintf(stderr,"Error: %s not a .npy file\n",path); exit(1); }

    uint8_t vmaj,vmin; f.read((char*)&vmaj,1); f.read((char*)&vmin,1);
    int64_t hdr_len;
    if (vmaj==1) { uint16_t hl; f.read((char*)&hl,2); hdr_len=hl; }
    else if (vmaj==2) { uint32_t hl; f.read((char*)&hl,4); hdr_len=hl; }
    else { fprintf(stderr,"Unsupported npy version\n"); exit(1); }

    std::string header((size_t)hdr_len,'\0'); f.read(&header[0],hdr_len);

    std::vector<int64_t> shape;
    size_t sp=header.find("'shape'"); if (sp==std::string::npos) sp=header.find("\"shape\"");
    if (sp!=std::string::npos) {
        size_t paren=header.find('(',sp); if (paren==std::string::npos) paren=header.find('[',sp);
        if (paren!=std::string::npos) {
            size_t eparen=header.find(')',paren); if (eparen==std::string::npos) eparen=header.find(']',paren);
            std::string shstr=header.substr(paren+1,eparen-paren-1);
            std::stringstream ss(shstr); std::string tok;
            while (std::getline(ss,tok,',')) {
                char* end=nullptr; int64_t v=strtoll(tok.c_str(),&end,10);
                if (end!=tok.c_str()) shape.push_back(v);
            }
        }
    }
    int64_t nelem=1; for (auto s:shape) nelem*=s;
    if (shape.empty()) { size=0; return; }

    int64_t offset=6+1+1+(vmaj==1?2:4)+hdr_len;
    f.seekg(offset);

    std::string descr;
    size_t dp=header.find("'descr'"); if (dp==std::string::npos) dp=header.find("\"descr\"");
    if (dp!=std::string::npos) {
        size_t colon=header.find(':',dp+7); if (colon!=std::string::npos) {
            size_t ds=header.find_first_not_of(" \"'",colon+1);
            if (ds!=std::string::npos) { size_t de=header.find_first_of(" ,}\n",ds); descr=header.substr(ds,de-ds); }
            if (!descr.empty()&&(descr[0]=='\''||descr[0]=='\"')) descr=descr.substr(1,descr.size()-2);
        }
    }

    bool is_int64 = (descr=="<i8"||descr=="|i8");
    if (is_int64) {
        std::vector<int64_t> tmp(nelem); f.read((char*)tmp.data(),nelem*sizeof(int64_t));
        data.resize(nelem); for (int64_t i=0;i<nelem;i++) data[i]=(int)tmp[i];
    } else {
        data.resize(nelem); f.read((char*)data.data(),nelem*sizeof(int));
    }
    size=(int)nelem;
}

/* =================================================================== */
/*  Device functions (AUC, Consistency)                                 */
/* =================================================================== */

__device__ float auc_device(const float* scores, const int* labels, int n) {
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 1) n_pos++;
        else n_neg++;
    }
    if (n_pos == 0 || n_neg == 0) return 0.5f;

    float correct = 0.0f;
    float ties = 0.0f;
    float total = (float)(n_pos * n_neg);

    for (int i = 0; i < n; i++) {
        if (labels[i] != 1) continue;
        float si = scores[i];
        for (int j = 0; j < n; j++) {
            if (labels[j] != 0) continue;
            float sj = scores[j];
            if (si > sj) correct += 1.0f;
            else if (si == sj) ties += 1.0f;
        }
    }
    return (correct + 0.5f * ties) / total;
}

__device__ float consistency_device(const float* scores, const int* labels, int n) {
    int idx[MAX_SAMPLES];
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int i = 1; i < n; i++) {
        float key_score = scores[idx[i]];
        int key_idx = idx[i];
        int j = i - 1;
        while (j >= 0 && scores[idx[j]] > key_score) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key_idx;
    }

    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 1) n_pos++;
        else n_neg++;
    }

    int tp = n_pos, tn = 0;
    float best = 0.0f;
    for (int i = 0; i < n; i++) {
        float tpr = (n_pos > 0) ? (float)tp / n_pos : 1.0f;
        float tnr = (n_neg > 0) ? (float)tn / n_neg : 1.0f;
        float bal_acc = (tpr + tnr) * 0.5f;
        if (bal_acc > best) best = bal_acc;
        if (labels[idx[i]] == 1) tp--;
        else tn++;
    }
    return best;
}

/* =================================================================== */
/*  CUDA Kernels                                                        */
/* =================================================================== */

__global__ void evaluate_full_kernel(
    const float* __restrict__ A,
    const float* __restrict__ profiles,
    const int* __restrict__ labels,
    const float* __restrict__ weights,
    float* __restrict__ out_auc,
    float* __restrict__ out_consistency,
    int n_samples, int n_items, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    float w0 = weights[idx * 3 + 0];
    float w1 = weights[idx * 3 + 1];
    float w2 = weights[idx * 3 + 2];

    float scores[MAX_SAMPLES];
    for (int s = 0; s < n_samples; s++) scores[s] = 0.0f;

    for (int i = 0; i < n_items; i++) {
        float p = profiles[i * 3 + 0] * w0 +
                  profiles[i * 3 + 1] * w1 +
                  profiles[i * 3 + 2] * w2;
        for (int s = 0; s < n_samples; s++) {
            scores[s] += A[s * n_items + i] * p;
        }
    }

    out_auc[idx] = auc_device(scores, labels, n_samples);
    out_consistency[idx] = consistency_device(scores, labels, n_samples);
}

__global__ void evaluate_precompute_kernel(
    const float* __restrict__ B,
    const int* __restrict__ labels,
    const float* __restrict__ weights,
    float* __restrict__ out_auc,
    float* __restrict__ out_consistency,
    int n_samples, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    float w0 = weights[idx * 3 + 0];
    float w1 = weights[idx * 3 + 1];
    float w2 = weights[idx * 3 + 2];

    float scores[MAX_SAMPLES];
    for (int s = 0; s < n_samples; s++) {
        scores[s] = B[s * 3 + 0] * w0 +
                    B[s * 3 + 1] * w1 +
                    B[s * 3 + 2] * w2;
    }

    out_auc[idx] = auc_device(scores, labels, n_samples);
    out_consistency[idx] = consistency_device(scores, labels, n_samples);
}

/* =================================================================== */
/*  GPU Reduction Kernel                                                */
/* =================================================================== */

struct BestVal {
    float auc;
    float consistency;
    int index;
};

__device__ bool is_better(const BestVal& a, const BestVal& b) {
    if (a.auc > b.auc) return true;
    if (a.auc < b.auc) return false;
    if (a.consistency > b.consistency) return true;
    if (a.consistency < b.consistency) return false;
    return a.index < b.index;
}

__global__ void reduce_best_kernel(
    const float* __restrict__ aucs,
    const float* __restrict__ consistencies,
    float* out_best_auc,
    float* out_best_consistency,
    int* out_best_index,
    int K)
{
    extern __shared__ char shared_mem[];
    BestVal* sdata = (BestVal*)shared_mem;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < K) {
        sdata[tid].auc = aucs[idx];
        sdata[tid].consistency = consistencies[idx];
        sdata[tid].index = idx;
    } else {
        sdata[tid].auc = -1.0f;
        sdata[tid].consistency = -1.0f;
        sdata[tid].index = K;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_better(sdata[tid + s], sdata[tid])) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        BestVal cur;
        cur.auc = *out_best_auc;
        cur.consistency = *out_best_consistency;
        cur.index = *out_best_index;

        if (is_better(sdata[0], cur)) {
            *out_best_auc = sdata[0].auc;
            *out_best_consistency = sdata[0].consistency;
            *out_best_index = sdata[0].index;
        }
    }
}

/* =================================================================== */
/*  Host weight generation                                              */
/* =================================================================== */

static void generate_random_weights(std::vector<float>& weights, int K, unsigned int seed) {
    weights.resize(K * 3);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < K; i++) {
        float x1 = -std::log(1.0f - dist(rng) + 1e-37f);
        float x2 = -std::log(1.0f - dist(rng) + 1e-37f);
        float x3 = -std::log(1.0f - dist(rng) + 1e-37f);
        float sum = x1 + x2 + x3;
        weights[i * 3 + 0] = x1 / sum;
        weights[i * 3 + 1] = x2 / sum;
        weights[i * 3 + 2] = x3 / sum;
    }
}

static int generate_grid_weights(std::vector<float>& weights, int resolution) {
    weights.clear();
    for (int i = 0; i <= resolution; i++) {
        for (int j = 0; j <= resolution - i; j++) {
            int k = resolution - i - j;
            weights.push_back((float)i / resolution);
            weights.push_back((float)j / resolution);
            weights.push_back((float)k / resolution);
        }
    }
    return (int)(weights.size() / 3);
}

static int resolution_for_K(int K) {
    int r = (int)std::sqrt(2.0 * K);
    return std::max(r, 1);
}

static void generate_dirichlet_weights(std::vector<float>& weights,
                                        const float* alpha, int n,
                                        std::mt19937& rng) {
    weights.resize(n * 3);
    std::gamma_distribution<float> gamma1(alpha[0], 1.0f);
    std::gamma_distribution<float> gamma2(alpha[1], 1.0f);
    std::gamma_distribution<float> gamma3(alpha[2], 1.0f);
    for (int i = 0; i < n; i++) {
        float x1 = gamma1(rng);
        float x2 = gamma2(rng);
        float x3 = gamma3(rng);
        float sum = x1 + x2 + x3;
        weights[i * 3 + 0] = x1 / sum;
        weights[i * 3 + 1] = x2 / sum;
        weights[i * 3 + 2] = x3 / sum;
    }
}

/* =================================================================== */
/*  GPU Context + Evaluation helpers                                    */
/* =================================================================== */

struct GPUData {
    float *d_A, *d_profiles, *d_B;
    int *d_labels;
    int n_samples, n_items;
    bool mode_full;
};

static void evaluate_batch(const GPUData& gpu,
                           const float* h_weights, int K_batch,
                           float* h_aucs, float* h_cons,
                           int block_size)
{
    float *d_w, *d_auc, *d_con;
    size_t wb = K_batch * 3 * sizeof(float);
    size_t rb = K_batch * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_w, wb));
    CUDA_CHECK(cudaMalloc(&d_auc, rb));
    CUDA_CHECK(cudaMalloc(&d_con, rb));
    CUDA_CHECK(cudaMemcpy(d_w, h_weights, wb, cudaMemcpyHostToDevice));

    int grid = (K_batch + block_size - 1) / block_size;
    if (gpu.mode_full) {
        evaluate_full_kernel<<<grid, block_size>>>(
            gpu.d_A, gpu.d_profiles, gpu.d_labels,
            d_w, d_auc, d_con,
            gpu.n_samples, gpu.n_items, K_batch);
    } else {
        evaluate_precompute_kernel<<<grid, block_size>>>(
            gpu.d_B, gpu.d_labels,
            d_w, d_auc, d_con,
            gpu.n_samples, K_batch);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_aucs, d_auc, rb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cons, d_con, rb, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_auc));
    CUDA_CHECK(cudaFree(d_con));
}

struct SearchOutput {
    float best_auc;
    float best_consistency;
    float best_w0, best_w1, best_w2;
    int K_actual;
    double time_sec;
};

/* =================================================================== */
/*  Search strategies                                                   */
/* =================================================================== */

static SearchOutput run_random_search(
    const GPUData& gpu, int K, unsigned int seed,
    int block_size, int batch_size)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> all_w;
    generate_random_weights(all_w, K, seed);

    float best_auc = -1.0f, best_cons = -1.0f;
    int best_idx = -1;
    std::vector<float> ba(batch_size), bc(batch_size);

    for (int off = 0; off < K; off += batch_size) {
        int bk = std::min(batch_size, K - off);
        evaluate_batch(gpu, &all_w[off * 3], bk, ba.data(), bc.data(), block_size);
        for (int i = 0; i < bk; i++) {
            int gi = off + i;
            bool better = false;
            if (ba[i] > best_auc) better = true;
            else if (ba[i] == best_auc && bc[i] > best_cons) better = true;
            else if (ba[i] == best_auc && bc[i] == best_cons && gi < best_idx) better = true;
            if (better) { best_auc = ba[i]; best_cons = bc[i]; best_idx = gi; }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    SearchOutput out;
    out.best_auc = best_auc;
    out.best_consistency = best_cons;
    out.best_w0 = all_w[best_idx * 3 + 0];
    out.best_w1 = all_w[best_idx * 3 + 1];
    out.best_w2 = all_w[best_idx * 3 + 2];
    out.K_actual = K;
    out.time_sec = std::chrono::duration<double>(t1 - t0).count();
    return out;
}

static SearchOutput run_grid_search(
    const GPUData& gpu, int K_hint,
    int block_size, int batch_size,
    int grid_resolution)
{
    int resolution = (grid_resolution > 0) ? grid_resolution : resolution_for_K(K_hint);
    std::vector<float> all_w;
    int actual_K = generate_grid_weights(all_w, resolution);

    auto t0 = std::chrono::high_resolution_clock::now();
    float best_auc = -1.0f, best_cons = -1.0f;
    int best_idx = -1;
    std::vector<float> ba(batch_size), bc(batch_size);

    for (int off = 0; off < actual_K; off += batch_size) {
        int bk = std::min(batch_size, actual_K - off);
        evaluate_batch(gpu, &all_w[off * 3], bk, ba.data(), bc.data(), block_size);
        for (int i = 0; i < bk; i++) {
            int gi = off + i;
            bool better = false;
            if (ba[i] > best_auc) better = true;
            else if (ba[i] == best_auc && bc[i] > best_cons) better = true;
            else if (ba[i] == best_auc && bc[i] == best_cons && gi < best_idx) better = true;
            if (better) { best_auc = ba[i]; best_cons = bc[i]; best_idx = gi; }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    SearchOutput out;
    out.best_auc = best_auc;
    out.best_consistency = best_cons;
    out.best_w0 = all_w[best_idx * 3 + 0];
    out.best_w1 = all_w[best_idx * 3 + 1];
    out.best_w2 = all_w[best_idx * 3 + 2];
    out.K_actual = actual_K;
    out.time_sec = std::chrono::duration<double>(t1 - t0).count();
    return out;
}

static SearchOutput run_hybrid_search(
    const GPUData& gpu, int K, unsigned int seed,
    int block_size, int batch_size)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // Budget: 20% grid, 60% random, 20% local
    int K_grid_max = (int)(K * 0.2f);
    int resolution = resolution_for_K(K_grid_max);
    std::vector<float> grid_w;
    int K_grid = generate_grid_weights(grid_w, resolution);

    int K_random = (int)(K * 0.6f);
    int K_local = K - K_grid - K_random;
    if (K_local < 0) { K_random = K - K_grid; K_local = 0; }

    // Container for all weights
    std::vector<float> all_w;

    // Phase 1: Grid
    all_w.insert(all_w.end(), grid_w.begin(), grid_w.end());

    // Phase 2: Random
    {
        std::vector<float> rw;
        generate_random_weights(rw, K_random, seed + 1);
        all_w.insert(all_w.end(), rw.begin(), rw.end());
    }

    int total_K = K_grid + K_random;
    std::vector<float> ba(batch_size), bc(batch_size);
    float best_auc = -1.0f, best_cons = -1.0f;
    int best_idx = -1;

    // Evaluate phases 1+2
    for (int off = 0; off < total_K; off += batch_size) {
        int bk = std::min(batch_size, total_K - off);
        evaluate_batch(gpu, &all_w[off * 3], bk, ba.data(), bc.data(), block_size);
        for (int i = 0; i < bk; i++) {
            int gi = off + i;
            bool better = false;
            if (ba[i] > best_auc) better = true;
            else if (ba[i] == best_auc && bc[i] > best_cons) better = true;
            else if (ba[i] == best_auc && bc[i] == best_cons && gi < best_idx) better = true;
            if (better) { best_auc = ba[i]; best_cons = bc[i]; best_idx = gi; }
        }
    }

    // Phase 3: Local around best from phases 1+2
    if (K_local > 0 && best_idx >= 0) {
        float bw0 = all_w[best_idx * 3 + 0];
        float bw1 = all_w[best_idx * 3 + 1];
        float bw2 = all_w[best_idx * 3 + 2];
        float alpha[3] = {
            std::max(bw0 * 100.0f, 1e-3f),
            std::max(bw1 * 100.0f, 1e-3f),
            std::max(bw2 * 100.0f, 1e-3f)
        };
        std::mt19937 rng(seed + 2);
        std::vector<float> local_w;
        generate_dirichlet_weights(local_w, alpha, K_local, rng);
        int local_offset = (int)(all_w.size() / 3);
        all_w.insert(all_w.end(), local_w.begin(), local_w.end());

        for (int off = 0; off < K_local; off += batch_size) {
            int bk = std::min(batch_size, K_local - off);
            evaluate_batch(gpu, &local_w[off * 3], bk, ba.data(), bc.data(), block_size);
            for (int i = 0; i < bk; i++) {
                int gi = local_offset + off + i;
                bool better = false;
                if (ba[i] > best_auc) better = true;
                else if (ba[i] == best_auc && bc[i] > best_cons) better = true;
                else if (ba[i] == best_auc && bc[i] == best_cons && gi < best_idx) better = true;
                if (better) { best_auc = ba[i]; best_cons = bc[i]; best_idx = gi; }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    SearchOutput out;
    out.best_auc = best_auc;
    out.best_consistency = best_cons;
    out.best_w0 = all_w[best_idx * 3 + 0];
    out.best_w1 = all_w[best_idx * 3 + 1];
    out.best_w2 = all_w[best_idx * 3 + 2];
    out.K_actual = K;
    out.time_sec = std::chrono::duration<double>(t1 - t0).count();
    return out;
}

/* =================================================================== */
/*  Validation                                                          */
/* =================================================================== */

static void validate_data(int nA0, int nA1, int nP0, int nP1, int nL) {
    if (nP1 != 3) {
        fprintf(stderr, "Error: profiles debe tener 3 columnas, tiene %d\n", nP1);
        exit(1);
    }
    if (nA1 != nP0) {
        fprintf(stderr, "Error: columnas de A (%d) != filas de profiles (%d)\n", nA1, nP0);
        exit(1);
    }
    if (nA0 != nL) {
        fprintf(stderr, "Error: filas de A (%d) != largo de labels (%d)\n", nA0, nL);
        exit(1);
    }
}

static void validate_labels(const std::vector<int>& labels) {
    bool h0 = false, h1 = false;
    for (int v : labels) { if (v == 0) h0 = true; if (v == 1) h1 = true; }
    if (!h0 || !h1) {
        fprintf(stderr, "Error: labels debe contener al menos un 0 y un 1\n");
        exit(1);
    }
}

/* =================================================================== */
/*  Main                                                                */
/* =================================================================== */

int main(int argc, char** argv) {
    int K = 10000;
    int seed = 42;
    std::string search_mode = "random";
    std::string data_dir = "data/npy";
    int block_size = 256;
    int batch_size = 1000000;
    std::string mode = "full";
    int grid_resolution = 0;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--search") == 0 && i + 1 < argc) search_mode = argv[++i];
        else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(argv[i], "--grid-resolution") == 0 && i + 1 < argc) grid_resolution = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
    }

    bool mode_full = (mode == "full");

    // ── Load data ──
    std::string A_path = data_dir + "/matrix_A.npy";
    std::string labels_path = data_dir + "/labels.npy";
    std::string profiles_path = data_dir + "/profiles_TSF.npy";

    std::vector<float> h_A, h_P;
    std::vector<int> h_L;
    int nA0 = 0, nA1 = 0, nP0 = 0, nP1 = 0, nL = 0;

    read_npy_float32(A_path.c_str(), h_A, nA0, nA1);
    read_npy_float32(profiles_path.c_str(), h_P, nP0, nP1);
    read_npy_int32(labels_path.c_str(), h_L, nL);

    validate_data(nA0, nA1, nP0, nP1, nL);
    validate_labels(h_L);

    if (nA0 > MAX_SAMPLES || nP0 > MAX_ITEMS) {
        fprintf(stderr, "Error: data too large (samples=%d > %d or items=%d > %d)\n",
                nA0, MAX_SAMPLES, nP0, MAX_ITEMS);
        return 1;
    }

    if (verbose) fprintf(stderr, "Data: A(%d,%d) profiles(%d,%d) labels(%d)\n",
                         nA0, nA1, nP0, nP1, nL);

    // ── GPU allocation ──
    GPUData gpu;
    gpu.n_samples = nA0;
    gpu.n_items = nA1;
    gpu.mode_full = mode_full;

    CUDA_CHECK(cudaMalloc(&gpu.d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_profiles, h_P.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_labels, h_L.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu.d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_profiles, h_P.data(), h_P.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_labels, h_L.data(), h_L.size() * sizeof(int), cudaMemcpyHostToDevice));

    if (!mode_full) {
        std::vector<float> h_B(nA0 * 3, 0.0f);
        for (int s = 0; s < nA0; s++)
            for (int i = 0; i < nA1; i++) {
                float a = h_A[s * nA1 + i];
                h_B[s * 3 + 0] += a * h_P[i * 3 + 0];
                h_B[s * 3 + 1] += a * h_P[i * 3 + 1];
                h_B[s * 3 + 2] += a * h_P[i * 3 + 2];
            }
        CUDA_CHECK(cudaMalloc(&gpu.d_B, nA0 * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(gpu.d_B, h_B.data(), nA0 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        gpu.d_B = nullptr;
    }

    // ── Run search ──
    SearchOutput result;
    if (search_mode == "grid") {
        result = run_grid_search(gpu, K, block_size, batch_size, grid_resolution);
    } else if (search_mode == "hybrid") {
        result = run_hybrid_search(gpu, K, seed, block_size, batch_size);
    } else {
        result = run_random_search(gpu, K, seed, block_size, batch_size);
    }

    // ── Output CSV ──
    printf("cuda_c,%s,%s,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d\n",
           search_mode.c_str(), mode.c_str(), K, result.K_actual, gpu.n_items,
           result.best_auc, result.best_consistency,
           result.best_w0, result.best_w1, result.best_w2,
           result.time_sec, seed, block_size);

    if (verbose) {
        float wsum = result.best_w0 + result.best_w1 + result.best_w2;
        fprintf(stderr, "W sum = %.9f (should be ~1)\n", wsum);
    }

    // ── Cleanup ──
    CUDA_CHECK(cudaFree(gpu.d_A));
    CUDA_CHECK(cudaFree(gpu.d_profiles));
    CUDA_CHECK(cudaFree(gpu.d_labels));
    if (!mode_full) CUDA_CHECK(cudaFree(gpu.d_B));

    return 0;
}
