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

#define MAX_SAMPLES 4096
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

    // Detect fortran_order
    bool fortran = false;
    {
        std::string fk = find_key("fortran_order");
        if (fk == "True" || fk == "True," || fk == "1") fortran = true;
    }

    int64_t offset = 6+1+1+(vmaj==1?2:4)+hdr_len;
    f.seekg(offset);

    bool is_double = (descr=="<f8"||descr=="|f8");
    if (is_double) {
        std::vector<double> tmp(nelem); f.read((char*)tmp.data(),nelem*sizeof(double));
        data.resize(nelem); for (int64_t i=0;i<nelem;i++) data[i]=(float)tmp[i];
    } else {
        data.resize(nelem); f.read((char*)data.data(),nelem*sizeof(float));
    }

    // Transpose from column-major to row-major if needed
    if (fortran && shape.size() == 2) {
        int64_t rows = shape[0], cols = shape[1];
        std::vector<float> tmp(data.size());
        for (int64_t c = 0; c < cols; c++)
            for (int64_t r = 0; r < rows; r++)
                tmp[r * cols + c] = data[c * rows + r];
        data.swap(tmp);
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

    // Detect fortran_order (same logic as float32 reader)
    bool fortran32 = false;
    {
        auto fk_fn = [&](const std::string& key) -> std::string {
            size_t p = header.find("'"+key+"'"); if (p==std::string::npos) p=header.find("\""+key+"\"");
            if (p==std::string::npos) return "";
            size_t cl = header.find(':',p+key.size()+2); if (cl==std::string::npos) return "";
            size_t st = header.find_first_not_of(" \"'",cl+1); if (st==std::string::npos) return "";
            size_t en = header.find_first_of(",}\n",st);
            return header.substr(st,en-st);
        };
        std::string fk = fk_fn("fortran_order");
        if (fk == "True" || fk == "True," || fk == "1") fortran32 = true;
    }

    bool is_int64 = (descr=="<i8"||descr=="|i8");
    if (is_int64) {
        std::vector<int64_t> tmp(nelem); f.read((char*)tmp.data(),nelem*sizeof(int64_t));
        data.resize(nelem); for (int64_t i=0;i<nelem;i++) data[i]=(int)tmp[i];
    } else {
        data.resize(nelem); f.read((char*)data.data(),nelem*sizeof(int));
    }

    // Transpose 2D int32 data if fortran_order
    if (fortran32 && shape.size() == 2) {
        int64_t rows = shape[0], cols = shape[1];
        std::vector<int> tmp(data.size());
        for (int64_t c = 0; c < cols; c++)
            for (int64_t r = 0; r < rows; r++)
                tmp[r * cols + c] = data[c * rows + r];
        data.swap(tmp);
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
/*  ANSI Logger (mismo estilo que python/logger.py)                    */
/* =================================================================== */

#define ANSI_BOLD    "\x1b[1m"
#define ANSI_GREEN   "\x1b[32m"
#define ANSI_YELLOW  "\x1b[33m"
#define ANSI_CYAN    "\x1b[36m"
#define ANSI_MAGENTA "\x1b[35m"
#define ANSI_GOLD    "\x1b[1;33m"
#define ANSI_RESET   "\x1b[0m"
#define ANSI_DIM     "\x1b[2m"

static int term_unicode(void) {
#ifdef _WIN32
    return 0;
#else
    return 1;
#endif
}

static void log_header_cuda(const char *impl, int n_items, long k) {
    int u = term_unicode();
    const char *tl = u ? "\xe2\x95\xad" : "+"; // ╭
    (void)u;
    const char *tr = u ? "\xe2\x95\xae" : "+";
    const char *bl = u ? "\xe2\x95\xb0" : "+";
    const char *br = u ? "\xe2\x95\xaf" : "+";
    const char *h  = u ? "\xe2\x95\x90" : "-";
    const char *v  = u ? "\xe2\x95\x91" : "|";

    int impl_len = (int)strlen(impl);
    int pad = 48 - impl_len;
    if (pad < 2) pad = 2;

    fprintf(stderr, "\n");
    fprintf(stderr, "%s%s%s %s ", ANSI_BOLD ANSI_CYAN, tl, h, impl);
    for (int i = 0; i < pad; i++) fprintf(stderr, "%s", h);
    fprintf(stderr, "%s%s\n", tr, ANSI_RESET);

    fprintf(stderr, "%s%s%s  %s%s%s%s\n", ANSI_CYAN, v, ANSI_RESET,
            ANSI_BOLD "BUSQUEDA DE PESOS OPTIMOS" ANSI_RESET, ANSI_CYAN, v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  items (N) ... %d%s%s\n", ANSI_CYAN, v, ANSI_RESET,
            n_items, ANSI_CYAN, v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  candidatos   %ld%s%s\n", ANSI_CYAN, v, ANSI_RESET,
            k, ANSI_CYAN, v, ANSI_RESET);
    fprintf(stderr, "%s%s", ANSI_CYAN, bl);
    for (int i = 0; i < 56; i++) fprintf(stderr, "%s", h);
    fprintf(stderr, "%s%s\n\n", br, ANSI_RESET);
}

static void log_improvement_cuda(long iteration, long k,
                                  double auc, double prev_auc,
                                  double cons, double w1, double w2, double w3) {
    const char *arrow = term_unicode() ? "\xe2\x9e\x9c" : "->";

    if (prev_auc < 0) {
        fprintf(stderr, "  %s%s%s  %sAUC %.6f%s  %s(initial)%s  iter %ld/%ld  consist=%.4f  w=[%.4f %.4f %.4f]\n",
                ANSI_GOLD, arrow, ANSI_RESET,
                ANSI_BOLD, auc, ANSI_RESET,
                ANSI_GREEN, ANSI_RESET,
                iteration, k, cons, w1, w2, w3);
    } else {
        double delta = auc - prev_auc;
        fprintf(stderr, "  %s%s%s  %sAUC %.6f%s  %s(+%.6f)%s  iter %ld/%ld  consist=%.4f  w=[%.4f %.4f %.4f]\n",
                ANSI_GOLD, arrow, ANSI_RESET,
                ANSI_BOLD, auc, ANSI_RESET,
                ANSI_GREEN, delta, ANSI_RESET,
                iteration, k, cons, w1, w2, w3);
    }
}

static void log_summary_cuda(const char *impl, double auc, double cons,
                              double w1, double w2, double w3,
                              double time_sec, int improvements) {
    int u = term_unicode();
    const char *tl = u ? "\xe2\x95\xad" : "+";
    const char *tr = u ? "\xe2\x95\xae" : "+";
    const char *bl = u ? "\xe2\x95\xb0" : "+";
    const char *br = u ? "\xe2\x95\xaf" : "+";
    const char *h  = u ? "\xe2\x95\x90" : "-";
    const char *v  = u ? "\xe2\x95\x91" : "|";

    char count_str[32];
    snprintf(count_str, sizeof(count_str), "%d", improvements);
    int count_len = (int)strlen(count_str);
    int pad = 40 - count_len;
    if (pad < 2) pad = 2;

    fprintf(stderr, "\n");
    fprintf(stderr, "%s%s%s MEJOR RESULTADO ", ANSI_BOLD ANSI_MAGENTA, tl, h);
    for (int i = 0; i < pad; i++) fprintf(stderr, "%s", h);
    fprintf(stderr, "%s%s\n", tr, ANSI_RESET);

    fprintf(stderr, "%s%s%s  %simplementacion%s %s %s%s\n",
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET,
            ANSI_BOLD, ANSI_RESET, impl,
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  %smejoras%s        %s %s%s\n",
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET,
            ANSI_BOLD, ANSI_RESET, count_str,
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  %sAUC%s            %s %.9f%s%s\n",
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET,
            ANSI_BOLD, ANSI_RESET,
            ANSI_GOLD, auc, ANSI_RESET,
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  %sconsistencia%s   %s %.4f%s%s\n",
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET,
            ANSI_BOLD, ANSI_RESET,
            ANSI_BOLD ANSI_MAGENTA, cons,
            v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  %spesos W%s        %s [%.9f, %.9f, %.9f]%s%s\n",
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET,
            ANSI_BOLD, ANSI_RESET,
            ANSI_BOLD ANSI_MAGENTA, w1, w2, w3,
            v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  %ssuma W%s         %s %.9f%s%s\n",
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET,
            ANSI_BOLD, ANSI_RESET,
            ANSI_BOLD ANSI_MAGENTA, w1 + w2 + w3,
            v, ANSI_RESET);
    fprintf(stderr, "%s%s%s  %stiempo%s         %s %s%.6f s%s%s%s\n",
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET,
            ANSI_BOLD, ANSI_RESET,
            ANSI_BOLD ANSI_MAGENTA,
            ANSI_CYAN, time_sec, ANSI_RESET,
            ANSI_BOLD ANSI_MAGENTA, v, ANSI_RESET);
    fprintf(stderr, "%s%s", ANSI_BOLD ANSI_MAGENTA, bl);
    for (int i = 0; i < 56; i++) fprintf(stderr, "%s", h);
    fprintf(stderr, "%s%s\n\n", br, ANSI_RESET);
}

/* =================================================================== */
/*  PCG64 portable (MSVC + GCC) — mismo RNG que OpenMP/MPI/Python      */
/* =================================================================== */

#ifdef _MSC_VER
#include <intrin.h>
static inline uint64_t mulh64(uint64_t a, uint64_t b, uint64_t *hi) {
    return _umul128(a, b, hi);
}
#else
static inline uint64_t mulh64(uint64_t a, uint64_t b, uint64_t *hi) {
    __uint128_t r = (__uint128_t)a * b;
    *hi = (uint64_t)(r >> 64);
    return (uint64_t)r;
}
#endif

static inline uint64_t rotr64(uint64_t x, unsigned r) {
    return (x >> r) | (x << ((-r) & 63));
}

#define PCG_MULT_HI 2549297995355413924ULL
#define PCG_MULT_LO 4865540595714422341ULL

static void pcg64_seed(uint64_t *s, uint64_t seed) {
    // SeedSequence simplificado (equivalente a numpy)
    uint32_t pool[4];
    uint32_t hc = 0x43b0d7e5U;
    uint32_t ent[2] = { (uint32_t)(seed & 0xFFFFFFFFULL), (uint32_t)(seed >> 32) };
    int nent = (seed <= 0xFFFFFFFFULL) ? 1 : 2;
    for (int i = 0; i < 4; i++) pool[i] = 0;
    for (int i = 0; i < 4; i++) {
        uint32_t v = (i < nent) ? ent[i] : 0;
        v ^= hc; hc = (hc * 0x931e8875U) & 0xFFFFFFFFU;
        v = (v * hc) & 0xFFFFFFFFU; v ^= (v >> 16);
        pool[i] = v;
    }
    for (int src = 0; src < 4; src++)
        for (int dst = 0; dst < 4; dst++)
            if (src != dst) {
                uint32_t x = pool[dst], y = pool[src];
                uint32_t mix = (0xca01f9ddU * x - 0x4973f715U * y) & 0xFFFFFFFFU;
                mix ^= (mix >> 16);
                uint32_t hv = y ^ hc; hc = (hc * 0x58f38dedU) & 0xFFFFFFFFU;
                hv = (hv * hc) & 0xFFFFFFFFU; hv ^= (hv >> 16);
                pool[dst] = mix ^ hv;
            }

    uint64_t ss[4];
    uint32_t hc2 = 0x8b51f9ddU;
    for (int i = 0; i < 4; i++) {
        uint32_t lo, hi;
        uint32_t dv = pool[i % 4] ^ hc2;
        hc2 = (hc2 * 0x58f38dedU) & 0xFFFFFFFFU;
        dv = (dv * hc2) & 0xFFFFFFFFU; dv ^= (dv >> 16);
        lo = dv;
        i++;
        dv = pool[i % 4] ^ hc2;
        hc2 = (hc2 * 0x58f38dedU) & 0xFFFFFFFFU;
        dv = (dv * hc2) & 0xFFFFFFFFU; dv ^= (dv >> 16);
        hi = dv;
        ss[i/2] = (uint64_t)lo | ((uint64_t)hi << 32);
    }

    uint64_t initstate_hi = ss[0], initstate_lo = ss[1];
    uint64_t initseq_hi   = ss[2], initseq_lo   = ss[3];

    // inc = (initseq << 1) | 1  (128-bit shift)
    uint64_t inc_hi = (initseq_hi << 1) | (initseq_lo >> 63);
    uint64_t inc_lo = (initseq_lo << 1) | 1;

    // state = 0, then state = state * mult + inc twice
    uint64_t st_hi = 0, st_lo = 0;
    for (int r = 0; r < 2; r++) {
        // mul128(st, PCG_MULT)
        uint64_t lo, hi1, lo2, hi2, lo3, hi3;
        lo  = mulh64(st_lo, PCG_MULT_LO, &hi1);
        lo2 = mulh64(st_hi, PCG_MULT_LO, &hi2);
        lo3 = mulh64(st_lo, PCG_MULT_HI, &hi3);
        uint64_t m_lo = lo;
        uint64_t m_hi = hi1 + lo2 + lo3 + st_hi * PCG_MULT_HI;

        // m += inc
        uint64_t sum_lo = m_lo + inc_lo;
        uint64_t sum_hi = m_hi + inc_hi + (sum_lo < m_lo ? 1 : 0);
        st_hi = sum_hi; st_lo = sum_lo;

        if (r == 0) {
            // state += initstate
            uint64_t a_lo = st_lo + initstate_lo;
            uint64_t a_hi = st_hi + initstate_hi + (a_lo < st_lo ? 1 : 0);
            st_hi = a_hi; st_lo = a_lo;
        }
    }
    s[0] = st_hi; s[1] = st_lo; s[2] = inc_hi; s[3] = inc_lo;
}

static uint64_t pcg64_xs(uint64_t *s) {
    uint64_t hi = s[0], st_lo = s[1];
    uint64_t lo, hi1, lo2, hi2, lo3, hi3;
    lo  = mulh64(st_lo, PCG_MULT_LO, &hi1);
    lo2 = mulh64(hi,  PCG_MULT_LO, &hi2);
    lo3 = mulh64(st_lo, PCG_MULT_HI, &hi3);
    uint64_t m_lo = lo;
    uint64_t m_hi = hi1 + lo2 + lo3 + hi * PCG_MULT_HI;
    uint64_t sum_lo = m_lo + s[3];
    uint64_t sum_hi = m_hi + s[2] + (sum_lo < m_lo ? 1 : 0);
    s[0] = sum_hi; s[1] = sum_lo;

    unsigned rot = (unsigned)(hi >> 58);
    return rotr64(hi ^ st_lo, rot);
}

static double pcg64_u01(uint64_t *s) {
    return (pcg64_xs(s) >> 11) * (1.0 / 9007199254740992.0);
}

static void pcg64_simplex(uint64_t *s, double w[3]) {
    double a = -log(pcg64_u01(s));
    double b = -log(pcg64_u01(s));
    double c = -log(pcg64_u01(s));
    double sum = a + b + c;
    w[0] = a / sum; w[1] = b / sum; w[2] = c / sum;
}

static void pcg64_dirichlet(const double alpha[3], uint64_t *s, double w[3]) {
    // Gamma(alpha, 1) via Marsaglia-Tsang for alpha >= 1, Best for alpha < 1
    auto gamma_sample = [&](double a) -> double {
        if (a <= 0.0) return 0.0;
        if (a == 1.0) return -log(pcg64_u01(s));
        if (a < 1.0) {
            double e = 2.71828182845904523536;
            double thr = e / (e + a);
            for (;;) {
                double u = pcg64_u01(s), v = pcg64_u01(s);
                if (u <= thr) {
                    double z = pow(v * a, 1.0 / a);
                    if (z <= 1.0) return z;
                } else {
                    double z = 1.0 - log(v);
                    if (u <= pow(z, a - 1.0)) return z;
                }
            }
        } else {
            double d = a - 1.0 / 3.0;
            double c = 1.0 / sqrt(9.0 * d);
            for (;;) {
                double u1, u2;
                double x, v, v3;
                do {
                    u1 = pcg64_u01(s);
                    while (u1 == 0.0) u1 = pcg64_u01(s);
                    u2 = pcg64_u01(s);
                    x = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
                    v = 1.0 + c * x;
                } while (v <= 0.0);
                v3 = v * v * v;
                double u = pcg64_u01(s);
                if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v3;
                if (log(u) < 0.5 * x * x + d * (1.0 - v3 + log(v3))) return d * v3;
            }
        }
    };
    double g[3], sum = 0.0;
    for (int i = 0; i < 3; i++) { g[i] = gamma_sample(alpha[i]); sum += g[i]; }
    for (int i = 0; i < 3; i++) w[i] = g[i] / sum;
}

/* =================================================================== */
/*  Host weight generation                                              */
/* =================================================================== */

static void generate_random_weights(std::vector<float>& weights, int K, unsigned int seed) {
    weights.resize(K * 3);
    uint64_t pcg[4];
    pcg64_seed(pcg, (uint64_t)seed);
    for (int i = 0; i < K; i++) {
        double w[3];
        pcg64_simplex(pcg, w);
        weights[i * 3 + 0] = (float)w[0];
        weights[i * 3 + 1] = (float)w[1];
        weights[i * 3 + 2] = (float)w[2];
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
                                        uint64_t* pcg) {
    weights.resize(n * 3);
    double alphad[3] = { (double)alpha[0], (double)alpha[1], (double)alpha[2] };
    for (int i = 0; i < n; i++) {
        double w[3];
        pcg64_dirichlet(alphad, pcg, w);
        weights[i * 3 + 0] = (float)w[0];
        weights[i * 3 + 1] = (float)w[1];
        weights[i * 3 + 2] = (float)w[2];
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
    int block_size, int batch_size, int verbose)
{
    if (verbose) log_header_cuda("cuda_c (random)", gpu.n_items, K);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> all_w;
    generate_random_weights(all_w, K, seed);

    float best_auc = -1.0f, best_cons = -1.0f, prev_auc = -1.0f;
    int best_idx = -1;
    int improvements = 0;
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
            if (better) {
                prev_auc = best_auc;
                best_auc = ba[i]; best_cons = bc[i]; best_idx = gi;
                improvements++;
                if (verbose) {
                    float *w = &all_w[gi * 3];
                    log_improvement_cuda(gi, K, best_auc, prev_auc, best_cons,
                                         w[0], w[1], w[2]);
                }
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

    if (verbose) {
        log_summary_cuda("cuda_c (random)", best_auc, best_cons,
                         out.best_w0, out.best_w1, out.best_w2,
                         out.time_sec, improvements);
    }
    return out;
}

static SearchOutput run_grid_search(
    const GPUData& gpu, int K_hint,
    int block_size, int batch_size,
    int grid_resolution, int verbose)
{
    int resolution = (grid_resolution > 0) ? grid_resolution : resolution_for_K(K_hint);
    std::vector<float> all_w;
    int actual_K = generate_grid_weights(all_w, resolution);

    if (verbose) log_header_cuda("cuda_c (grid)", gpu.n_items, actual_K);

    auto t0 = std::chrono::high_resolution_clock::now();
    float best_auc = -1.0f, best_cons = -1.0f, prev_auc = -1.0f;
    int best_idx = -1;
    int improvements = 0;
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
            if (better) {
                prev_auc = best_auc;
                best_auc = ba[i]; best_cons = bc[i]; best_idx = gi;
                improvements++;
                if (verbose) {
                    float *w = &all_w[gi * 3];
                    log_improvement_cuda(gi, actual_K, best_auc, prev_auc, best_cons,
                                         w[0], w[1], w[2]);
                }
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
    out.K_actual = actual_K;
    out.time_sec = std::chrono::duration<double>(t1 - t0).count();

    if (verbose) {
        log_summary_cuda("cuda_c (grid)", best_auc, best_cons,
                         out.best_w0, out.best_w1, out.best_w2,
                         out.time_sec, improvements);
    }
    return out;
}

static SearchOutput run_hybrid_search(
    const GPUData& gpu, int K, unsigned int seed,
    int block_size, int batch_size, int verbose)
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
            if (better) {
                best_auc = ba[i]; best_cons = bc[i]; best_idx = gi;
                if (verbose) {
                    float *w = &all_w[gi * 3];
                    fprintf(stderr, "  \x1b[1;33m->\x1b[0m  "
                            "\x1b[1mAUC %.6f\x1b[0m  "
                            "iter %d/%d  consist=%.4f  w=[%.4f %.4f %.4f]\n",
                            best_auc, gi, total_K, best_cons, w[0], w[1], w[2]);
                }
            }
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
        uint64_t pcg_local[4];
        pcg64_seed(pcg_local, (uint64_t)(seed + 2));
        std::vector<float> local_w;
        generate_dirichlet_weights(local_w, alpha, K_local, pcg_local);
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
                if (better) {
                    best_auc = ba[i]; best_cons = bc[i]; best_idx = gi;
                    if (verbose) {
                        float *w = &all_w[gi * 3];
                        fprintf(stderr, "  \x1b[1;33m->\x1b[0m  "
                                "\x1b[1mAUC %.6f\x1b[0m  "
                                "iter %d/%d  consist=%.4f  w=[%.4f %.4f %.4f]\n",
                                best_auc, gi, K, best_cons, w[0], w[1], w[2]);
                    }
                }
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
    std::string weights_file = "";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--search") == 0 && i + 1 < argc) search_mode = argv[++i];
        else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(argv[i], "--grid-resolution") == 0 && i + 1 < argc) grid_resolution = atoi(argv[++i]);
        else if (strcmp(argv[i], "--weights-file") == 0 && i + 1 < argc) weights_file = argv[++i];
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

    if (!weights_file.empty()) {
        // Load pre-generated weights from file (same RNG as Python)
        std::vector<float> ext_weights;
        FILE *fw = fopen(weights_file.c_str(), "rb");
        if (!fw) { fprintf(stderr, "Error: cannot open %s\n", weights_file.c_str()); return 1; }
        fseek(fw, 0, SEEK_END); long fsize = ftell(fw); rewind(fw);
        int n_floats = (int)(fsize / sizeof(float));
        if (n_floats % 3 != 0) { fprintf(stderr, "Error: weights file size not multiple of 3\n"); return 1; }
        int ext_K = n_floats / 3;
        ext_weights.resize(n_floats);
        fread(ext_weights.data(), sizeof(float), n_floats, fw);
        fclose(fw);

        if (verbose) {
            log_header_cuda("cuda_c (weights-file)", gpu.n_items, ext_K);
        }

        // Evaluate all weights in batches
        float best_auc = -1.0f, best_cons = -1.0f, prev_auc = -1.0f;
        int best_idx = -1;
        int improvements = 0;
        std::vector<float> ba(batch_size), bc(batch_size);
        for (int off = 0; off < ext_K; off += batch_size) {
            int bk = std::min(batch_size, ext_K - off);
            evaluate_batch(gpu, &ext_weights[off * 3], bk, ba.data(), bc.data(), block_size);
            for (int i = 0; i < bk; i++) {
                int gi = off + i;
                bool better = false;
                if (ba[i] > best_auc) better = true;
                else if (ba[i] == best_auc && bc[i] > best_cons) better = true;
                else if (ba[i] == best_auc && bc[i] == best_cons && gi < best_idx) better = true;
                if (better) {
                    prev_auc = best_auc;
                    best_auc = ba[i]; best_cons = bc[i]; best_idx = gi;
                    improvements++;
                    if (verbose) {
                        float *w = &ext_weights[gi * 3];
                        log_improvement_cuda(gi, ext_K, best_auc, prev_auc, best_cons,
                                             w[0], w[1], w[2]);
                    }
                }
            }
        }

        if (verbose) {
            log_summary_cuda("cuda_c (weights-file)", best_auc, best_cons,
                             ext_weights[best_idx*3+0], ext_weights[best_idx*3+1],
                             ext_weights[best_idx*3+2], 0.0, improvements);
        }
        result.best_auc = best_auc;
        result.best_consistency = best_cons;
        result.best_w0 = ext_weights[best_idx * 3 + 0];
        result.best_w1 = ext_weights[best_idx * 3 + 1];
        result.best_w2 = ext_weights[best_idx * 3 + 2];
        result.K_actual = ext_K;
        result.time_sec = 0.0;
        auto t0e = std::chrono::high_resolution_clock::now();
        result.time_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0e).count();
    } else if (search_mode == "grid") {
        result = run_grid_search(gpu, K, block_size, batch_size, grid_resolution, verbose);
    } else if (search_mode == "hybrid") {
        result = run_hybrid_search(gpu, K, seed, block_size, batch_size, verbose);
    } else {
        result = run_random_search(gpu, K, seed, block_size, batch_size, verbose);
    }

    // ── Output CSV ──
    printf("cuda_c,%s,%s,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d\n",
           search_mode.c_str(), mode.c_str(), K, result.K_actual, gpu.n_items,
           result.best_auc, result.best_consistency,
           result.best_w0, result.best_w1, result.best_w2,
           result.time_sec, seed, block_size);

    // ── Cleanup ──
    CUDA_CHECK(cudaFree(gpu.d_A));
    CUDA_CHECK(cudaFree(gpu.d_profiles));
    CUDA_CHECK(cudaFree(gpu.d_labels));
    if (!mode_full) CUDA_CHECK(cudaFree(gpu.d_B));

    return 0;
}
