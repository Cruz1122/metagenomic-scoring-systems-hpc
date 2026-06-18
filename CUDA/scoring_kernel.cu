/**
 * scoring_kernel.cu — Búsqueda de pesos en GPU.
 * Estrategias: random, grid
 * Modo: full (P=profiles@W, scores=A@P)
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

#define MAX_SAMPLES 4096
#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); \
        exit(2); \
    } \
} while(0)

// ── NPY reader ──────────────────────────────────────────────────────
static void read_npy(const char* path, std::vector<float>& data, int& d0, int& d1) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Error: cannot open %s\n", path); exit(1); }
    char magic[8]; f.read(magic, 8);
    if (magic[0]!=(char)0x93||magic[1]!='N'||magic[2]!='U'||magic[3]!='M'||magic[4]!='P'||magic[5]!='Y')
        { fprintf(stderr,"Not a npy (magic=%02x %02x %02x %02x %02x %02x): %s\n",
            (unsigned char)magic[0],(unsigned char)magic[1],(unsigned char)magic[2],
            (unsigned char)magic[3],(unsigned char)magic[4],(unsigned char)magic[5],path); exit(1); }
    uint8_t vmaj=(uint8_t)magic[6], vmin=(uint8_t)magic[7];
    int64_t hdr_len;
    if (vmaj==1) { uint16_t hl; f.read((char*)&hl,2); hdr_len=hl; }
    else { uint32_t hl; f.read((char*)&hl,4); hdr_len=hl; }
    std::string header((size_t)hdr_len,'\0'); f.read(&header[0],hdr_len);

    bool fortran = false;
    // Detect fortran_order
    size_t fp=header.find("'fortran_order'");if(fp==std::string::npos)fp=header.find("\"fortran_order\"");
    if(fp!=std::string::npos){
        size_t fc=header.find(':',fp+15);if(fc!=std::string::npos){
            size_t fs=header.find_first_not_of(" \"'",fc+1);if(fs!=std::string::npos){
                std::string fv=header.substr(fs,header.find_first_of(",}\n",fs)-fs);
                if(fv=="True")fortran=true;
            }
        }
    }

    std::vector<int64_t> shape;
    size_t sp=header.find("'shape'"); if(sp==std::string::npos)sp=header.find("\"shape\"");
    if(sp!=std::string::npos){
        size_t pa=header.find('(',sp); if(pa==std::string::npos)pa=header.find('[',sp);
        if(pa!=std::string::npos){
            size_t ep=header.find(')',pa); if(ep==std::string::npos)ep=header.find(']',pa);
            std::string sh=header.substr(pa+1,ep-pa-1);
            std::stringstream ss(sh); std::string tok;
            while(std::getline(ss,tok,',')){char*e=nullptr;int64_t v=strtoll(tok.c_str(),&e,10);if(e!=tok.c_str())shape.push_back(v);}
        }
    }
    int64_t nelem=1; for(auto s:shape)nelem*=s;
    if(shape.empty()){d0=0;d1=0;return;}

    int64_t offset=6+1+1+(vmaj==1?2:4)+hdr_len;
    f.seekg(offset);
    data.resize(nelem); f.read((char*)data.data(),nelem*sizeof(float));

    if(fortran&&shape.size()==2){
        int64_t rows=shape[0],cols=shape[1];
        std::vector<float> tmp(nelem);
        for(int64_t c=0;c<cols;c++) for(int64_t r=0;r<rows;r++) tmp[r*cols+c]=data[c*rows+r];
        data.swap(tmp);
    }

    if(shape.size()==1){d0=(int)shape[0];d1=1;}
    else if(shape.size()==2){d0=(int)shape[0];d1=(int)shape[1];}
    else{d0=(int)nelem;d1=1;}
}

static void read_npy_int(const char* path, std::vector<int>& data, int& sz) {
    std::ifstream f(path,std::ios::binary);
    if(!f.is_open()){fprintf(stderr,"Error: cannot open %s\n",path);exit(1);}
    char magic[6];f.read(magic,6);
    uint8_t vmaj,vmin;f.read((char*)&vmaj,1);f.read((char*)&vmin,1);
    int64_t hdr_len;
    if(vmaj==1){uint16_t hl;f.read((char*)&hl,2);hdr_len=hl;}
    else{uint32_t hl;f.read((char*)&hl,4);hdr_len=hl;}
    std::string header((size_t)hdr_len,'\0');f.read(&header[0],hdr_len);
    std::vector<int64_t> shape;
    size_t sp=header.find("'shape'");if(sp==std::string::npos)sp=header.find("\"shape\"");
    if(sp!=std::string::npos){
        size_t pa=header.find('(',sp);if(pa==std::string::npos)pa=header.find('[',sp);
        if(pa!=std::string::npos){
            size_t ep=header.find(')',pa);if(ep==std::string::npos)ep=header.find(']',pa);
            std::string sh=header.substr(pa+1,ep-pa-1);
            std::stringstream ss(sh);std::string tok;
            while(std::getline(ss,tok,',')){char*e=nullptr;int64_t v=strtoll(tok.c_str(),&e,10);if(e!=tok.c_str())shape.push_back(v);}
        }
    }
    int64_t nelem=1;for(auto s:shape)nelem*=s;
    int64_t offset=6+1+1+(vmaj==1?2:4)+hdr_len;
    f.seekg(offset);data.resize(nelem);f.read((char*)data.data(),nelem*sizeof(int));
    sz=(int)nelem;
}

// ── Device AUC & Consistency ────────────────────────────────────────
__device__ float auc_dev(const float* s, const int* y, int n) {
    int np=0,nn=0; for(int i=0;i<n;i++){if(y[i]==1)np++;else nn++;}
    if(np==0||nn==0)return 0.5f;
    float cor=0,ties=0,tot=(float)(np*nn);
    for(int i=0;i<n;i++){
        if(y[i]!=1)continue;
        for(int j=0;j<n;j++){
            if(y[j]!=0)continue;
            if(s[i]>s[j])cor+=1; else if(s[i]==s[j])ties+=1;
        }
    }
    return (cor+0.5f*ties)/tot;
}

__device__ float cons_dev(const float* s, const int* y, int n) {
    int idx[MAX_SAMPLES]; for(int i=0;i<n;i++)idx[i]=i;
    for(int i=1;i<n;i++){
        float ks=s[idx[i]]; int ki=idx[i]; int j=i-1;
        while(j>=0&&s[idx[j]]>ks){idx[j+1]=idx[j];j--;} idx[j+1]=ki;
    }
    int np=0,nn=0; for(int i=0;i<n;i++){if(y[i]==1)np++;else nn++;}
    int tp=np,tn=0; float best=0;
    for(int i=0;i<n;i++){
        float tpr=(np>0)?(float)tp/np:1.0f, tnr=(nn>0)?(float)tn/nn:1.0f;
        float ba=(tpr+tnr)*0.5f; if(ba>best)best=ba;
        if(y[idx[i]]==1)tp--;else tn++;
    }
    return best;
}

// ── Kernel ──────────────────────────────────────────────────────────
__global__ void eval_kernel(const float* A, const float* P, const int* y,
    const float* W, float* out_auc, float* out_cons, int ns, int ni, int K) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=K)return;
    float w0=W[idx*3],w1=W[idx*3+1],w2=W[idx*3+2];
    float sc[MAX_SAMPLES]; for(int s=0;s<ns;s++)sc[s]=0;
    for(int i=0;i<ni;i++){
        float p=P[i*3]*w0+P[i*3+1]*w1+P[i*3+2]*w2;
        for(int s=0;s<ns;s++)sc[s]+=A[s*ni+i]*p;
    }
    out_auc[idx]=auc_dev(sc,y,ns);
    out_cons[idx]=cons_dev(sc,y,ns);
}

// ── Host helpers ────────────────────────────────────────────────────
static void gen_weights(std::vector<float>& w, int K, unsigned seed) {
    w.resize(K*3);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(0,1);
    for(int i=0;i<K;i++){
        float x1=-log(1-d(rng)+1e-37f),x2=-log(1-d(rng)+1e-37f),x3=-log(1-d(rng)+1e-37f);
        float s=x1+x2+x3; w[i*3]=x1/s; w[i*3+1]=x2/s; w[i*3+2]=x3/s;
    }
}

static int gen_grid(std::vector<float>& w, int res) {
    w.clear();
    for(int i=0;i<=res;i++) for(int j=0;j<=res-i;j++){
        int k=res-i-j; w.push_back((float)i/res); w.push_back((float)j/res); w.push_back((float)k/res);
    }
    return (int)w.size()/3;
}

static int res_for_K(int K){return std::max((int)sqrt(2.0*K),1);}

// ── ANSI logger ─────────────────────────────────────────────────────
#define BLD "\x1b[1m"
#define GRN "\x1b[32m"
#define YLW "\x1b[33m"
#define CYN "\x1b[36m"
#define MGT "\x1b[35m"
#define GLD "\x1b[1;33m"
#define RST "\x1b[0m"

static void log_hdr(const char* impl, int N, long K) {
    fprintf(stderr,"\n" BLD CYN "  +- %s ",impl);
    for(int i=0;i<45-(int)strlen(impl);i++)fprintf(stderr,"-");
    fprintf(stderr,"+" RST "\n");
    fprintf(stderr,CYN "  |" RST "  " BLD "BUSQUEDA DE PESOS OPTIMOS" RST "\n");
    fprintf(stderr,CYN "  |" RST "  items (N) ... %d\n",N);
    fprintf(stderr,CYN "  |" RST "  candidatos   %ld\n",K);
    fprintf(stderr,CYN "  +");for(int i=0;i<55;i++)fprintf(stderr,"-");fprintf(stderr,"+" RST "\n\n");
}

static void log_imp(long i, long K, double auc, double prev, double cons, double w1, double w2, double w3) {
    if(prev<0) fprintf(stderr,"  " GLD "->" RST "  " BLD "AUC %.6f" RST "  " GRN "(initial)" RST "  iter %ld/%ld  consist=%.4f  w=[%.4f %.4f %.4f]\n",auc,i,K,cons,w1,w2,w3);
    else fprintf(stderr,"  " GLD "->" RST "  " BLD "AUC %.6f" RST "  " GRN "(+%.6f)" RST "  iter %ld/%ld  consist=%.4f  w=[%.4f %.4f %.4f]\n",auc,auc-prev,i,K,cons,w1,w2,w3);
}

static void log_sum(const char* impl, double auc, double cons, double w1, double w2, double w3, double t, int impr) {
    fprintf(stderr,"\n" BLD MGT "  +- MEJOR RESULTADO ");
    for(int i=0;i<38;i++)fprintf(stderr,"-");fprintf(stderr,"+" RST "\n");
    fprintf(stderr,BLD MGT "  |" RST "  implementacion ... %s\n",impl);
    fprintf(stderr,BLD MGT "  |" RST "  mejoras        ... %d\n",impr);
    fprintf(stderr,BLD MGT "  |" RST "  AUC            ... " GLD "%.9f" RST "\n",auc);
    fprintf(stderr,BLD MGT "  |" RST "  consistencia   ... %.4f\n",cons);
    fprintf(stderr,BLD MGT "  |" RST "  pesos W        ... [%.9f, %.9f, %.9f]\n",w1,w2,w3);
    fprintf(stderr,BLD MGT "  |" RST "  suma W         ... %.9f\n",w1+w2+w3);
    fprintf(stderr,BLD MGT "  |" RST "  tiempo         ... " CYN "%.6f s" RST "\n",t);
    fprintf(stderr,BLD MGT "  +");for(int i=0;i<55;i++)fprintf(stderr,"-");fprintf(stderr,"+" RST "\n\n");
}

// ── Main ────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int K=10000, seed=42, bs=256, gr=0, vb=0;
    std::string sm="random", dd="data/npy";
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--k")==0&&i+1<argc)K=atoi(argv[++i]);
        else if(strcmp(argv[i],"--seed")==0&&i+1<argc)seed=atoi(argv[++i]);
        else if(strcmp(argv[i],"--search")==0&&i+1<argc)sm=argv[++i];
        else if(strcmp(argv[i],"--data-dir")==0&&i+1<argc)dd=argv[++i];
        else if(strcmp(argv[i],"--block-size")==0&&i+1<argc)bs=atoi(argv[++i]);
        else if(strcmp(argv[i],"--grid-resolution")==0&&i+1<argc)gr=atoi(argv[++i]);
        else if(strcmp(argv[i],"--verbose")==0)vb=1;
    }

    // Load data
    std::vector<float> hA, hP; std::vector<int> hY;
    int ns=0,ni=0,nc=0,ny=0;
    read_npy((dd+"/matrix_A.npy").c_str(),hA,ns,ni);
    read_npy((dd+"/profiles_TSF.npy").c_str(),hP,ni,nc);
    read_npy_int((dd+"/labels.npy").c_str(),hY,ny);
    if(ns>MAX_SAMPLES){fprintf(stderr,"Error: samples=%d > %d\n",ns,MAX_SAMPLES);return 1;}
    if(vb)fprintf(stderr,"Data: A(%d,%d) profiles(%d,%d) labels(%d)\n",ns,ni,ni,nc,ny);

    // Copy to GPU
    float *dA, *dP; int *dY;
    CUDA_CHECK(cudaMalloc(&dA,hA.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP,hP.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY,hY.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA,hA.data(),hA.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dP,hP.data(),hP.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY,hY.data(),hY.size()*sizeof(int),cudaMemcpyHostToDevice));

    // Generate weights
    std::vector<float> allW; int actualK=K;
    if(sm=="grid"){actualK=gen_grid(allW,gr?gr:res_for_K(K));if(vb)log_hdr("cuda_c (grid)",ni,actualK);}
    else{gen_weights(allW,K,seed);if(vb)log_hdr("cuda_c (random)",ni,K);}

    // Evaluate in batches
    int batch=1000000;
    std::vector<float> ba(batch), bc(batch);
    float best_auc=-1,best_cons=-1,prev=-1;
    int best_idx=-1, impr=0;
    auto t0=std::chrono::high_resolution_clock::now();

    for(int off=0;off<actualK;off+=batch){
        int bk=std::min(batch,actualK-off);
        float *dW,*dAuc,*dCon;
        CUDA_CHECK(cudaMalloc(&dW,bk*3*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dAuc,bk*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCon,bk*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dW,&allW[off*3],bk*3*sizeof(float),cudaMemcpyHostToDevice));
        int grid=(bk+bs-1)/bs;
        eval_kernel<<<grid,bs>>>(dA,dP,dY,dW,dAuc,dCon,ns,ni,bk);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ba.data(),dAuc,bk*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bc.data(),dCon,bk*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dW));CUDA_CHECK(cudaFree(dAuc));CUDA_CHECK(cudaFree(dCon));

        for(int i=0;i<bk;i++){
            int gi=off+i;
            bool better=false;
            if(ba[i]>best_auc)better=true;
            else if(ba[i]==best_auc&&bc[i]>best_cons)better=true;
            else if(ba[i]==best_auc&&bc[i]==best_cons&&gi<best_idx)better=true;
            if(better){
                prev=best_auc;best_auc=ba[i];best_cons=bc[i];best_idx=gi;impr++;
                if(vb)log_imp(gi,actualK,best_auc,prev,best_cons,allW[gi*3],allW[gi*3+1],allW[gi*3+2]);
            }
        }
    }

    auto t1=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration<double>(t1-t0).count();

    if(vb)log_sum("cuda_c",best_auc,best_cons,allW[best_idx*3],allW[best_idx*3+1],allW[best_idx*3+2],elapsed,impr);

    printf("cuda_c,%s,full,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d\n",
           sm.c_str(),K,actualK,ni,best_auc,best_cons,allW[best_idx*3],allW[best_idx*3+1],allW[best_idx*3+2],elapsed,seed,bs);

    CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dP));CUDA_CHECK(cudaFree(dY));
    return 0;
}
