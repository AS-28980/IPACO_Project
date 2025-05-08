/*****************************************************************************
 * 2mm_overlap_streams.cu – Two‑stream FP32 + FP64 execution of PolyBench 2MM
 *
 *   • TILE 32×32 (identical to your last version)
 *   • Stream 0  →  FP32 kernels (hits FP32 pipes)
 *   • Stream 1  →  FP64 kernels (hits FP64 pipes)
 *
 *   Both streams run *concurrently*; the runtime interleaves warps so that
 *   FP32 and FP64 functional units are busy at the same time.
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define POLYBENCH_TIME 1
#define TILE 32
#define PERCENT_DIFF_ERROR_THRESHOdim 0.05
#define GPU_DEVICE 0

#include "2mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

template<typename T>
__global__ void mm2_kernel1(int ni,int nj,int nk, T alpha, const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ tmp, int row_offset)
/*  tmp = α · A · B   (rows starting at row_offset) */
{
    __shared__ T shA[TILE][TILE];
    __shared__ T shB[TILE][TILE];

    int row = row_offset + blockIdx.y*TILE + threadIdx.y;   // <── slice
    int col = blockIdx.x*TILE + threadIdx.x;

    T acc = 0;

    for (int t=0; t<nk; t+=TILE) {

        int aCol = t + threadIdx.x;
        shA[threadIdx.y][threadIdx.x] = (row<ni && aCol<nk) ? A[row*NK + aCol] : T(0);

        int bRow = t + threadIdx.y;
        shB[threadIdx.y][threadIdx.x] = (bRow<nk && col<nj) ? B[bRow*NJ + col] : T(0);

        __syncthreads();
        #pragma unroll
        for (int k=0;k<TILE;++k)
            acc += shA[threadIdx.y][k]*shB[k][threadIdx.x];
        __syncthreads();
    }

    if (row<ni && col<nj)
        tmp[row*NJ + col] = alpha*acc;
}

template<typename T>
__global__ void mm2_kernel2(int ni,int nj,int nk,int nl, T beta, const T* __restrict__ tmp, const T* __restrict__ C, T* __restrict__ D, int row_offset)
/*  D = β·D + tmp·C   (rows starting at row_offset) */
{
    __shared__ T shT[TILE][TILE];
    __shared__ T shC[TILE][TILE];

    int row = row_offset + blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;

    T acc = 0;
    for (int t=0; t<nj; t+=TILE) {

        int tCol = t + threadIdx.x;
        shT[threadIdx.y][threadIdx.x] = (row<ni && tCol<nj) ? tmp[row*NJ + tCol] : T(0);

        int cRow = t + threadIdx.y;
        shC[threadIdx.y][threadIdx.x] = (cRow<nj && col<nl) ? C[cRow*NL + col] : T(0);

        __syncthreads();
        #pragma unroll
        for (int k=0;k<TILE;++k)
            acc += shT[threadIdx.y][k]*shC[k][threadIdx.x];
        __syncthreads();
    }

    if (row<ni && col<nl)
        D[row*NL + col] = D[row*NL + col]*beta + acc;
}

/* ------------------------------------------------------------------------- */
/*                           host‑side helpers                               */
/* ------------------------------------------------------------------------- */
template<typename T>
T* dmalloc(size_t nbytes){ 
    T* p; 
    cudaMalloc(&p,nbytes); 
    return p; 
}

template<typename Tout,typename Tin>
__global__ void cast_kernel(Tin* in, Tout* out, size_t n){
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx<n) 
        out[idx] = static_cast<Tout>(in[idx]);
}

__global__ void downcast_D64_to_F32(const double* __restrict__ src64, float* __restrict__ dst32, int start_row, int rows, int dim, int cols)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;

    if (idx >= size) 
        return;

    int r = idx / cols;
    int c = idx % cols;
    int gRow = start_row + r;          // global row in full matrix
    dst32[gRow*dim + c] = static_cast<float>(src64[gRow*dim + c]);
}


void mm2Cuda_twoStreams(int ni,int nj,int nk,int nl, float alpha32,float beta32, float*  A_h, float*  B_h, float*  C_h, float*  D_h, float*  D_out_h)
{
    /* ---------- allocate & copy (unchanged) ---------------------------- */
    size_t szA_f = NI*NK*sizeof(float);
    size_t szB_f = NK*NJ*sizeof(float);
    size_t szC_f = NL*NJ*sizeof(float);
    size_t szD_f = NI*NL*sizeof(float);
    size_t szT_f = NI*NJ*sizeof(float);

    size_t szA_d = NI*NK*sizeof(double);
    size_t szB_d = NK*NJ*sizeof(double);
    size_t szC_d = NL*NJ*sizeof(double);
    size_t szD_d = NI*NL*sizeof(double);
    size_t szT_d = NI*NJ*sizeof(double);

    float  *A_f=dmalloc<float>(szA_f), 
           *B_f=dmalloc<float>(szB_f),
           *C_f=dmalloc<float>(szC_f), 
           *D_f=dmalloc<float>(szD_f),
           *T_f=dmalloc<float>(szT_f);

    double *A_d=dmalloc<double>(szA_d), 
           *B_d=dmalloc<double>(szB_d),
           *C_d=dmalloc<double>(szC_d), 
           *D_d=dmalloc<double>(szD_d),
           *T_d=dmalloc<double>(szT_d);

    cudaMemcpy(A_f,A_h,szA_f,cudaMemcpyHostToDevice);
    cudaMemcpy(B_f,B_h,szB_f,cudaMemcpyHostToDevice);
    cudaMemcpy(C_f,C_h,szC_f,cudaMemcpyHostToDevice);
    cudaMemcpy(D_f,D_h,szD_f,cudaMemcpyHostToDevice);

    /* ---- Copy matrices from Float to Double by TypeCasting -------------------------- */
    int threads=256;
    int nA=NI*NK, nB=NK*NJ, nC=NL*NJ, nD=NI*NL;
    cast_kernel<<<(nA+threads-1)/threads,threads>>>(A_f,A_d,nA);
    cast_kernel<<<(nB+threads-1)/threads,threads>>>(B_f,B_d,nB);
    cast_kernel<<<(nC+threads-1)/threads,threads>>>(C_f,C_d,nC);
    cast_kernel<<<(nD+threads-1)/threads,threads>>>(D_f,D_d,nD);

    /* ------------- split rows ------- */
    const int mid = 31*ni/32;           
    const int rows32 = mid;
    const int rows64 = ni - mid;

    dim3 block(TILE,TILE);

    /* grids for the two slices */
    dim3 g32_ab((nj+TILE-1)/TILE, (rows32+TILE-1)/TILE);
    dim3 g64_ab((nj+TILE-1)/TILE, (rows64+TILE-1)/TILE);

    dim3 g32_dc((nl+TILE-1)/TILE, (rows32+TILE-1)/TILE);
    dim3 g64_dc((nl+TILE-1)/TILE, (rows64+TILE-1)/TILE);

    polybench_start_instruments;

    /* ------------ create two streams ---------------------------------- */
    cudaStream_t s32, s64;
    cudaStreamCreate(&s32);
    cudaStreamCreate(&s64);

    /* ----------------  FP32 half –– rows 0 … mid‑1  -------------------- */
    mm2_kernel1<float><<<g32_ab,block,0,s32>>>(ni,nj,nk, alpha32, A_f,B_f,T_f, /*row_offset=*/0);

    mm2_kernel2<float><<<g32_dc,block,0,s32>>>(ni,nj,nk,nl, beta32, T_f,C_f,D_f, /*row_offset=*/0);

    /* ----------------  FP64 half –– rows mid … NI‑1  ------------------- */
    const double alpha64 = static_cast<double>(alpha32);
    const double beta64  = static_cast<double>(beta32);

    mm2_kernel1<double><<<g64_ab,block,0,s64>>>(ni,nj,nk, alpha64, A_d,B_d,T_d, /*row_offset=*/mid);

    mm2_kernel2<double><<<g64_dc,block,0,s64>>>(ni,nj,nk,nl, beta64, T_d,C_d,D_d, /*row_offset=*/mid);

    /* ----------------  wait & copy result ------------------------------ */
    cudaStreamSynchronize(s32);
    cudaStreamSynchronize(s64);

    printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

    /* ----- convert the FP64 half into D_f ----------------------------- */
    int elems64 = rows64 * nl;
    int threadsCast = 256;
    downcast_D64_to_F32<<<(elems64+threadsCast-1)/threadsCast, threadsCast>>>(D_d, D_f, mid, rows64, nl, nl);

    cudaMemcpy(D_out_h, D_f, szD_f, cudaMemcpyDeviceToHost);

    /* ----------------  clean‑up  --------------------------------------- */
    cudaFree(A_f); 
    cudaFree(B_f); 
    cudaFree(C_f); 
    cudaFree(D_f); 
    cudaFree(T_f);
    cudaFree(A_d); 
    cudaFree(B_d); 
    cudaFree(C_d); 
    cudaFree(D_d); 
    cudaFree(T_d);
    cudaStreamDestroy(s32); 
    cudaStreamDestroy(s64);
}


void mm2_cpu(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta,
                    DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
                    DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl)){
                            
    int i, j, k;

    /* D := alpha*A*B*C + beta*D */
    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NJ; j++)
        {
            tmp[i][j] = 0;
            for (k = 0; k < _PB_NK; ++k)
            {
                tmp[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }

    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NL; j++)
        {
            D[i][j] *= beta;
            for (k = 0; k < _PB_NJ; ++k)
            {
                D[i][j] += tmp[i][k] * C[k][j];
            }
        }
    }
}


/* DCE code. Must scan the entire live-out data.
Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl, DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
    {
    int i, j;

    for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
    if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
    fprintf (stderr, "\n");
    }

    void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), 
        DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj), 
        DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
    {
    int i, j;

    *alpha = 32412;
    *beta = 2123;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE) i*j) / NI;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
        }
    }

    for (i = 0; i < nl; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nl; j++)
        {
            D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;	
        }
    }
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI, NL, ni, nl))
{
    int i,j,fail;
    fail = 0;

    for (i=0; i < ni; i++)
    {
        for (j=0; j < nl; j++)
        {
            if (percentDiff(D[i][j], D_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOdim)
            {
                fail++;
            }
        }
    }

    // print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshodim of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOdim, fail);
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice( GPU_DEVICE );
}

int main()
{
    int ni=NI,nj=NJ,nk=NK,nl=NL;
    float alpha=32412.0f, beta=2123.0f;

    POLYBENCH_2D_ARRAY_DECL(A,float,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,float,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,float,NL,NJ,nl,nj);
    POLYBENCH_2D_ARRAY_DECL(D,float,NI,NL,ni,nl);
    POLYBENCH_2D_ARRAY_DECL(tmp,float,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(D_gpu,float,NI,NL,ni,nl);

    init_array(ni,nj,nk,nl,&alpha,&beta,
                POLYBENCH_ARRAY(A),POLYBENCH_ARRAY(B),
                POLYBENCH_ARRAY(C),POLYBENCH_ARRAY(D));

    GPU_argv_init();

    mm2Cuda_twoStreams(ni,nj,nk,nl, alpha, beta,
    (float*)POLYBENCH_ARRAY(A),
    (float*)POLYBENCH_ARRAY(B),
    (float*)POLYBENCH_ARRAY(C),
    (float*)POLYBENCH_ARRAY(D),
    (float*)POLYBENCH_ARRAY(D_gpu));

    polybench_start_instruments;
    mm2_cpu(ni, nj, nk, nl,
        alpha, beta,
        POLYBENCH_ARRAY(tmp),
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(D));

    polybench_stop_instruments;
    polybench_print_instruments;


    compareResults(ni,nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_gpu));
    return 0;
}

/* … your code … */

#include "../../common/polybench.c"   //  <-- make sure the path is correct


/* ------------------------------------------------------------------------- */
/*  (Your odim helpers – init_array, mm2_cpu, etc. – go here unchanged.)      */
/* ------------------------------------------------------------------------- */
