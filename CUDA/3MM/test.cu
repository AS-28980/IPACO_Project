/*****************************************************************************
 * 3mm_overlap_streams.cu  –  FP32 + FP64 two‑stream execution of PolyBench 3MM
 *
 *  ──────────────────────────────────────────────────────────────────────────
 *      • TILE 32×32 (same as your tiled baseline)
 *      • Stream S32 : rows 0 … mid‑1     →  FP32 kernels
 *      • Stream S64 : rows mid … NI‑1    →  FP64 kernels
 *
 *    compute flow
 *    ─────────────
 *       S32 :  E32‑upper     →  G32‑upper
 *       S64 :  cast(F32→64)  →  E64‑lower  →  G64‑lower
 *
 *    final step
 *       down‑cast G64‑lower → G32‑lower   (single kernel)
 *
 ****************************************************************************/

 #include <cstdio>
 #include <cuda.h>
 
 #define POLYBENCH_TIME 1
 #define TILE 32
 #define GPU_DEVICE 0
 #define PERCENT_DIFF_ERROR_THRESHOLD 0.05
 
 #include "3mm.cuh"
 #include "../../common/polybench.h"
 #include "../../common/polybenchUtilFuncts.h"

 void init_array(int ni, int nj, int nk, int nl, int nm, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), 
        DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm), DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
    int i, j;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE) i*j) / ni;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
        }
    }

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nm; j++)
        {
            C[i][j] = ((DATA_TYPE) i*(j+3)) / nl;
        }
    }

    for (i = 0; i < nm; i++)
    {
        for (j = 0; j < nl; j++)
        {
            D[i][j] = ((DATA_TYPE) i*(j+2)) / nk;
        }
    }
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
    int i,j,fail;
    fail = 0;

    for (i=0; i < ni; i++)
    {
        for (j=0; j < nl; j++)
        {
            if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;				
            }
        }
    }
    
    // print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice( GPU_DEVICE );
}
 
 /* ───────────────────── tiled kernels (with row‑slice support) ─────────── */
 template<typename T>
 __global__ void k1_tiled_slice(int ni,int nj,int nk,
                                const T* __restrict__ A,
                                const T* __restrict__ B,
                                T* __restrict__ E,
                                int row_off)
 /* E = A·B  (rows row_off … )  ──  E ∈ [NI×NJ] */
 {
   __shared__ T shA[TILE][TILE], shB[TILE][TILE];
   int row = row_off + blockIdx.y*TILE + threadIdx.y;
   int col =           blockIdx.x*TILE + threadIdx.x;
 
   T acc = 0;
   for(int t=0;t<nk;t+=TILE){
     int aCol = t + threadIdx.x;
     shA[threadIdx.y][threadIdx.x] = (row<ni && aCol<nk) ? A[row*NK + aCol] : T(0);
     int bRow = t + threadIdx.y;
     shB[threadIdx.y][threadIdx.x] = (bRow<nk && col<nj) ? B[bRow*NJ + col] : T(0);
     __syncthreads();
     #pragma unroll
     for(int k=0;k<TILE;++k) acc += shA[threadIdx.y][k]*shB[k][threadIdx.x];
     __syncthreads();
   }
   if(row<ni && col<nj) E[row*NJ+col]=acc;
 }
 
 template<typename T>
 __global__ void k2_tiled(int nj,int nl,int nm,
                          const T* __restrict__ C,
                          const T* __restrict__ D,
                          T* __restrict__ F)
 /* F = C·D   (whole matrix, NJ×NL)                                      */
 {
   __shared__ T shC[TILE][TILE], shD[TILE][TILE];
   int row = blockIdx.y*TILE + threadIdx.y;
   int col = blockIdx.x*TILE + threadIdx.x;
   T acc=0;
   for(int t=0;t<nm;t+=TILE){
     int cCol=t+threadIdx.x;
     shC[threadIdx.y][threadIdx.x]=(row<nj&&cCol<nm)?C[row*NM+cCol]:T(0);
     int dRow=t+threadIdx.y;
     shD[threadIdx.y][threadIdx.x]=(dRow<nm&&col<nl)?D[dRow*NL+col]:T(0);
     __syncthreads();
     #pragma unroll
     for(int k=0;k<TILE;++k) acc+=shC[threadIdx.y][k]*shD[k][threadIdx.x];
     __syncthreads();
   }
   if(row<nj&&col<nl) F[row*NL+col]=acc;
 }
 
 template<typename T>
 __global__ void k3_tiled_slice(int ni,int nj,int nl,
                                const T* __restrict__ E,
                                const T* __restrict__ F,
                                T* __restrict__ G,
                                int row_off)
 /* G = E·F  (rows row_off … )                                           */
 {
   __shared__ T shE[TILE][TILE], shF[TILE][TILE];
   int row = row_off + blockIdx.y*TILE + threadIdx.y;
   int col =           blockIdx.x*TILE + threadIdx.x;
   T acc=0;
   for(int t=0;t<nj;t+=TILE){
     int eCol=t+threadIdx.x;
     shE[threadIdx.y][threadIdx.x]=(row<ni&&eCol<nj)?E[row*NJ+eCol]:T(0);
     int fRow=t+threadIdx.y;
     shF[threadIdx.y][threadIdx.x]=(fRow<nj&&col<nl)?F[fRow*NL+col]:T(0);
     __syncthreads();
     #pragma unroll
     for(int k=0;k<TILE;++k) acc+=shE[threadIdx.y][k]*shF[k][threadIdx.x];
     __syncthreads();
   }
   if(row<ni&&col<nl) G[row*NL+col]=acc;
 }
 
 /* ───────────────────── utility cast kernels ───────────────────────────── */
 template<typename Tout,typename Tin>
 __global__ void cast(const Tin* in, Tout* out, size_t n){
   size_t i=blockIdx.x*blockDim.x+threadIdx.x;
   if(i<n) out[i]=static_cast<Tout>(in[i]);
 }
 
 /* down‑cast the lower half of G64 into G32 */
 __global__ void downcast_G64_into_G32(const double* src64, float* dst32,
                                       int start_row,int rows,int ld,int cols){
   int idx = blockIdx.x*blockDim.x+threadIdx.x;
   int total = rows*cols;
   if(idx>=total) return;
   int r=idx/cols, c=idx%cols;
   int gRow = start_row+r;
   dst32[gRow*ld+c] = static_cast<float>(src64[gRow*ld+c]);
 }

 void mm3_cpu(int ni, int nj, int nk, int nl, int nm,
    DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
    DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
    DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
    DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
    DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
int i, j, k;

/* E := A*B */
for (i = 0; i < _PB_NI; i++)
{
    for (j = 0; j < _PB_NJ; j++)
    {
        E[i][j] = 0;
        for (k = 0; k < _PB_NK; ++k)
        {
            E[i][j] += A[i][k] * B[k][j];
        }
    }
}

/* F := C*D */
for (i = 0; i < _PB_NJ; i++)
{
    for (j = 0; j < _PB_NL; j++)
    {
        F[i][j] = 0;
        for (k = 0; k < _PB_NM; ++k)
        {
            F[i][j] += C[i][k] * D[k][j];
        }
    }
}

/* G := E*F */
for (i = 0; i < _PB_NI; i++)
{
    for (j = 0; j < _PB_NL; j++)
    {
        G[i][j] = 0;
        for (k = 0; k < _PB_NJ; ++k)
        {
            G[i][j] += E[i][k] * F[k][j];
        }
    }
}
}
 
 /* ───────────────────── two‑stream driver ──────────────────────────────── */
 /* ─────────── two‑stream driver (fixed dependency on F) ─────────── */
void mm3Cuda_twoStreams(int ni,int nj,int nk,int nl,int nm,
    float* A_h,float* B_h,float* C_h,float* D_h,
    float* G_out_h)
{
/* ------------ device buffers (unchanged) ---------------------- */
size_t sz_Af=NI*NK*sizeof(float),   sz_Bf=NK*NJ*sizeof(float);
size_t sz_Cf=NJ*NM*sizeof(float),   sz_Df=NM*NL*sizeof(float);
size_t sz_Ef=NI*NJ*sizeof(float),   sz_Ff=NJ*NL*sizeof(float);
size_t sz_Gf=NI*NL*sizeof(float);

size_t sz_Ad=NI*NK*sizeof(double),  sz_Bd=NK*NJ*sizeof(double);
size_t sz_Cd=NJ*NM*sizeof(double),  sz_Dd=NM*NL*sizeof(double);
size_t sz_Ed=NI*NJ*sizeof(double),  sz_Fd=NJ*NL*sizeof(double);
size_t sz_Gd=NI*NL*sizeof(double);

auto dmallocF=[&](size_t s){ float  *p; cudaMalloc(&p,s); return p; };
auto dmallocD=[&](size_t s){ double *p; cudaMalloc(&p,s); return p; };

float  *A_f=dmallocF(sz_Af), *B_f=dmallocF(sz_Bf),
*C_f=dmallocF(sz_Cf), *D_f=dmallocF(sz_Df),
*E_f=dmallocF(sz_Ef), *F_f=dmallocF(sz_Ff),
*G_f=dmallocF(sz_Gf);

double *A_d=dmallocD(sz_Ad), *B_d=dmallocD(sz_Bd),
*C_d=dmallocD(sz_Cd), *D_d=dmallocD(sz_Dd),
*E_d=dmallocD(sz_Ed), *F_d=dmallocD(sz_Fd),
*G_d=dmallocD(sz_Gd);

cudaMemcpy(A_f,A_h,sz_Af,cudaMemcpyHostToDevice);
cudaMemcpy(B_f,B_h,sz_Bf,cudaMemcpyHostToDevice);
cudaMemcpy(C_f,C_h,sz_Cf,cudaMemcpyHostToDevice);
cudaMemcpy(D_f,D_h,sz_Df,cudaMemcpyHostToDevice);

/* ---- cast inputs once ------------------------------------------------ */
int threads=256;
cast<<<(NI*NK+threads-1)/threads,threads>>>(A_f,A_d,NI*NK);
cast<<<(NK*NJ+threads-1)/threads,threads>>>(B_f,B_d,NK*NJ);
cast<<<(NJ*NM+threads-1)/threads,threads>>>(C_f,C_d,NJ*NM);
cast<<<(NM*NL+threads-1)/threads,threads>>>(D_f,D_d,NM*NL);

/* ---- launch parameters ---------------------------------------------- */
dim3 block(TILE,TILE);
const int mid = ni/2;
const int rows32 = mid, rows64 = ni-mid;

dim3 gE32((NJ+TILE-1)/TILE, (rows32+TILE-1)/TILE);
dim3 gE64((NJ+TILE-1)/TILE, (rows64+TILE-1)/TILE);
dim3 gF  ((NL+TILE-1)/TILE, (NJ+TILE-1)/TILE);
dim3 gG32((NL+TILE-1)/TILE, (rows32+TILE-1)/TILE);
dim3 gG64((NL+TILE-1)/TILE, (rows64+TILE-1)/TILE);

cudaStream_t S32,S64;
cudaStreamCreate(&S32);
cudaStreamCreate(&S64);

/* 1)   E‑slices in parallel, F (full) in FP32 -------------------------- */
k1_tiled_slice<float ><<<gE32,block,0,S32>>>(ni,nj,nk,A_f,B_f,E_f,0);
k1_tiled_slice<double><<<gE64,block,0,S64>>>(ni,nj,nk,A_d,B_d,E_d,mid);

k2_tiled<float><<<gF,block,0,S32>>>(nj,nl,nm,C_f,D_f,F_f);

/* 2)   wait for F_f, then cast it to double (needed by G64) ------------ */
cudaStreamSynchronize(S32);                  // F_f ready
cast<<<(NJ*NL+threads-1)/threads,threads>>>(F_f,F_d,NJ*NL);

/* 3)   down‑cast E64 rows so G32 can read a single FP32 matrix --------- */
cudaStreamSynchronize(S64);                  // E_d ready
int elem64E = rows64*nj;
cast<<<(elem64E+threads-1)/threads,threads>>>
(E_d,E_f,elem64E);   // simple 1‑kernel full‑matrix cast

/* 4)   G‑slices (concurrent again) ------------------------------------- */
k3_tiled_slice<float ><<<gG32,block,0,S32>>>(ni,nj,nl,E_f,F_f,G_f,0);
k3_tiled_slice<double><<<gG64,block,0,S64>>>(ni,nj,nl,E_d,F_d,G_d,mid);

cudaStreamSynchronize(S32);
cudaStreamSynchronize(S64);

/* 5)   merge: down‑cast lower half of G_d into G_f --------------------- */
int elems64G = rows64*nl;
downcast_G64_into_G32<<<(elems64G+threads-1)/threads,threads>>>
(G_d,G_f,mid,rows64,nl,nl);

cudaMemcpy(G_out_h,G_f,sz_Gf,cudaMemcpyDeviceToHost);

/* clean‑up ------------------------------------------------------------- */
cudaFree(A_f);cudaFree(B_f);cudaFree(C_f);cudaFree(D_f);
cudaFree(E_f);cudaFree(F_f);cudaFree(G_f);
cudaFree(A_d);cudaFree(B_d);cudaFree(C_d);cudaFree(D_d);
cudaFree(E_d);cudaFree(F_d);cudaFree(G_d);
cudaStreamDestroy(S32); cudaStreamDestroy(S64);
}

 
 static
 void print_array(int ni, int nl,
         DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
 {
 int i, j;
 
 for (i = 0; i < ni; i++)
     for (j = 0; j < nl; j++) {
     fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
     if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
     }
 fprintf (stderr, "\n");
 }
 
 
 int main(int argc, char** argv)
 {
     int ni = NI;
     int nj = NJ;
     int nk = NK;
     int nl = NL;
     int nm = NM;
 
     /* Variable declaration/allocation. */
     POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
     POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
     POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
     POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
     POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
     POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
     POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
     POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);
 
     init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
 
     GPU_argv_init();
 
     polybench_start_instruments;
     mm3Cuda_twoStreams(ni, nj, nk, nl, nm, (float*)POLYBENCH_ARRAY(A), (float*)POLYBENCH_ARRAY(B), (float*)POLYBENCH_ARRAY(C), (float*)POLYBENCH_ARRAY(D), (float*)POLYBENCH_ARRAY(G_outputFromGpu));
     polybench_stop_instruments;
        polybench_print_instruments;
 
    //  #ifdef RUN_ON_CPU
 
         /* Start timer. */
         polybench_start_instruments;
 
         mm3_cpu(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E), 
             POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G));
     
         /* Stop and print timer. */
         printf("CPU Time in seconds:\n");
         polybench_stop_instruments;
         polybench_print_instruments;
 
         compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));
 
    //  #endif //RUN_ON_CPU
 
 
     POLYBENCH_FREE_ARRAY(A);
     POLYBENCH_FREE_ARRAY(B);
     POLYBENCH_FREE_ARRAY(C);
     POLYBENCH_FREE_ARRAY(D);
     POLYBENCH_FREE_ARRAY(E);
     POLYBENCH_FREE_ARRAY(F);
     POLYBENCH_FREE_ARRAY(G);
     POLYBENCH_FREE_ARRAY(G_outputFromGpu);
 
     return 0;
 }
 
 #include "../../common/polybench.c"
 
 
 
 