/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1
#define GPU_DEVICE 0
#define TILE 32

#include "3mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU


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

    
// __global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E)
// {
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     int i = blockIdx.y * blockDim.y + threadIdx.y;

//     if ((i < _PB_NI) && (j < _PB_NJ))
//     {
//         E[i * NJ + j] = 0;
//         int k;
//         for(k=0; k < _PB_NK; k++)
//         {
//             E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
//         }
//     }
// }

__global__ void mm3_kernel1_tiled(int ni,int nj,int nk, const DATA_TYPE* __restrict__ A, const DATA_TYPE* __restrict__ B, DATA_TYPE* __restrict__ E)
{
    __shared__ DATA_TYPE shA[TILE][TILE];
    __shared__ DATA_TYPE shB[TILE][TILE];

    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;

    DATA_TYPE acc = 0;
    for (int t = 0; t < nk; t += TILE) {

    shA[threadIdx.y][threadIdx.x] = (row < ni && t+threadIdx.x < nk) ? A[row*NK + t+threadIdx.x] : 0;
    
    shB[threadIdx.y][threadIdx.x] = (t+threadIdx.y < nk && col < nj) ? B[(t+threadIdx.y)*NJ + col] : 0;
    
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k)
        acc += shA[threadIdx.y][k] * shB[k][threadIdx.x];
    __syncthreads();
    }

    if (row < ni && col < nj) 
        E[row*NJ + col] = acc;
}

    
// __global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F)
// {
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     int i = blockIdx.y * blockDim.y + threadIdx.y;

//     if ((i < _PB_NJ) && (j < _PB_NL))
//     {
//         F[i * NL + j] = 0;
//         int k;
//         for(k=0; k < _PB_NM; k++)
//         {
//             F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
//         }
//     }
// }
__global__ void mm3_kernel2_tiled(int nj,int nl,int nm, const DATA_TYPE* __restrict__ C, const DATA_TYPE* __restrict__ D, DATA_TYPE* __restrict__ F)
{
    __shared__ DATA_TYPE shC[TILE][TILE];
    __shared__ DATA_TYPE shD[TILE][TILE];

    int row = blockIdx.y*TILE + threadIdx.y;     // over NJ
    int col = blockIdx.x*TILE + threadIdx.x;     // over NL

    DATA_TYPE acc = 0;

    for (int t = 0; t < nm; t += TILE) {

    shC[threadIdx.y][threadIdx.x] = (row < nj && t+threadIdx.x < nm) ? C[row*NM + t+threadIdx.x] : 0;
    
    shD[threadIdx.y][threadIdx.x] = (t+threadIdx.y < nm && col < nl) ? D[(t+threadIdx.y)*NL + col] : 0;
    
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k)
        acc += shC[threadIdx.y][k] * shD[k][threadIdx.x];
    __syncthreads();
    }

    if (row < nj && col < nl) 
        F[row*NL + col] = acc;
}


    
// __global__ void mm3_kernel3(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
// {
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     int i = blockIdx.y * blockDim.y + threadIdx.y;

//     if ((i < _PB_NI) && (j < _PB_NL))
//     {
//         G[i * NL + j] = 0;
//         int k;
//         for(k=0; k < _PB_NJ; k++)
//         {
//             G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
//         }
//     }
// }
__global__ void mm3_kernel3_tiled(int ni,int nj,int nl, const DATA_TYPE* __restrict__ E, const DATA_TYPE* __restrict__ F, DATA_TYPE* __restrict__ G)
{
    __shared__ DATA_TYPE shE[TILE][TILE];
    __shared__ DATA_TYPE shF[TILE][TILE];

    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;

    DATA_TYPE acc = 0;

    for (int t = 0; t < nj; t += TILE) {

    shE[threadIdx.y][threadIdx.x] = (row < ni && t+threadIdx.x < nj) ? E[row*NJ + t+threadIdx.x] : 0;
    
    shF[threadIdx.y][threadIdx.x] = (t+threadIdx.y < nj && col < nl) ? F[(t+threadIdx.y)*NL + col] : 0;
    
    __syncthreads();
    
    #pragma unroll
    for (int k = 0; k < TILE; ++k)
        acc += shE[threadIdx.y][k] * shF[k][threadIdx.x];
    
        __syncthreads();
    }

    if (row < ni && col < nl) 
        G[row*NL + col] = acc;
}



/* Main computational kernel on CPU */
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


void mm3Cuda(int ni, int nj, int nk, int nl, int nm,
        DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
        DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
        DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
        DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
        DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl),
        DATA_TYPE POLYBENCH_2D(G_outputFromGpu,NI,NL,ni,nl))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;
    DATA_TYPE *D_gpu;
    DATA_TYPE *E_gpu;
    DATA_TYPE *F_gpu;
    DATA_TYPE *G_gpu;
    
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NM);
    cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NM * NL);
    cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL);
    cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice);
    cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
    cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);	
    
    // dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    // dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
    // dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
    // dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

    // /* Start timer. */
    // polybench_start_instruments;

    // mm3_kernel1<<<grid1,block>>>(ni, nj, nk, nl, nm, A_gpu, B_gpu, E_gpu);
    // cudaThreadSynchronize();
    // mm3_kernel2<<<grid2,block>>>(ni, nj, nk, nl, nm, C_gpu, D_gpu, F_gpu);
    // cudaThreadSynchronize();
    // mm3_kernel3<<<grid3,block>>>(ni, nj, nk, nl, nm, E_gpu, F_gpu, G_gpu);
    // cudaThreadSynchronize();

    dim3 block(TILE, TILE);
    dim3 g1((NJ+TILE-1)/TILE, (NI+TILE-1)/TILE);
    dim3 g2((NL+TILE-1)/TILE, (NJ+TILE-1)/TILE);
    dim3 g3((NL+TILE-1)/TILE, (NI+TILE-1)/TILE);

    polybench_start_instruments;
    mm3_kernel1_tiled<<<g1, block>>>(ni,nj,nk,A_gpu,B_gpu,E_gpu);
    cudaDeviceSynchronize();
    mm3_kernel2_tiled<<<g2, block>>>(nj,nl,nm,C_gpu,D_gpu,F_gpu);
    cudaDeviceSynchronize();
    mm3_kernel3_tiled<<<g3, block>>>(ni,nj,nl,E_gpu,F_gpu,G_gpu);
    cudaDeviceSynchronize();
    polybench_stop_instruments;

    /* Stop and print timer. */
    printf("GPU Time in seconds:\n");
    polybench_print_instruments;
    cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);
    
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(D_gpu);
    cudaFree(E_gpu);
    cudaFree(F_gpu);
    cudaFree(G_gpu);
}


/* DCE code. Must scan the entire live-out data.
Can be used also to check the correctness of the output. */
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

    mm3Cuda(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E), 
        POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

    #ifdef RUN_ON_CPU

        /* Start timer. */
        polybench_start_instruments;

        mm3_cpu(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E), 
            POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G));
    
        /* Stop and print timer. */
        printf("CPU Time in seconds:\n");
        polybench_stop_instruments;
        polybench_print_instruments;

        compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

    #else //print output to stderr so no dead code elimination

        print_array(ni, nl, POLYBENCH_ARRAY(G_outputFromGpu));

    #endif //RUN_ON_CPU


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

