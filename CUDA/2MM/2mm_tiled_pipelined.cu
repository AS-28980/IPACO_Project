/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define TILE DIM_THREAD_BLOCK_X

#include "2mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU


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
            if (percentDiff(D[i][j], D_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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


// __global__ void mm2_kernel1(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *A, DATA_TYPE *B)
// {
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     int i = blockIdx.y * blockDim.y + threadIdx.y;

//     if ((i < _PB_NI) && (j < _PB_NJ))
//     { 
//         tmp[i * NJ + j] = 0;
//         int k;
//         for (k = 0; k < _PB_NK; k++)
//         {
//             tmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
//         }
//     }
// }

__global__ void mm2_kernel1_tiled_pipelined(int ni,int nj,int nk, DATA_TYPE alpha, const DATA_TYPE* __restrict__ A, const DATA_TYPE* __restrict__ B, DATA_TYPE* __restrict__ tmp)
{
    __shared__ DATA_TYPE shA1[TILE][TILE];
    __shared__ DATA_TYPE shA2[TILE][TILE];
    __shared__ DATA_TYPE shB1[TILE][TILE];
    __shared__ DATA_TYPE shB2[TILE][TILE];

    // int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    DATA_TYPE acc1 = 0;
    DATA_TYPE acc2 = 0;

    DATA_TYPE * shAptr = &shA1[0][0];
    DATA_TYPE * shBptr = &shB1[0][0];

    if (threadIdx.y < TILE/2){
        int isub1= 2*threadIdx.y;
        int isub2= isub1+1;
        int row1 = blockIdx.y*blockDim.y + isub1;
        int row2 = blockIdx.y*blockDim.y + isub2;

        if (row1 < ni && threadIdx.x < nk){
            shAptr[isub1*TILE+ threadIdx.x] = A[row1 * nk + (threadIdx.x)];
            shAptr[isub2*TILE+ threadIdx.x] = A[row2 * nk + (threadIdx.x)];
        } else {
            shAptr[isub1*TILE+ threadIdx.x] = 0;
            shAptr[isub2*TILE+ threadIdx.x] = 0;
        }
        if (col < nj && isub1 < nk){
            shBptr[isub1*TILE+ threadIdx.x] = B[(isub1) * nj + col];
            shBptr[isub2*TILE+ threadIdx.x] = B[(isub2) * nj + col];
        } else {
            shBptr[isub1*TILE+ threadIdx.x] = 0;
            shBptr[isub2*TILE+ threadIdx.x] = 0;
        }
    }

    DATA_TYPE* shAptr2 = shAptr;
    DATA_TYPE* shBptr2 = shBptr;
    shAptr = &shA2[0][0];
    shBptr = &shB2[0][0];

    __syncthreads();

    for (int t = 0; t < ((nk + TILE- 1) / TILE)-1; t++){
        if (threadIdx.y < TILE/2){
            // load t+1's data, for 2*threadIdx.y and +1
            int isub1= 2*threadIdx.y;
            int isub2= isub1+1;
            int row1 = blockIdx.y*blockDim.y + isub1;
            int row2 = blockIdx.y*blockDim.y + isub2;
            if (row1 < ni && ((t+1)*TILE + threadIdx.x) < nk){
                shAptr[isub1*TILE + threadIdx.x] = A[row1 * nk + ((t+1)*TILE + threadIdx.x)];
                shAptr[isub2*TILE + threadIdx.x] = A[row2 * nk + ((t+1)*TILE + threadIdx.x)];
            } else {
                shAptr[isub1*TILE + threadIdx.x] = 0.0;
                shAptr[isub2*TILE + threadIdx.x] = 0.0;
            }
            if (col < nj && ((t+1)*TILE + isub1) < nk){
                shBptr[isub1*TILE + threadIdx.x] = B[((t+1)*TILE + isub1) * nj + col];
                shBptr[isub2*TILE + threadIdx.x] = B[((t+1)*TILE + isub2) * nj + col];
            } else {
                shBptr[isub1*TILE + threadIdx.x] = 0.0;
                shBptr[isub2*TILE + threadIdx.x] = 0.0;
            }
        } else {
            for (int k = 0; k < TILE; k++)
            {
                acc1 += shAptr2[2*(threadIdx.y-TILE/2)*TILE + k] * shBptr2[k*TILE + threadIdx.x];
                acc2 += shAptr2[(2*(threadIdx.y-TILE/2)+1)*TILE + k] * shBptr2[k*TILE + threadIdx.x];
            }
        }

        DATA_TYPE* shAtmp = shAptr;
        DATA_TYPE* shBtmp = shBptr;
        shAptr = shAptr2;
        shBptr = shBptr2;
        shAptr2 = shAtmp;
        shBptr2 = shBtmp;

        __syncthreads();

    }
    if (threadIdx.y >= TILE/2){
        // perform ops on t=((nk + TILE - 1) / TILE)-1, 2*(tid.y-16) and +1
        for (int k = 0; k < TILE; k++)
        {
            acc1 += shAptr2[2*(threadIdx.y-TILE/2)*TILE + k] * shBptr2[k*TILE + threadIdx.x];
            acc2 += shAptr2[(2*(threadIdx.y-TILE/2)+1)*TILE + k] * shBptr2[k*TILE + threadIdx.x];
        }
        int isub1= 2*(threadIdx.y-TILE/2);
        int isub2= isub1+1;
        int row1 = blockIdx.y*blockDim.y + isub1;
        int row2 = blockIdx.y*blockDim.y + isub2;
        if ((row1 < _PB_NI) && (col < _PB_NJ))
        {	
            tmp[row1*NJ + col] = acc1*alpha;
            tmp[row2*NJ + col] = acc2*alpha;
        }
    }
}


// __global__ void mm2_kernel2(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *C, DATA_TYPE *D)
// {
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     int i = blockIdx.y * blockDim.y + threadIdx.y;

//     if ((i < _PB_NI) && (j < _PB_NL))
//     { 
//         D[i * NL + j] *= beta;
//         int k;
//         for (k = 0; k < _PB_NJ; k++)
//         {
//             D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
//         }
//     }
// }

__global__ void mm2_kernel2_tiled(int ni,int nj,int nk,int nl, DATA_TYPE beta, const DATA_TYPE* __restrict__ tmp, const DATA_TYPE* __restrict__ C, DATA_TYPE* __restrict__ D)
{
    __shared__ DATA_TYPE shT1[TILE][TILE];
    __shared__ DATA_TYPE shT2[TILE][TILE];
    __shared__ DATA_TYPE shC1[TILE][TILE];
    __shared__ DATA_TYPE shC2[TILE][TILE];

    // int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    DATA_TYPE acc1 = 0;
    DATA_TYPE acc2 = 0;

    DATA_TYPE * shTptr = &shT1[0][0];
    DATA_TYPE * shCptr = &shC1[0][0];

    if (threadIdx.y < TILE/2){
        int isub1= 2*threadIdx.y;
        int isub2= isub1+1;
        int row1 = blockIdx.y*blockDim.y + isub1;
        int row2 = blockIdx.y*blockDim.y + isub2;

        if (row1 < ni && threadIdx.x < nj){
            shTptr[isub1*TILE+ threadIdx.x] = tmp[row1 * nj + (threadIdx.x)];
            shTptr[isub2*TILE+ threadIdx.x] = tmp[row2 * nj + (threadIdx.x)];
        } else {
            shTptr[isub1*TILE+ threadIdx.x] = 0;
            shTptr[isub2*TILE+ threadIdx.x] = 0;
        }
        if (col < nl && isub1 < nj){
            shCptr[isub1*TILE+ threadIdx.x] = C[(isub1) * nl + col];
            shCptr[isub2*TILE+ threadIdx.x] = C[(isub2) * nl + col];
        } else {
            shCptr[isub1*TILE+ threadIdx.x] = 0;
            shCptr[isub2*TILE+ threadIdx.x] = 0;
        }
    }

    DATA_TYPE* shTptr2 = shTptr;
    DATA_TYPE* shCptr2 = shCptr;
    shTptr = &shT2[0][0];
    shCptr = &shC2[0][0];

    __syncthreads();


    for (int t = 0; t < ((nk + TILE- 1) / TILE)-1; t++){
        if (threadIdx.y < TILE/2){
            // load t+1's data, for 2*threadIdx.y and +1
            int isub1= 2*threadIdx.y;
            int isub2= isub1+1;
            int row1 = blockIdx.y*blockDim.y + isub1;
            int row2 = blockIdx.y*blockDim.y + isub2;
            if (row1 < ni && ((t+1)*TILE + threadIdx.x) < nj){
                shTptr[isub1*TILE + threadIdx.x] = tmp[row1 * nj + ((t+1)*TILE + threadIdx.x)];
                shTptr[isub2*TILE + threadIdx.x] = tmp[row2 * nj + ((t+1)*TILE + threadIdx.x)];
            } else {
                shTptr[isub1*TILE + threadIdx.x] = 0.0;
                shTptr[isub2*TILE + threadIdx.x] = 0.0;
            }
            if (col < nl && ((t+1)*TILE + isub1) < nj){
                shCptr[isub1*TILE + threadIdx.x] = C[((t+1)*TILE + isub1) * nl + col];
                shCptr[isub2*TILE + threadIdx.x] = C[((t+1)*TILE + isub2) * nl + col];
            } else {
                shCptr[isub1*TILE + threadIdx.x] = 0.0;
                shCptr[isub2*TILE + threadIdx.x] = 0.0;
            }
        } else {
            for (int k = 0; k < TILE; k++)
            {
                acc1 += shTptr2[2*(threadIdx.y-TILE/2)*TILE + k] * shCptr2[k*TILE + threadIdx.x];
                acc2 += shTptr2[(2*(threadIdx.y-TILE/2)+1)*TILE + k] * shCptr2[k*TILE + threadIdx.x];
            }
        }

        DATA_TYPE* shTtmp = shTptr;
        DATA_TYPE* shCtmp = shCptr;
        shTptr = shTptr2;
        shCptr = shCptr2;
        shTptr2 = shTtmp;
        shCptr2 = shCtmp;

        __syncthreads();
    }
    if (threadIdx.y >= TILE/2){
        // perform ops on t=((nk + TILE - 1) / TILE)-1, 2*(tid.y-16) and +1
        for (int k = 0; k < TILE; k++)
        {
            acc1 += shTptr2[2*(threadIdx.y-TILE/2)*TILE + k] * shCptr2[k*TILE + threadIdx.x];
            acc2 += shTptr2[(2*(threadIdx.y-TILE/2)+1)*TILE + k] * shCptr2[k*TILE + threadIdx.x];
        }
        int isub1= 2*(threadIdx.y-TILE/2);
        int isub2= isub1+1;
        int row1 = blockIdx.y*blockDim.y + isub1;
        int row2 = blockIdx.y*blockDim.y + isub2;
        if ((row1 < _PB_NI) && (col < _PB_NJ))
        {	
            D[row1*NL + col] = D[row1*NL + col]*beta + acc1;
            D[row2*NL + col] = D[row2*NL + col]*beta + acc2;
        }
    }
}


void mm2_cpu(int ni, int nj, int nk, int nl,
        DATA_TYPE alpha,
        DATA_TYPE beta,
        DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
        DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
        DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
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
static
void print_array(int ni, int nl,
        DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
int i, j;

for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
    if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
fprintf (stderr, "\n");
}


void mm2Cuda(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), 
    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), 
    DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu,NI,NL,ni,nl))
{
    DATA_TYPE *tmp_gpu;
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;
    DATA_TYPE *D_gpu;

    cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NL * NJ);
    cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NI * NL);
    
    cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NL * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);	
        
    // dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    // dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
    // dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

    // /* Start timer. */
    // polybench_start_instruments;

    // mm2_kernel1<<<grid1,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, A_gpu, B_gpu);
    // cudaThreadSynchronize();
    // mm2_kernel2<<<grid2,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu, D_gpu);
    // cudaThreadSynchronize();


    dim3 block(TILE, TILE);
    dim3 grid1((NJ + TILE - 1) / TILE, (NI + TILE - 1) / TILE);
    dim3 grid2((NL + TILE - 1) / TILE, (NI + TILE - 1) / TILE);

    polybench_start_instruments;
    mm2_kernel1_tiled_pipelined<<<grid1, block>>>(ni,nj,nk,alpha,A_gpu,B_gpu,tmp_gpu);
    cudaDeviceSynchronize();
    mm2_kernel2_tiled<<<grid2, block>>>(ni,nj,nk,nl,beta,tmp_gpu,C_gpu,D_gpu);
    cudaDeviceSynchronize();
    polybench_stop_instruments;

    printf("GPU Time in seconds:\n");
    polybench_print_instruments;

    cudaMemcpy(D_outputFromGpu, D_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

    cudaFree(tmp_gpu);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(D_gpu);
}


int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NL,NJ,nl,nj);
    POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
    POLYBENCH_2D_ARRAY_DECL(D_outputFromGpu,DATA_TYPE,NI,NL,ni,nl);
    
    /* Initialize array(s). */
    init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
    GPU_argv_init();

    mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), 
        POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

    #ifdef RUN_ON_CPU

        /* Start timer. */
        polybench_start_instruments;

        mm2_cpu(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

        printf("CPU Time in seconds:\n");
        polybench_stop_instruments;
        polybench_print_instruments;

        compareResults(ni, nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

    #else //print output to stderr so no dead code elimination

        print_array(ni, nl, POLYBENCH_ARRAY(D_outputFromGpu));

    #endif //RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(D_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"

