EXECUTABLE := gemm.exe
CUFILES := gemm.cu

include ../common.mk

extra:
	nvcc -O3 gemm_tiled.cu -o gemm_tiled.exe
	nvcc -O3 gemm_tiled_pipelined.cu -o gemm_tiled_pipelined.exe
