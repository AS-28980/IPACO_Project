EXECUTABLE := 2mm.exe
CUFILES := 2mm.cu

include ../common.mk

extra:
	nvcc -O3 2mm_tiled.cu -o 2mm_tiled.exe
	nvcc -O3 2mm_tiled_pipelined.cu -o 2mm_tiled_pipelined.exe
