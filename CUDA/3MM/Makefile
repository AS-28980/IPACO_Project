EXECUTABLE := 3mm.exe
CUFILES := 3mm.cu

include ../common.mk

extra:
	nvcc -O3 3mm_tiled.cu -o 3mm_tiled.exe
	nvcc -O3 3mm_tiled_pipelined.cu -o 3mm_tiled_pipelined.exe
