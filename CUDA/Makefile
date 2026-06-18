NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_75
.PHONY: all clean
all: scoring_cuda
scoring_cuda: scoring_kernel.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
clean:
	rm -f scoring_cuda scoring_cuda_fp64 *.o
