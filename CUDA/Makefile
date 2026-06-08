NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17
.PHONY: all clean
all: scoring_cuda
scoring_cuda: scoring_kernel.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
clean:
	rm -f scoring_cuda *.o
