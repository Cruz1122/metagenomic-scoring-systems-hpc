NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_75
.PHONY: all clean check-sdk

all: check-sdk scoring_cuda

check-sdk:
	@echo "Checking Windows SDK availability..."
	@if not exist "$(WindowsSdkDir)Include\$(WindowsSDKVersion)ucrt\corecrt.h" ( \
		echo WARNING: Windows SDK 10 not found. Compilation may fail if using MSVC. \
		echo Try: compiler=g++ or install Windows SDK. \
	) 2>nul || true

scoring_cuda: scoring_kernel.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

scoring_cuda_fp64: NVCCFLAGS += -DUSE_DOUBLE
scoring_cuda_fp64: scoring_cuda

clean:
	rm -f scoring_cuda scoring_cuda_fp64 *.o
