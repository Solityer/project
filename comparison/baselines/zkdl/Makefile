# Specify the CUDA and libtorch installation paths
CUDA_PATH ?= /usr/local/cuda
LIBTORCH_PATH ?= $(VIRTUAL_ENV)/lib/python3.10/site-packages/torch

# Compilers
NVCC := $(CUDA_PATH)/bin/nvcc

# Include and library paths
INCLUDES := -I$(CUDA_PATH)/include # -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
#LIBS     := -L$(CUDA_PATH)/lib64 -lcudart -L$(LIBTORCH_PATH)/lib -ltorch -lc10
LIBS := -L$(CUDA_PATH)/lib64 -lcudart # -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda


# NVCC compiler flags
NVCC_FLAGS ?= -arch=sm_80 -std=c++17 -O3 -Xcompiler=-Wall -Xcompiler=-Wextra

# Formal targets used by the adapter.
TARGETS := gat_zkdl formal_opening_selftest formal_sumcheck_selftest formal_relation_selftest
COMMON_HEADERS := bls12-381.cuh fr-tensor.cuh g1-tensor.cuh commitment.cuh proof.cuh formal_relations.cuh zkfc.cuh zkmatmul.cuh zkrelu.cuh

all: $(TARGETS)

gat_zkdl: gat_zkdl.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

formal_opening_selftest: formal_opening_selftest.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

formal_sumcheck_selftest: formal_sumcheck_selftest.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

formal_relation_selftest: formal_relation_selftest.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@
	
clean:
	rm -f $(TARGETS)
