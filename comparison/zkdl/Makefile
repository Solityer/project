CUDA_PATH ?= /usr/local/cuda
LIBTORCH_PATH ?= $(VIRTUAL_ENV)/lib/python3.10/site-packages/torch

NVCC := $(CUDA_PATH)/bin/nvcc

INCLUDES := -I$(CUDA_PATH)/include

LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcrypto

CUDA_ARCH ?= sm_121
NVCC_FLAGS := -arch=$(CUDA_ARCH)

SRCS := $(wildcard *.cu)
TARGETS := $(SRCS:.cu=)
GAT_TARGET := gat_runner
GAT_SRC := gat_adapter/gat_runner.cu gat_adapter/gat_native_bridge.cu gat_adapter/gat_native_core.cu

.PHONY: all baseline-local-build test-short segmented-fixture-check clean

all: $(TARGETS) $(GAT_TARGET)

baseline-local-build: $(GAT_TARGET)

test-short: tests/core_smoke

segmented-fixture-check: tests/segmented_lookup_fixture
	 timeout 180s ./tests/segmented_lookup_fixture

tests/segmented_lookup_fixture: tests/segmented_lookup_fixture.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp
	 timeout 180s $(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) $(LIBS) -I. -Igat_adapter $< -o $@

tests/core_smoke: tests/core_smoke.cu proof.cuh commitment.cuh fr-tensor.cuh g1-tensor.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

%: %.cu
	$(NVCC)  $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

$(GAT_TARGET): $(GAT_SRC) gat_adapter/gat_native_resources.cu gat_adapter/gat_adapter.hpp gat_adapter/gat_native_abi.hpp gat_adapter/gat_native_bridge.hpp gat_adapter/gat_native_core.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_sparse_route.cuh g1-wire.cuh zkrelation-proof.cuh commitment.cuh proof.cuh fr-tensor.cuh g1-tensor.cuh zkmatmul.cuh zkprod.cuh
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) $(LIBS) -I. -Igat_adapter $(GAT_SRC) -o $@

clean:
	rm -f $(TARGETS) $(GAT_TARGET) tests/core_smoke
