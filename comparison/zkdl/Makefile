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
GAT_RESOURCE_TARGET := gat_resource_setup
BLST_LIBRARY := third_party/blst/libblst.a
BLST_SOURCES := $(shell find third_party/blst/src third_party/blst/build third_party/blst/bindings -type f)

.PHONY: all baseline-local-build test-short segmented-fixture-check output-stream-failure-check formal-closure-check clean

all: $(TARGETS) $(GAT_TARGET) $(GAT_RESOURCE_TARGET)

baseline-local-build: $(GAT_TARGET) $(GAT_RESOURCE_TARGET)

$(BLST_LIBRARY): $(BLST_SOURCES) third_party/blst/build.sh
	cd third_party/blst && ./build.sh

$(GAT_RESOURCE_TARGET): gat_adapter/gat_resource_setup.cu $(BLST_LIBRARY)
	$(CXX) -O2 -std=c++17 -x c++ -I. $< -x none $(BLST_LIBRARY) -lcrypto -o $@

test-short: tests/core_smoke

segmented-fixture-check: tests/segmented_lookup_fixture
	 timeout 180s ./tests/segmented_lookup_fixture

output-stream-failure-check: tests/output_stream_failure_check
	./tests/output_stream_failure_check

formal-closure-check:
	$(MAKE) -j2 baseline-local-build
	$(MAKE) output-stream-failure-check
	python3 tests/test_tooling.py ToolingTests.test_schema_fixtures ToolingTests.test_formal_five_gate ToolingTests.test_collection_binding

tests/output_stream_failure_check: tests/output_stream_failure_check.cpp gat_adapter/output_stream.hpp
	$(CXX) -O2 -std=c++17 -I. $< -o $@

tests/segmented_lookup_fixture: tests/segmented_lookup_fixture.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp
	 timeout 180s $(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) $(LIBS) -I. -Igat_adapter $< -o $@

tests/core_smoke: tests/core_smoke.cu proof.cuh commitment.cuh fr-tensor.cuh g1-tensor.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

tests/pointwise_algebra_check: tests/pointwise_algebra_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) $(LIBS) -I. -Igat_adapter $< -o $@

%: %.cu
	$(NVCC)  $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

$(GAT_TARGET): $(GAT_SRC) gat_adapter/gat_native_resources.cu gat_adapter/gat_adapter.hpp gat_adapter/gat_native_abi.hpp gat_adapter/gat_family_storage.hpp gat_adapter/gat_native_bridge.hpp gat_adapter/gat_native_core.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_sparse_route.cuh gat_adapter/output_stream.hpp g1-wire.cuh zkrelation-proof.cuh commitment.cuh proof.cuh fr-tensor.cuh g1-tensor.cuh zkmatmul.cuh zkprod.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $(GAT_SRC) $(BLST_LIBRARY) -o $@

clean:
	rm -f $(TARGETS) $(GAT_TARGET) $(GAT_RESOURCE_TARGET) tests/core_smoke
