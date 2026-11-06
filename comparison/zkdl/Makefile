CUDA_PATH ?= /usr/local/cuda
LIBTORCH_PATH ?= $(VIRTUAL_ENV)/lib/python3.10/site-packages/torch

NVCC := $(CUDA_PATH)/bin/nvcc

INCLUDES := -I$(CUDA_PATH)/include

LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcrypto

CUDA_ARCH ?= sm_121
NVCC_FLAGS := -arch=$(CUDA_ARCH)
NVCC_HOST_FLAGS := -Xcompiler=-O2

SRCS := $(wildcard *.cu)
TARGETS := $(SRCS:.cu=)
GAT_TARGET := gat_runner
GAT_SRC := gat_adapter/gat_runner.cu gat_adapter/gat_native_bridge.cu gat_adapter/gat_native_core.cu
GAT_OBJ := $(GAT_SRC:.cu=.o)
GAT_RESOURCE_TARGET := gat_resource_setup
BLST_LIBRARY := third_party/blst/libblst.a
BLST_SOURCES := $(shell find third_party/blst/src third_party/blst/build third_party/blst/bindings -type f)

.PHONY: all baseline-local-build test-short checkpoint-manifest-check ledger-contract-check lookup-table-index-bits-check signed-round-bounds-check static-table-bounds-check segmented-fixture-check segmented-shared-srs-check sorted-lookup-descriptor-check sorted-lookup-prefix-cache-check sorted-lookup-public-key-order-check public-field-vector-decode-check pointwise-eqmask-table-check cpu-g1-blst-check cpu-g1-msm-nbits-check output-stream-failure-check formal-closure-check hyrax-gpu-msm-fixture-check clean

all: $(TARGETS) $(GAT_TARGET) $(GAT_RESOURCE_TARGET)

baseline-local-build: $(GAT_TARGET) $(GAT_RESOURCE_TARGET)

$(BLST_LIBRARY): $(BLST_SOURCES) third_party/blst/build.sh
	cd third_party/blst && ./build.sh

$(GAT_RESOURCE_TARGET): gat_adapter/gat_resource_setup.cu gat_adapter/integer_bound_ledger.hpp gat_adapter/checkpoint_manifest_json.hpp gat_adapter/gat_adapter.hpp $(BLST_LIBRARY)
	$(CXX) -O2 -std=c++17 -x c++ -I. $< -x none $(BLST_LIBRARY) -lcrypto -o $@

test-short: tests/core_smoke

segmented-fixture-check: tests/segmented_lookup_fixture
	 timeout 180s ./tests/segmented_lookup_fixture

segmented-shared-srs-check: tests/segmented_lookup_fixture
	@test -n "$(SRS_CONFIG)" || { echo 'set SRS_CONFIG to an official authenticated Cora config' >&2; exit 2; }
	timeout 1800s ./tests/segmented_lookup_fixture --authenticated-srs-config "$(SRS_CONFIG)"

sorted-lookup-descriptor-check: tests/authenticated_sorted_lookup_descriptor_check
	timeout 60s ./tests/authenticated_sorted_lookup_descriptor_check

sorted-lookup-prefix-cache-check: tests/sorted_lookup_prefix_cache_check
	timeout 60s ./tests/sorted_lookup_prefix_cache_check

sorted-lookup-public-key-order-check: tests/sorted_lookup_public_key_order_check
	timeout 60s ./tests/sorted_lookup_public_key_order_check

public-field-vector-decode-check: tests/public_field_vector_decode_check
	timeout 60s ./tests/public_field_vector_decode_check

pointwise-eqmask-table-check: tests/pointwise_eqmask_table_check
	timeout 60s ./tests/pointwise_eqmask_table_check

cpu-g1-blst-check: tests/cpu_g1_blst_check
	./tests/cpu_g1_blst_check

cpu-g1-msm-nbits-check: tests/cpu_g1_msm_nbits_check
	timeout 60s ./tests/cpu_g1_msm_nbits_check

hyrax-gpu-msm-fixture-check: tests/hyrax_gpu_msm_fixture
	timeout 300s ./tests/hyrax_gpu_msm_fixture

output-stream-failure-check: tests/output_stream_failure_check
	./tests/output_stream_failure_check

formal-closure-check:
	$(MAKE) -j2 baseline-local-build
	$(MAKE) checkpoint-manifest-check
	$(MAKE) output-stream-failure-check
	python3 tests/test_tooling.py ToolingTests.test_schema_fixtures ToolingTests.test_formal_five_gate ToolingTests.test_collection_binding

checkpoint-manifest-check: tests/checkpoint_manifest_json
	./tests/checkpoint_manifest_json models/checkpoints/cora_gat/manifest.json models/checkpoints/citeseer_gat/manifest.json models/checkpoints/pubmed_gat/manifest.json models/checkpoints/ppi_gat/manifest.json models/checkpoints/ogbn_arxiv_gat/manifest.json

ledger-contract-check: tests/integer_bound_ledger_contract
	@test -n "$(LEDGER_DIR)" || { echo 'set LEDGER_DIR to the canonical five-ledger directory' >&2; exit 2; }
	python3 tests/test_integer_bound_ledger.py --native ./tests/integer_bound_ledger_contract --ledger-dir "$(LEDGER_DIR)"

lookup-table-index-bits-check: tests/lookup_table_index_bits
	./tests/lookup_table_index_bits

signed-round-bounds-check: tests/signed_round_bounds_check
	./tests/signed_round_bounds_check

static-table-bounds-check: tests/static_table_bounds_check
	./tests/static_table_bounds_check

tests/checkpoint_manifest_json: tests/checkpoint_manifest_json.cpp gat_adapter/checkpoint_manifest_json.hpp
	$(CXX) -O2 -std=c++17 -I. $< -o $@

tests/integer_bound_ledger_contract: tests/integer_bound_ledger_contract.cpp gat_adapter/integer_bound_ledger.hpp gat_adapter/checkpoint_manifest_json.hpp gat_adapter/gat_adapter.hpp
	$(CXX) -O2 -std=c++17 -I. $< -lcrypto -o $@

tests/lookup_table_index_bits: tests/lookup_table_index_bits.cpp gat_adapter/lookup_table_index_bits.hpp
	$(CXX) -O2 -std=c++17 -I. $< -o $@

tests/signed_round_bounds_check: tests/signed_round_bounds_check.cpp gat_adapter/signed_round_bounds.hpp gat_adapter/gat_adapter.hpp
	$(CXX) -O2 -std=c++17 -I. $< -o $@

tests/static_table_bounds_check: tests/static_table_bounds_check.cpp gat_adapter/static_table_bounds.hpp
	$(CXX) -O2 -std=c++17 -I. $< -o $@

tests/output_stream_failure_check: tests/output_stream_failure_check.cpp gat_adapter/output_stream.hpp
	$(CXX) -O2 -std=c++17 -I. $< -o $@

tests/segmented_lookup_fixture: tests/segmented_lookup_fixture.cu tests/sorted_lookup_prefix_cache_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_native_resources.cu gat_adapter/gat_adapter.hpp $(BLST_LIBRARY)
	 timeout 600s $(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter tests/segmented_lookup_fixture.cu $(BLST_LIBRARY) -o $@

tests/authenticated_sorted_lookup_descriptor_check: tests/authenticated_sorted_lookup_descriptor_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

tests/sorted_lookup_prefix_cache_check: tests/sorted_lookup_prefix_cache_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp $(BLST_LIBRARY)
	timeout 300s $(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

tests/sorted_lookup_public_key_order_check: tests/sorted_lookup_public_key_order_check.cu gat_adapter/gat_native_core.cuh gat_adapter/gat_native_bridge.hpp zkrelation-proof.cuh proof.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

tests/public_field_vector_decode_check: tests/public_field_vector_decode_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

tests/cpu_g1_blst_check: tests/cpu_g1_blst_check.cu proof.cuh cpu-backend.cuh g1-wire.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. $< $(BLST_LIBRARY) -o $@

tests/cpu_g1_msm_nbits_check: tests/cpu_g1_msm_nbits_check.cu proof.cuh cpu-backend.cuh g1-wire.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. $< $(BLST_LIBRARY) -o $@

tests/hyrax_gpu_msm_fixture: tests/hyrax_gpu_msm_fixture.cu gat_adapter/hyrax_gpu_fold.cu gat_adapter/hyrax_gpu_fold_api.hpp cpu-backend.cuh proof.cuh commitment.cuh fr-tensor.cuh g1-tensor.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

tests/core_smoke: tests/core_smoke.cu proof.cuh commitment.cuh fr-tensor.cuh g1-tensor.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

tests/pointwise_algebra_check: tests/pointwise_algebra_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) $(LIBS) -I. -Igat_adapter $< -o $@

tests/pointwise_eqmask_table_check: tests/pointwise_eqmask_table_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_adapter.hpp $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

tests/sparse_route_hidden_smoke: tests/sparse_route_hidden_smoke.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh gat_adapter/gat_sparse_route.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

%: %.cu
	$(NVCC)  $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $< -o $@

GAT_HDRS := gat_adapter/gat_native_resources.cu gat_adapter/gat_adapter.hpp gat_adapter/gat_witness_numerics.hpp gat_adapter/gat_native_abi.hpp gat_adapter/gat_family_storage.hpp gat_adapter/gat_native_bridge.hpp gat_adapter/gat_native_core.cuh gat_adapter/gat_native_resources.hpp gat_adapter/gat_sparse_route.cuh gat_adapter/output_stream.hpp gat_adapter/checkpoint_manifest_json.hpp gat_adapter/integer_bound_ledger.hpp gat_adapter/lookup_table_index_bits.hpp gat_adapter/signed_round_bounds.hpp gat_adapter/static_table_bounds.hpp gat_adapter/hyrax_gpu_fold.cu gat_adapter/hyrax_gpu_fold_api.hpp cpu-backend.cuh bls12-381.cuh g1-wire.cuh zkrelation-proof.cuh commitment.cuh proof.cuh fr-tensor.cuh g1-tensor.cuh zkmatmul-proof.cuh zkmatmul.cuh zkprod.cuh timer.hpp

$(GAT_OBJ): %.o: %.cu $(GAT_HDRS) $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) -I. -Igat_adapter -c $< -o $@

$(GAT_TARGET): $(GAT_OBJ) $(GAT_HDRS) $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $(GAT_OBJ) $(BLST_LIBRARY) -o $@

.PHONY: sumcheck-round-oracle-check
sumcheck-round-oracle-check: tests/sumcheck_round_oracle_check
	timeout 60s ./tests/sumcheck_round_oracle_check

tests/sumcheck_round_oracle_check: tests/sumcheck_round_oracle_check.cu zkrelation-proof.cuh proof.cuh cpu-backend.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. -Igat_adapter $< $(BLST_LIBRARY) -o $@

clean:
	rm -f $(TARGETS) $(GAT_TARGET) $(GAT_RESOURCE_TARGET) $(GAT_OBJ) tests/core_smoke

.PHONY: witness-numerics-check
witness-numerics-check: tests/gat_witness_numerics_check
	timeout 60s ./tests/gat_witness_numerics_check

tests/gat_witness_numerics_check: tests/gat_witness_numerics_check.cpp gat_adapter/gat_witness_numerics.hpp gat_adapter/gat_adapter.hpp
	$(CXX) -O2 -std=c++17 -I. $< -lcrypto -o $@

.PHONY: cpu-ipa-optimized-check
cpu-ipa-optimized-check: tests/cpu_ipa_optimized_check
	timeout 120s ./tests/cpu_ipa_optimized_check

tests/cpu_ipa_optimized_check: tests/cpu_ipa_optimized_check.cu cpu-backend.cuh proof.cuh g1-wire.cuh $(BLST_LIBRARY)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_HOST_FLAGS) -std=c++17 -DZKDL_USE_BLST_G1_CODEC $(INCLUDES) $(LIBS) -I. $< $(BLST_LIBRARY) -o $@
