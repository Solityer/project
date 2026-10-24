#!/usr/bin/env python3
"""Exercise the production proof-envelope key/value reader and writer."""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent

CPP = r'''
#define main zkdl_runner_entry
#include "gat_adapter/gat_runner.cu"
#undef main

#include <cassert>
#include <fstream>
#include <iterator>

using namespace zkdl_gat;

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::filesystem::path directory(argv[1]);
    const auto envelope_path = directory / "proof.meta";

    ProofEnvelope envelope;
    envelope.statement_id = std::string(64, 'a');
    envelope.backend = "cpu";
    envelope.status = "PROVED";
    envelope.static_table_descriptor_digest = std::string(64, 'b');
    envelope.scale_chain_digest = std::string(64, 'c');
    envelope.integer_bound_ledger_id = "ledger_fixture";
    envelope.integer_bound_ledger_digest = std::string(64, 'd');
    envelope.integer_bound_ledger_descriptor_digest = std::string(64, 'e');
    envelope.proof_size_bytes = 123;
    envelope.output_bound = true;

    write_envelope(envelope_path, envelope);
    std::ifstream input(envelope_path);
    const std::string written((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    assert(written.find("error=") == std::string::npos);
    assert(written.find("gpu_device_name=") == std::string::npos);
    assert(written.find("gpu_device_uuid=") == std::string::npos);
    const auto roundtrip = read_envelope(envelope_path);
    assert(roundtrip.statement_id == envelope.statement_id);
    assert(roundtrip.status == "PROVED");
    assert(roundtrip.error.empty());
    assert(roundtrip.gpu_device_name.empty());
    assert(roundtrip.gpu_device_uuid.empty());

    {
        std::ofstream legacy(envelope_path, std::ios::app);
        assert(legacy);
        legacy << "error=\ngpu_device_name=\ngpu_device_uuid=\n";
    }
    const auto legacy = read_envelope(envelope_path);
    assert(legacy.error.empty());
    assert(legacy.gpu_device_name.empty());
    assert(legacy.gpu_device_uuid.empty());
    bool generic_reader_rejected_empty_error = false;
    try {
        (void)read_kv(envelope_path);
    } catch (const std::exception &exception) {
        generic_reader_rejected_empty_error =
            std::string(exception.what()).find("empty key/value") != std::string::npos;
    }
    assert(generic_reader_rejected_empty_error);

    const auto config_path = directory / "config.cfg";
    {
        std::ofstream config(config_path);
        assert(config);
        config << "scale=\n";
    }
    bool generic_config_reader_rejected_empty_value = false;
    try {
        (void)read_kv(config_path);
    } catch (const std::exception &exception) {
        generic_config_reader_rejected_empty_value =
            std::string(exception.what()).find("empty key/value") != std::string::npos;
    }
    assert(generic_config_reader_rejected_empty_value);

    envelope.error = "proof diagnostic";
    envelope.gpu_device_name = "fixture-device";
    envelope.gpu_device_uuid = "fixture-uuid";
    write_envelope(envelope_path, envelope);
    const auto with_error = read_envelope(envelope_path);
    assert(with_error.error == "proof diagnostic");
    assert(with_error.gpu_device_name == "fixture-device");
    assert(with_error.gpu_device_uuid == "fixture-uuid");

    {
        std::ifstream source(envelope_path);
        std::string contents((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
        const std::string backend = "backend=cpu\n";
        const auto position = contents.find(backend);
        assert(position != std::string::npos);
        contents.replace(position, backend.size(), "backend=\n");
        std::ofstream malformed(envelope_path, std::ios::trunc);
        assert(malformed);
        malformed << contents;
    }
    bool empty_required_field_rejected = false;
    try {
        (void)read_envelope(envelope_path);
    } catch (const std::exception &exception) {
        empty_required_field_rejected =
            std::string(exception.what()).find("empty key/value") != std::string::npos;
    }
    assert(empty_required_field_rejected);

    std::cout << "PROOF_ENVELOPE_CHECK_OK ordinary-cpu-roundtrip legacy-empty-optional-fields nonempty-diagnostics required-field-strictness\n";
}
'''


with tempfile.TemporaryDirectory(prefix=".proof-envelope-check-", dir=ROOT / "tests") as raw:
    directory = Path(raw)
    translation = directory / "proof_envelope_check.cu"
    binary = directory / "proof_envelope_check"
    translation.write_text(CPP, encoding="utf-8")
    subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc",
            "-std=c++17",
            "-O0",
            "-arch=sm_121",
            "-Xcompiler=-ffunction-sections,-fdata-sections",
            "-Xlinker=--gc-sections",
            "-I",
            str(ROOT),
            "-x",
            "cu",
            str(translation),
            "-lcrypto",
            "-o",
            str(binary),
        ],
        check=True,
        timeout=90,
    )
    subprocess.run([str(binary), str(directory)], check=True, timeout=10)
