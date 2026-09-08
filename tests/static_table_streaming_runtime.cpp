#include <chrono>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/verifier_context.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::protocol::ProtocolContext;
    using gatzk::protocol::StaticTables;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    std::string sha256_hex(std::string_view bytes)
    {
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256(
            reinterpret_cast<const unsigned char *>(bytes.data()),
            bytes.size(),
            digest.data());
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto byte : digest)
            out << std::setw(2) << static_cast<unsigned int>(byte);
        return out.str();
    }

    // Independent test-only legacy encoder.  This is intentionally kept as
    // the pre-streaming ostringstream form so tests detect any byte drift in
    // the production encoder rather than sharing its implementation.
    std::string legacy_static_table_serialization(const StaticTables &tables)
    {
        std::ostringstream out;
        out << "static_tables|range=" << tables.range.size();
        for (const auto &value : tables.range)
            out << '|' << value.to_string();
        out << "|lrelu=" << tables.lrelu.size();
        for (const auto &[x, y] : tables.lrelu)
            out << '|' << x.to_string() << ',' << y.to_string();
        out << "|elu=" << tables.elu.size();
        for (const auto &[x, y] : tables.elu)
            out << '|' << x.to_string() << ',' << y.to_string();
        out << "|exp=" << tables.exp.size();
        for (const auto &[x, y] : tables.exp)
            out << '|' << x.to_string() << ',' << y.to_string();
        return out.str();
    }

    void append_u64(std::string *out, std::uint64_t value)
    {
        for (std::size_t index = 0; index < sizeof(value); ++index)
            out->push_back(static_cast<char>((value >> (8U * index)) & 0xffU));
    }

    void append_string(std::string *out, std::string_view value)
    {
        append_u64(out, value.size());
        out->append(value.data(), value.size());
    }

    void append_shape(std::string *out, const gatzk::protocol::CanonicalStaticTableShape &shape)
    {
        append_string(out, "canonical_static_table_shape");
        append_u64(out, shape.range_table_size);
        append_u64(out, static_cast<std::uint64_t>(shape.lrelu_min_raw));
        append_u64(out, static_cast<std::uint64_t>(shape.lrelu_max_raw));
        append_u64(out, shape.lrelu_table_size);
        append_u64(out, shape.elu_bound);
        append_u64(out, shape.elu_table_size);
        append_u64(out, shape.lrelu_output_bound);
        append_u64(out, shape.exp_input_bound);
        append_u64(out, shape.exp_table_size);
        append_u64(out, shape.table_edge_domain_size);
        append_u64(out, shape.large_whole_graph ? 1U : 0U);
    }

    ProtocolContext fixture_context(const StaticTables &tables)
    {
        ProtocolContext context;
        context.config.dataset = "cora";
        context.config.batching_rule = "whole_graph_single";
        context.config.static_table_id = "streaming-fixture";
        context.config.range_bits = 3;
        context.config.quantization.s_delta = 7;
        context.config.quantization.s_exp = 11;
        context.dataset.num_nodes = 4;
        context.dataset.num_features = 2;
        context.local.num_nodes = 4;
        context.local.edges.resize(3);
        context.tables = tables;
        auto compiler_artifact =
            std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>();
        compiler_artifact->integer_bound_ledger = {
            {.id = "fixture.bound", .stage = "fixture", .source = "test", .absolute_bound = 17, .limb_count = 2},
            {.id = "fixture.bound.2", .stage = "fixture", .source = "test/second", .absolute_bound = 99, .limb_count = 3},
        };
        context.compiler_artifact = std::move(compiler_artifact);
        return context;
    }

    std::string legacy_static_table_spec_serialization(
        const ProtocolContext &context)
    {
        std::string bytes;
        append_string(&bytes, "static_table_spec");
        append_string(&bytes, "zkgat_static_table_generator");
        append_string(&bytes, context.config.static_table_id);
        append_u64(&bytes, context.config.range_bits);
        append_string(
            &bytes,
            context.config.quantization.canonical_serialization());
        append_string(
            &bytes,
            "range=ascending-u64;LReLU=canonical-signed-round;"
            "ELU=exp-minus-one-signed-round;"
            "ExpMap=exp-negative-delta-positive-saturate-one;"
            "pairs=ordered-functional;field=canonical");
        append_shape(
            &bytes,
            gatzk::protocol::plan_canonical_static_table_shape(context));
        append_string(
            &bytes,
            legacy_static_table_serialization(context.tables));
        return bytes;
    }

    std::string legacy_range_scale_certificate_serialization(
        const ProtocolContext &context)
    {
        std::string bytes;
        append_string(&bytes, "range_scale_certificate");
        append_u64(&bytes, context.config.range_bits);
        append_string(
            &bytes,
            context.config.quantization.canonical_serialization());
        append_shape(
            &bytes,
            gatzk::protocol::plan_canonical_static_table_shape(context));
        append_string(
            &bytes,
            legacy_static_table_serialization(context.tables));
        append_string(
            &bytes,
            gatzk::protocol::integer_bound_ledger_canonical_bytes(
                context.compiler_artifact->integer_bound_ledger));
        return bytes;
    }

    StaticTables boundary_fixture()
    {
        StaticTables tables;
        tables.range = {
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::from_signed(-1),
            FieldElement::from_signed(-2)};
        tables.lrelu = {
            {FieldElement::from_signed(-2), FieldElement::from_signed(-1)},
            {FieldElement::zero(), FieldElement::zero()},
            {FieldElement(2), FieldElement(3)}};
        tables.elu = {
            {FieldElement::from_signed(-3), FieldElement::from_signed(-2)},
            {FieldElement::one(), FieldElement(4)}};
        tables.exp = {
            {FieldElement::zero(), FieldElement(11)},
            {FieldElement::from_signed(-1), FieldElement::one()}};
        return tables;
    }

    void check_static_table_fixture(const StaticTables &tables, std::string_view label)
    {
        const auto legacy = legacy_static_table_serialization(tables);
        const auto expected_digest = sha256_hex(legacy);
        const auto encoded_size =
            gatzk::protocol::detail::canonical_static_table_serialization_size(tables);
        expect(
            encoded_size <= std::numeric_limits<std::uint64_t>::max(),
            std::string(label) + " serialization length exceeds uint64 contract");
        expect(
            encoded_size == legacy.size(),
            std::string(label) + " serialization length drifted");
        expect(
            gatzk::protocol::detail::canonical_static_table_digest(tables)
                == expected_digest,
            std::string(label) + " streaming digest drifted");
    }

    StaticTables moderate_fixture()
    {
        constexpr std::size_t row_count = 1U << 18;
        StaticTables tables;
        tables.range.reserve(row_count);
        tables.lrelu.reserve(row_count / 4U);
        tables.elu.reserve(row_count / 4U);
        tables.exp.reserve(row_count / 4U);
        for (std::size_t index = 0; index < row_count; ++index)
            tables.range.emplace_back(static_cast<std::uint64_t>(index));
        for (std::size_t index = 0; index < row_count / 4U; ++index)
        {
            const auto x = FieldElement(static_cast<std::uint64_t>(index));
            const auto y = FieldElement(static_cast<std::uint64_t>(index + 1U));
            tables.lrelu.emplace_back(x, y);
            tables.elu.emplace_back(y, x);
            tables.exp.emplace_back(x, FieldElement::one());
        }
        return tables;
    }

    int benchmark(const std::string &mode)
    {
        const auto tables = moderate_fixture();
        const auto start = std::chrono::steady_clock::now();
        std::string digest;
        if (mode == "legacy")
        {
            for (std::size_t repeat = 0; repeat < 2; ++repeat)
                digest = sha256_hex(legacy_static_table_serialization(tables));
        }
        else if (mode == "streaming")
        {
            for (std::size_t repeat = 0; repeat < 2; ++repeat)
                digest = gatzk::protocol::detail::canonical_static_table_digest(tables);
        }
        else
        {
            throw std::runtime_error("usage: --bench legacy|streaming");
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start);
        std::cout << "static_table_streaming_bench mode=" << mode
                  << " rows=" << tables.range.size()
                  << " digest=" << digest
                  << " wall_ms=" << std::fixed << std::setprecision(3)
                  << elapsed.count() << '\n';
        return 0;
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc == 3 && std::string(argv[1]) == "--bench")
            return benchmark(argv[2]);
        expect(argc == 1, "unexpected arguments");

        check_static_table_fixture(StaticTables{}, "empty");
        const auto boundary = boundary_fixture();
        check_static_table_fixture(boundary, "boundary");

        auto reordered = boundary;
        std::swap(reordered.lrelu.front(), reordered.lrelu.back());
        check_static_table_fixture(reordered, "reordered");
        expect(
            gatzk::protocol::detail::canonical_static_table_digest(boundary)
                != gatzk::protocol::detail::canonical_static_table_digest(reordered),
            "table row order was not bound by the streaming digest");

        const auto context = fixture_context(boundary);
        const auto expected_spec = sha256_hex(
            legacy_static_table_spec_serialization(context));
        const auto expected_certificate = sha256_hex(
            legacy_range_scale_certificate_serialization(context));
        expect(
            gatzk::protocol::detail::canonical_static_table_spec_hash(context)
                == expected_spec,
            "streaming static-table spec hash drifted");
        expect(
            gatzk::protocol::detail::canonical_range_and_scale_certificate(context)
                == expected_certificate,
            "streaming range/scale certificate drifted");

        std::cout << "static_table_streaming_runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "static_table_streaming_runtime: FAIL: " << error.what() << '\n';
        return 1;
    }
}
