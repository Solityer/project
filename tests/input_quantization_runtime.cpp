#include "gatzk/data/loader.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::data::GraphDataset;
    using gatzk::data::LocalGraph;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    void expect_rejected(
        const std::function<void()> &operation,
        const std::string &label)
    {
        bool rejected = false;
        try
        {
            operation();
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        expect(rejected, label + " was accepted");
    }

    GraphDataset make_dataset()
    {
        GraphDataset dataset;
        dataset.name = "cora";
        dataset.input_preprocess_id = "row_normalize_v1";
        dataset.num_nodes = 2;
        dataset.num_features = 2;
        dataset.num_classes = 1;
        dataset.features_fp = {{0.5, -0.5}, {0.0, 1.49}};
        dataset.features = {
            {FieldElement::from_signed(1), FieldElement::from_signed(-1)},
            {FieldElement::zero(), FieldElement::from_signed(1)},
        };
        return dataset;
    }

    LocalGraph make_local()
    {
        LocalGraph local;
        local.name = "cora_normalized";
        local.input_preprocess_id = "row_normalize_v1";
        local.num_nodes = 2;
        local.num_features = 2;
        local.num_classes = 1;
        local.absolute_ids = {0, 1};
        local.features_fp = {{0.5, -0.5}, {0.0, 1.49}};
        local.features = {
            {FieldElement::from_signed(1), FieldElement::from_signed(-1)},
            {FieldElement::zero(), FieldElement::from_signed(1)},
        };
        return local;
    }
}

int main()
{
    try
    {
        using gatzk::data::quantize_raw_at_scale;

        expect(quantize_raw_at_scale(0.0, 1) == 0,
               "zero must quantize to zero");
        expect(quantize_raw_at_scale(0.49, 1) == 0,
               "positive non-integer must round to nearest integer");
        expect(quantize_raw_at_scale(-0.49, 1) == 0,
               "negative non-integer must round to nearest integer");
        expect(quantize_raw_at_scale(0.5, 1) == 1,
               "positive half tie must round away from zero");
        expect(quantize_raw_at_scale(-0.5, 1) == -1,
               "negative half tie must round away from zero");
        expect(quantize_raw_at_scale(0.05, 10) == 1,
               "scaled positive half tie must round away from zero");
        expect(quantize_raw_at_scale(-0.05, 10) == -1,
               "scaled negative half tie must round away from zero");
        expect(
            quantize_raw_at_scale(
                static_cast<double>(std::numeric_limits<std::int64_t>::min()),
                1)
                == std::numeric_limits<std::int64_t>::min(),
            "int64 minimum must remain representable");

        expect_rejected(
            [] { (void)quantize_raw_at_scale(0.0, 0); },
            "zero scale");
        expect_rejected(
            [] { (void)quantize_raw_at_scale(std::numeric_limits<double>::quiet_NaN(), 1); },
            "NaN input");
        expect_rejected(
            [] { (void)quantize_raw_at_scale(std::numeric_limits<double>::infinity(), 1); },
            "+Inf input");
        expect_rejected(
            [] { (void)quantize_raw_at_scale(-std::numeric_limits<double>::infinity(), 1); },
            "-Inf input");
        expect_rejected(
            [] { (void)quantize_raw_at_scale(1.0, std::numeric_limits<std::uint64_t>::max()); },
            "positive int64 overflow");
        expect_rejected(
            [] { (void)quantize_raw_at_scale(-1.0, std::numeric_limits<std::uint64_t>::max()); },
            "negative int64 overflow");

        const auto dataset = make_dataset();
        const auto local = make_local();
        gatzk::data::validate_input_feature_views(dataset, local, 1, "row_normalize_v1");

        auto dataset_raw_tampered = dataset;
        dataset_raw_tampered.features_fp[0][0] = 0.25;
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(dataset_raw_tampered, local, 1, "row_normalize_v1"); },
            "dataset raw feature tamper");

        auto dataset_quantized_tampered = dataset;
        dataset_quantized_tampered.features[0][0] = FieldElement::zero();
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(dataset_quantized_tampered, local, 1, "row_normalize_v1"); },
            "dataset quantized feature tamper");

        auto local_raw_tampered = local;
        local_raw_tampered.features_fp[0][0] = 0.25;
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(dataset, local_raw_tampered, 1, "row_normalize_v1"); },
            "local raw feature tamper");

        auto local_quantized_tampered = local;
        local_quantized_tampered.features[0][0] = FieldElement::zero();
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(dataset, local_quantized_tampered, 1, "row_normalize_v1"); },
            "local quantized feature tamper");

        auto local_id_tampered = local;
        local_id_tampered.absolute_ids = {1, 0};
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(dataset, local_id_tampered, 1, "row_normalize_v1"); },
            "dataset/local absolute-id binding tamper");

        auto redacted_dataset = dataset;
        redacted_dataset.features_fp.clear();
        redacted_dataset.features.clear();
        // This is the authenticated full-graph context form after the
        // duplicated dataset view has been dropped for memory/privacy.
        gatzk::data::validate_input_feature_views(redacted_dataset, local, 1, "row_normalize_v1");

        auto redacted_local_tampered = local;
        redacted_local_tampered.features_fp[0][0] = 0.25;
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(redacted_dataset, redacted_local_tampered, 1, "row_normalize_v1"); },
            "redacted-context raw view tamper");

        redacted_local_tampered = local;
        redacted_local_tampered.features[0][0] = FieldElement::zero();
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(redacted_dataset, redacted_local_tampered, 1, "row_normalize_v1"); },
            "redacted-context quantized view tamper");

        auto partial_dataset = dataset;
        partial_dataset.features.clear();
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(partial_dataset, local, 1, "row_normalize_v1"); },
            "partially redacted dataset view");

        auto partial_local = local;
        partial_local.num_nodes = 1;
        partial_local.absolute_ids.resize(1);
        partial_local.features_fp.resize(1);
        partial_local.features.resize(1);
        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(redacted_dataset, partial_local, 1, "row_normalize_v1"); },
            "partial local view without dataset source");

        expect_rejected(
            [&] { gatzk::data::validate_input_feature_views(dataset, local, 0, "row_normalize_v1"); },
            "validator zero scale");

        std::cout << "input quantization runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "input quantization runtime: FAIL: " << error.what() << '\n';
        return 1;
    }
}
