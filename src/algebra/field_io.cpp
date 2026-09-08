#include "gatzk/algebra/field_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

namespace gatzk::algebra
{
    namespace
    {

        constexpr std::size_t kFieldBytes = 32;
        constexpr std::size_t kIoBufferBytes = 16ULL << 20;
        constexpr std::size_t kFieldsPerBuffer =
            kIoBufferBytes / kFieldBytes;
        constexpr std::size_t kMinFieldsPerWorker = 1ULL << 15;
        constexpr std::size_t kMaxFieldIoWorkers = 8;

        template <typename Fn>
        void parallel_for_fields(
            std::size_t count,
            Fn &&fn)
        {
            if (count == 0)
            {
                return;
            }

            const auto hardware_workers =
                std::max<std::size_t>(
                    1,
                    std::thread::hardware_concurrency());

            const auto requested_workers =
                std::max<std::size_t>(
                    1,
                    (count + kMinFieldsPerWorker - 1) /
                        kMinFieldsPerWorker);

            const auto worker_count =
                std::min<std::size_t>(
                    kMaxFieldIoWorkers,
                    std::min(
                        hardware_workers,
                        requested_workers));

            if (worker_count <= 1)
            {
                fn(0, count);
                return;
            }

            const auto chunk_size =
                (count + worker_count - 1) /
                worker_count;

            std::vector<std::future<void>> futures;
            futures.reserve(worker_count);

            for (std::size_t worker = 0;
                 worker < worker_count;
                 ++worker)
            {
                const auto begin =
                    worker * chunk_size;
                const auto end =
                    std::min(
                        count,
                        begin + chunk_size);

                if (begin >= end)
                {
                    break;
                }

                futures.push_back(
                    std::async(
                        std::launch::async,
                        [&, begin, end]()
                        {
                            fn(begin, end);
                        }));
            }

            for (auto &future : futures)
            {
                future.get();
            }
        }

    } // namespace

    std::size_t field_elements_binary_size(
        std::size_t value_count)
    {
        if (value_count >
            std::numeric_limits<std::size_t>::max() /
                kFieldBytes)
        {
            throw std::runtime_error(
                "field-element spill byte size overflow");
        }
        return value_count * kFieldBytes;
    }

    void require_field_elements_binary_size(
        const std::filesystem::path &path,
        std::size_t value_count,
        std::string_view operation)
    {
        const auto expected_bytes =
            field_elements_binary_size(value_count);

        std::error_code error;
        const auto actual_bytes =
            std::filesystem::file_size(path, error);
        if (error)
        {
            throw std::runtime_error(
                std::string(operation) +
                " failed to stat field-element spill: " +
                path.string() + ": " +
                error.message());
        }

        if (actual_bytes !=
            static_cast<std::uintmax_t>(expected_bytes))
        {
            throw std::runtime_error(
                std::string(operation) +
                " field-element spill length mismatch: " +
                path.string() + " expected=" +
                std::to_string(expected_bytes) +
                " actual=" +
                std::to_string(actual_bytes));
        }
    }

    void write_field_elements_binary(
        std::ostream &stream,
        std::span<const FieldElement> values)
    {
        if (values.empty())
        {
            return;
        }

        std::vector<std::uint8_t> buffer(
            kFieldsPerBuffer * kFieldBytes);

        for (std::size_t offset = 0;
             offset < values.size();
             offset += kFieldsPerBuffer)
        {
            const auto count =
                std::min(
                    kFieldsPerBuffer,
                    values.size() - offset);

            parallel_for_fields(
                count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t i = begin;
                         i < end;
                         ++i)
                    {
                        values[offset + i]
                            .write_little_endian(
                                buffer.data() +
                                    i * kFieldBytes,
                                kFieldBytes);
                    }
                });

            stream.write(
                reinterpret_cast<const char *>(
                    buffer.data()),
                static_cast<std::streamsize>(
                    count * kFieldBytes));

            if (!stream)
            {
                throw std::runtime_error(
                    "failed to write field-element spill data");
            }
        }
    }

    void read_field_elements_binary_into(
        std::istream &stream,
        std::span<FieldElement> values)
    {
        if (values.empty())
        {
            return;
        }

        std::vector<std::uint8_t> buffer(
            kFieldsPerBuffer * kFieldBytes);

        for (std::size_t offset = 0;
             offset < values.size();
             offset += kFieldsPerBuffer)
        {
            const auto count =
                std::min(
                    kFieldsPerBuffer,
                    values.size() - offset);

            stream.read(
                reinterpret_cast<char *>(
                    buffer.data()),
                static_cast<std::streamsize>(
                    count * kFieldBytes));

            if (!stream)
            {
                throw std::runtime_error(
                    "failed to read field-element spill data");
            }

            parallel_for_fields(
                count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t i = begin;
                         i < end;
                         ++i)
                    {
                        const auto *encoded =
                            buffer.data() + i * kFieldBytes;
                        const auto value =
                            FieldElement::from_little_endian_mod(
                                encoded,
                                kFieldBytes);
                        std::array<std::uint8_t, kFieldBytes>
                            canonical{};
                        value.write_little_endian(
                            canonical.data(),
                            canonical.size());
                        if (std::memcmp(
                                encoded,
                                canonical.data(),
                                canonical.size()) != 0)
                        {
                            throw std::runtime_error(
                                "non-canonical field encoding in spill data");
                        }
                        values[offset + i] = value;
                    }
                });
        }
    }

    std::vector<FieldElement> read_field_elements_binary(
        std::istream &stream,
        std::size_t value_count)
    {
        std::vector<FieldElement> values(
            value_count,
            FieldElement::zero());

        read_field_elements_binary_into(
            stream,
            std::span<FieldElement>(
                values.data(),
                values.size()));

        return values;
    }

} // namespace gatzk::algebra
