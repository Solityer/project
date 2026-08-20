#include "gatzk/algebra/field_io.hpp"

#include <algorithm>
#include <cstdint>
#include <future>
#include <istream>
#include <ostream>
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
                        values[offset + i] =
                            FieldElement::
                                from_little_endian_mod(
                                    buffer.data() +
                                        i * kFieldBytes,
                                    kFieldBytes);
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