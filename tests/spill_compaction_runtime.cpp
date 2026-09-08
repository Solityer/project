#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/field_io.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/util/storage_reservation.hpp"
#include "gatzk/util/trace_spill.hpp"

#include <atomic>
#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <span>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <thread>
#include <vector>

namespace
{
    using gatzk::algebra::DomainEvaluationWeights;
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::Polynomial;
    using gatzk::algebra::RootOfUnityDomain;
    using gatzk::protocol::TraceArtifacts;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename Fn>
    void expect_failure(Fn &&fn, const std::string &message)
    {
        bool failed = false;
        try
        {
            fn();
        }
        catch (const std::exception &)
        {
            failed = true;
        }
        expect(failed, message);
    }

    template <typename Fn>
    void expect_failure_message(
        Fn &&fn,
        const std::string &fragment,
        const std::string &message)
    {
        bool matched = false;
        try
        {
            fn();
        }
        catch (const std::exception &error)
        {
            matched =
                std::string(error.what()).find(fragment) !=
                std::string::npos;
        }
        expect(matched, message);
    }

    void write_values(
        const std::filesystem::path &path,
        const std::vector<FieldElement> &values,
        std::size_t count)
    {
        expect(count <= values.size(), "spill fixture prefix exceeds source");
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        expect(stream.good(), "unable to create spill fixture");
        gatzk::algebra::write_field_elements_binary(
            stream,
            std::span<const FieldElement>(values.data(), count));
        stream.flush();
        expect(stream.good(), "unable to flush spill fixture");
        expect(::chmod(path.c_str(), S_IRUSR | S_IWUSR) == 0,
               "unable to secure spill fixture");
    }

    void append_byte(const std::filesystem::path &path)
    {
        std::ofstream stream(path, std::ios::binary | std::ios::app);
        expect(stream.good(), "unable to append spill fixture");
        const char marker = static_cast<char>(0xa5);
        stream.write(&marker, 1);
        stream.flush();
        expect(stream.good(), "unable to append spill fixture byte");
    }

    void write_noncanonical_field(const std::filesystem::path &path)
    {
        std::array<std::uint8_t, 32> bytes{};
        bytes.fill(0xff);
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        expect(stream.good(), "unable to create noncanonical spill fixture");
        stream.write(
            reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        stream.flush();
        expect(stream.good(), "unable to flush noncanonical spill fixture");
        expect(::chmod(path.c_str(), S_IRUSR | S_IWUSR) == 0,
               "unable to secure noncanonical spill fixture");
    }

    FieldElement non_subgroup_point(
        const std::shared_ptr<RootOfUnityDomain> &domain,
        std::uint64_t seed)
    {
        auto point = FieldElement(seed);
        while (domain->rotation_shift(FieldElement::one(), point).has_value())
        {
            point += FieldElement::one();
        }
        return point;
    }

    TraceArtifacts::SpilledEvaluationPolynomial descriptor(
        const std::filesystem::path &path,
        const std::shared_ptr<RootOfUnityDomain> &domain,
        std::size_t logical_size,
        std::size_t physical_size,
        bool compacted,
        const std::shared_ptr<const gatzk::util::TraceSpillLease> &lease)
    {
        TraceArtifacts::SpilledEvaluationPolynomial spilled;
        spilled.path = path.string();
        spilled.domain_name = domain->name;
        spilled.size = logical_size;
        spilled.physical_size = physical_size;
        spilled.zero_tail_compacted = compacted;
        spilled.lease = lease;
        return spilled;
    }

    std::vector<FieldElement> evaluate(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<RootOfUnityDomain> &domain,
        const std::vector<FieldElement> &points,
        const std::vector<DomainEvaluationWeights> &weight_storage)
    {
        std::vector<const DomainEvaluationWeights *> weights;
        weights.reserve(weight_storage.size());
        for (const auto &entry : weight_storage)
        {
            weights.push_back(&entry);
        }
        return gatzk::protocol::evaluate_spilled_evaluation_polynomial_points(
            spilled,
            domain,
            points,
            weights);
    }

    struct Usage
    {
        long max_rss_kb = 0;
        long in_block = 0;
        long out_block = 0;
    };

    Usage usage()
    {
        rusage value{};
        expect(getrusage(RUSAGE_SELF, &value) == 0, "getrusage failed");
        return Usage{
            .max_rss_kb = value.ru_maxrss,
            .in_block = value.ru_inblock,
            .out_block = value.ru_oublock,
        };
    }

    double timed_scan(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<RootOfUnityDomain> &domain,
        const std::vector<FieldElement> &points,
        const std::vector<DomainEvaluationWeights> &weights,
        const std::vector<FieldElement> &expected)
    {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < 3; ++repeat)
        {
            const auto actual = evaluate(spilled, domain, points, weights);
            expect(actual == expected, "repeated sequential scan changed value");
        }
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    }

    std::uint64_t current_start_ticks()
    {
        std::ifstream stream("/proc/self/stat");
        expect(stream.good(), "unable to read current process stat");
        std::string text;
        std::getline(stream, text);
        const auto close_paren = text.rfind(')');
        expect(
            close_paren != std::string::npos && close_paren + 2 < text.size(),
            "current process stat is malformed");
        std::istringstream fields(text.substr(close_paren + 2));
        std::string token;
        for (int field = 3; field <= 22; ++field)
        {
            expect(static_cast<bool>(fields >> token),
                   "current process stat is truncated");
            if (field == 22)
            {
                std::size_t consumed = 0;
                const auto value = std::stoull(token, &consumed, 10);
                expect(consumed == token.size() && value != 0,
                       "current process start ticks are invalid");
                return static_cast<std::uint64_t>(value);
            }
        }
        throw std::runtime_error("current process start ticks unavailable");
    }
}

int main()
{
    try
    {
        const auto root =
            std::filesystem::temp_directory_path() /
            "gatzk_trace_spill_compaction_runtime";
        std::error_code cleanup_error;
        std::filesystem::remove_all(root, cleanup_error);
        std::filesystem::create_directories(root);
        const auto spill_lease =
            gatzk::util::TraceSpillLease::create(root);
        const auto spill_root = spill_lease->path();

        // Exercise the production process-local reservation under genuine
        // overlap.  One holder remains alive until the second worker has
        // attempted the same near-budget request, so a serial-release bug
        // cannot make both reservations appear successful.
        const auto reservable_budget =
            gatzk::util::FilesystemSpaceReservation::available_budget(root);

        // Ledger staging names are authenticated by pid+start and a CSPRNG
        // nonce.  A crash-before-rename temp from a dead pid is reclaimed,
        // while a temp bearing this live process identity is preserved.  A
        // same-pid, wrong-start fixture models PID reuse and is reclaimable.
        struct stat root_status{};
        expect(::stat(root.c_str(), &root_status) == 0,
               "storage ledger fixture root stat failed");
        const auto ledger_root =
            std::filesystem::temp_directory_path() /
            ".gatzk-storage-ledger";
        std::filesystem::create_directories(ledger_root);
        const auto ledger_prefix =
            std::string(".dev-") +
            std::to_string(static_cast<std::uint64_t>(root_status.st_dev)) +
            ".ledger.tmp-";
        const auto stale_temp = ledger_root /
            (ledger_prefix +
             "999999999-1-0123456789abcdef0123456789abcdef");
        const auto active_temp = ledger_root /
            (ledger_prefix +
             std::to_string(static_cast<std::uint64_t>(::getpid())) + "-" +
             std::to_string(current_start_ticks()) +
             "-abcdefabcdefabcdefabcdefabcdefab");
        const auto reused_pid_temp = ledger_root /
            (ledger_prefix +
             std::to_string(static_cast<std::uint64_t>(::getpid())) +
             "-1-fedcbafedcbafedcbafedcbafedcbafe");
        for (const auto &fixture : {stale_temp, active_temp, reused_pid_temp})
        {
            std::ofstream fixture_stream(fixture, std::ios::binary | std::ios::trunc);
            expect(fixture_stream.good(), "unable to create ledger temp fixture");
            fixture_stream << "crash-before-rename";
            fixture_stream.flush();
            expect(fixture_stream.good(), "unable to write ledger temp fixture");
            expect(::chmod(fixture.c_str(), S_IRUSR | S_IWUSR) == 0,
                   "unable to secure ledger temp fixture");
        }
        (void)gatzk::util::FilesystemSpaceReservation::available_budget(root);
        expect(!std::filesystem::exists(stale_temp),
               "stale ledger temp from crashed writer was not reclaimed");
        expect(std::filesystem::exists(active_temp),
               "active ledger temp was reclaimed");
        expect(!std::filesystem::exists(reused_pid_temp),
               "PID-reuse ledger temp was not reclaimed");
        std::filesystem::remove(active_temp, cleanup_error);
        const auto reservation_request =
            reservable_budget / 2 + 1;
        std::atomic<std::size_t> reservation_successes{0};
        std::atomic<std::size_t> reservation_failures{0};
        std::barrier reservation_start(3);
        std::barrier reservation_done(3);
        const auto reservation_worker = [&]()
        {
            std::unique_ptr<gatzk::util::FilesystemSpaceReservation>
                reservation;
            reservation_start.arrive_and_wait();
            try
            {
                reservation = std::make_unique<
                    gatzk::util::FilesystemSpaceReservation>(
                    root,
                    reservation_request,
                    "spill-compaction-concurrency");
                ++reservation_successes;
            }
            catch (const std::exception &)
            {
                ++reservation_failures;
            }
            reservation_done.arrive_and_wait();
        };
        std::thread reservation_first(reservation_worker);
        std::thread reservation_second(reservation_worker);
        reservation_start.arrive_and_wait();
        reservation_done.arrive_and_wait();
        reservation_first.join();
        reservation_second.join();
        expect(
            reservation_successes.load() <= 1 &&
                reservation_failures.load() >= 1,
            "concurrent storage reservations exceeded the safe budget");

        const auto peer_root = root.parent_path() /
            "gatzk_trace_spill_compaction_runtime_peer";
        std::filesystem::remove_all(peer_root, cleanup_error);
        std::filesystem::create_directories(peer_root);
        {
            gatzk::util::FilesystemSpaceReservation primary_filesystem_holder(
                root,
                reservation_request,
                "spill-compaction-shared-device-primary");
            expect_failure(
                [&]()
                {
                    gatzk::util::FilesystemSpaceReservation peer_filesystem_holder(
                        peer_root,
                        reservation_request,
                        "spill-compaction-shared-device-peer");
                },
                "same-device storage roots did not share the reservation ledger");
        }
        std::filesystem::remove_all(peer_root, cleanup_error);

        // Cross-process ledger ownership must remain visible while the child
        // is alive and must be reclaimable after a crash-style _exit.
        int child_ready[2] = {-1, -1};
        int child_release[2] = {-1, -1};
        expect(::pipe(child_ready) == 0, "storage child ready pipe failed");
        expect(::pipe(child_release) == 0, "storage child release pipe failed");
        const auto storage_child = ::fork();
        expect(storage_child >= 0, "storage reservation fork failed");
        if (storage_child == 0)
        {
            (void)::close(child_ready[0]);
            (void)::close(child_release[1]);
            try
            {
                gatzk::util::FilesystemSpaceReservation child_reservation(
                    root,
                    reservation_request,
                    "spill-compaction-cross-process");
                const char marker = 'R';
                if (::write(child_ready[1], &marker, 1) != 1)
                    ::_exit(2);
                char release = 0;
                if (::read(child_release[0], &release, 1) != 1)
                    ::_exit(2);
                ::_exit(0);
            }
            catch (const std::exception &)
            {
                ::_exit(3);
            }
        }
        (void)::close(child_ready[1]);
        (void)::close(child_release[0]);
        char ready_marker = 0;
        expect(
            ::read(child_ready[0], &ready_marker, 1) == 1 &&
                ready_marker == 'R',
            "storage child reservation did not become active");
        (void)::close(child_ready[0]);
        expect_failure(
            [&]()
            {
                gatzk::util::FilesystemSpaceReservation blocked_reservation(
                    root,
                    reservation_request,
                    "spill-compaction-cross-process-blocked");
            },
            "cross-process storage ledger allowed an over-budget reservation");
        const char release_marker = 'S';
        expect(
            ::write(child_release[1], &release_marker, 1) == 1,
            "storage child release failed");
        (void)::close(child_release[1]);
        int child_status = 0;
        expect(
            ::waitpid(storage_child, &child_status, 0) == storage_child &&
                WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
            "storage child reservation failed");
        {
            gatzk::util::FilesystemSpaceReservation recovered_reservation(
                root,
                reservation_request,
                "spill-compaction-cross-process-recovered");
        }

        std::error_code alias_error;
        const auto alias_root =
            root.parent_path() /
            (root.filename().string() + "_alias");
        std::filesystem::remove(alias_root, alias_error);
        alias_error.clear();
        std::filesystem::create_directory_symlink(
            root,
            alias_root,
            alias_error);
        if (!alias_error && reservable_budget != 0)
        {
            const auto alias_request = reservable_budget / 2 + 1;
            gatzk::util::FilesystemSpaceReservation canonical_holder(
                root,
                alias_request,
                "spill-compaction-canonical-alias");
            expect_failure(
                [&]()
                {
                    gatzk::util::FilesystemSpaceReservation alias_holder(
                        alias_root,
                        alias_request,
                        "spill-compaction-symlink-alias");
                },
                "symlink alias bypassed canonical storage reservation");

            std::error_code relative_error;
            const auto relative_root = std::filesystem::relative(
                root,
                std::filesystem::current_path(),
                relative_error);
            if (!relative_error && !relative_root.empty())
            {
                expect_failure(
                    [&]()
                    {
                        gatzk::util::FilesystemSpaceReservation relative_holder(
                            relative_root,
                            alias_request,
                            "spill-compaction-relative-alias");
                    },
                    "relative alias bypassed canonical storage reservation");
            }
        }
        std::filesystem::remove(alias_root, alias_error);

        if (reservable_budget != 0)
        {
            gatzk::util::FilesystemSpaceReservation released(
                root,
                1,
                "spill-compaction-release");
        }

        const std::size_t logical_size = 1ULL << 16;
        const std::size_t physical_size = logical_size / 2;
        const auto domain = RootOfUnityDomain::create(
            "trace_spill_runtime",
            logical_size);

        std::vector<FieldElement> tail_values(
            logical_size,
            FieldElement::zero());
        for (std::size_t index = 0; index < physical_size; ++index)
        {
            tail_values[index] = FieldElement(index + 1);
        }

        std::vector<FieldElement> dense_values(
            logical_size,
            FieldElement::zero());
        for (std::size_t index = 0; index < logical_size; ++index)
        {
            dense_values[index] = FieldElement(index + 1);
        }

        const auto full_path = spill_root / "full.bin";
        const auto compact_path = spill_root / "compact.bin";
        const auto dense_path = spill_root / "dense.bin";
        const auto zero_path = spill_root / "zero.bin";
        write_values(full_path, tail_values, logical_size);
        write_values(compact_path, tail_values, physical_size);
        write_values(dense_path, dense_values, logical_size);
        write_values(zero_path, tail_values, 0);

        const auto noncanonical_path = spill_root / "noncanonical.bin";
        write_noncanonical_field(noncanonical_path);

        const auto full = descriptor(
            full_path,
            domain,
            logical_size,
            logical_size,
            false,
            spill_lease);
        const auto compact = descriptor(
            compact_path,
            domain,
            logical_size,
            physical_size,
            true,
            spill_lease);
        const auto dense = descriptor(
            dense_path,
            domain,
            logical_size,
            logical_size,
            true,
            spill_lease);
        const auto all_zero = descriptor(
            zero_path,
            domain,
            logical_size,
            0,
            true,
            spill_lease);
        const auto noncanonical = descriptor(
            noncanonical_path,
            domain,
            1,
            1,
            true,
            spill_lease);

        expect(
            gatzk::protocol::spilled_file_value_count(compact) == physical_size,
            "compact physical count mismatch");
        expect(
            gatzk::protocol::spilled_file_value_count(full) == logical_size,
            "uncompacted physical count mismatch");
        expect(
            gatzk::protocol::spilled_file_value_count(dense) == logical_size,
            "dense physical count mismatch");
        expect(
            gatzk::protocol::spilled_file_value_count(all_zero) == 0,
            "all-zero physical count mismatch");
        expect(
            std::filesystem::file_size(compact_path) <
                std::filesystem::file_size(full_path),
            "zero-tail spill was not physically smaller");
        expect(
            std::filesystem::file_size(zero_path) == 0,
            "all-zero spill was not empty");
        expect(
            std::filesystem::file_size(compact_path) ==
                gatzk::algebra::field_elements_binary_size(physical_size),
            "compact spill byte count disagrees with canonical encoding");

        const std::vector<FieldElement> points = {
            non_subgroup_point(domain, 2),
            non_subgroup_point(domain, 3),
            non_subgroup_point(domain, 5),
        };
        std::vector<DomainEvaluationWeights> weights;
        weights.reserve(points.size());
        for (const auto &point : points)
        {
            weights.push_back(
                gatzk::algebra::build_domain_evaluation_weights(
                    domain,
                    point));
            expect(
                !weights.back().direct_index.has_value(),
                "sequential fixture unexpectedly selected a direct point");
        }

        auto missing_empty_spill = all_zero;
        missing_empty_spill.path = (spill_root / "missing-empty.bin").string();
        expect_failure_message(
            [&]()
            {
                (void)evaluate(
                    missing_empty_spill,
                    domain,
                    {},
                    {});
            },
            "failed to open trace spill payload",
            "empty-point evaluation skipped missing spill validation");

        auto nonempty_empty_spill = all_zero;
        nonempty_empty_spill.path = compact_path.string();
        expect_failure_message(
            [&]()
            {
                (void)evaluate(
                    nonempty_empty_spill,
                    domain,
                    {},
                    {});
            },
            "length mismatch",
            "empty-point evaluation skipped spill length validation");

        expect_failure_message(
            [&]()
            {
                (void)evaluate(
                    noncanonical,
                    domain,
                    points,
                    weights);
            },
            "non-canonical field encoding",
            "same-length noncanonical spill encoding was accepted");

        auto bad_domain = compact;
        bad_domain.domain_name = "wrong_domain";
        expect_failure_message(
            [&]()
            {
                (void)evaluate(
                    bad_domain,
                    domain,
                    points,
                    weights);
            },
            "domain metadata mismatch",
            "spilled domain-name mismatch was accepted");

        auto bad_logical_size = compact;
        bad_logical_size.size = domain->size + 1;
        expect_failure_message(
            [&]()
            {
                (void)evaluate(
                    bad_logical_size,
                    domain,
                    points,
                    weights);
            },
            "logical size exceeds",
            "spilled logical size beyond domain was accepted");

        auto bad_full_physical = full;
        bad_full_physical.physical_size = full.size - 1;
        expect_failure_message(
            [&]()
            {
                (void)evaluate(
                    bad_full_physical,
                    domain,
                    points,
                    weights);
            },
            "physical metadata is inconsistent",
            "uncompacted physical metadata mismatch was accepted");

        const auto expected_tail = Polynomial::from_evaluations(
                                       "tail",
                                       tail_values,
                                       domain)
                                       .evaluate(points.front());
        const auto expected_full = evaluate(
            full,
            domain,
            points,
            weights);
        const auto expected_compact = evaluate(
            compact,
            domain,
            points,
            weights);
        expect(
            expected_compact == expected_full,
            "compact sequential evaluation differs from full file");
        expect(
            expected_compact.front() == expected_tail,
            "compact sequential evaluation disagrees with polynomial");

        const auto expected_dense = evaluate(
            dense,
            domain,
            points,
            weights);
        const auto dense_reference = Polynomial::from_evaluations(
                                         "dense",
                                         dense_values,
                                         domain);
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            expect(
                expected_dense[index] == dense_reference.evaluate(points[index]),
                "no-zero-tail sequential evaluation mismatch");
        }

        const auto direct_points = std::vector<FieldElement>{
            domain->point_at(3),
            domain->point_at(physical_size - 1),
            domain->point_at(logical_size - 1),
        };
        std::vector<DomainEvaluationWeights> direct_weights;
        for (const auto &point : direct_points)
        {
            direct_weights.push_back(
                gatzk::algebra::build_domain_evaluation_weights(
                    domain,
                    point));
            expect(
                direct_weights.back().direct_index.has_value(),
                "direct fixture did not select a domain index");
        }
        const auto direct_actual = evaluate(
            compact,
            domain,
            direct_points,
            direct_weights);
        expect(
            direct_actual ==
            std::vector<FieldElement>{
                    tail_values[3],
                    tail_values[physical_size - 1],
                    FieldElement::zero()},
            "direct-index compact tail reconstruction mismatch");

        const auto all_zero_actual = evaluate(
            all_zero,
            domain,
            points,
            weights);
        expect(
            all_zero_actual ==
                std::vector<FieldElement>(
                    points.size(),
                    FieldElement::zero()),
            "all-zero sequential reconstruction mismatch");

        const std::vector<FieldElement> blinder = {
            FieldElement(7),
            FieldElement(11),
        };
        auto compact_blinded = compact;
        compact_blinded.zero_domain_blinder = blinder;
        const auto direct_blinded_actual = evaluate(
            compact_blinded,
            domain,
            direct_points,
            direct_weights);
        expect(
            direct_blinded_actual == direct_actual,
            "domain-point zero-domain blinder changed direct values");
        const auto blinded_reference = [&]()
        {
            auto polynomial = Polynomial::from_evaluations(
                "blinded",
                tail_values,
                domain);
            polynomial.set_zero_domain_blinder(blinder);
            return polynomial;
        }();
        const auto compact_blinded_actual = evaluate(
            compact_blinded,
            domain,
            points,
            weights);
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            expect(
                compact_blinded_actual[index] ==
                    blinded_reference.evaluate(points[index]),
                "compact zero-domain blinder reconstruction mismatch");
        }

        auto all_zero_blinded = all_zero;
        all_zero_blinded.zero_domain_blinder = blinder;
        auto all_zero_reference = Polynomial::from_evaluations(
            "all_zero_blinded",
            std::vector<FieldElement>(
                logical_size,
                FieldElement::zero()),
            domain);
        all_zero_reference.set_zero_domain_blinder(blinder);
        const auto all_zero_blinded_actual = evaluate(
            all_zero_blinded,
            domain,
            points,
            weights);
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            expect(
                all_zero_blinded_actual[index] ==
                    all_zero_reference.evaluate(points[index]),
                "all-zero zero-domain blinder reconstruction mismatch");
        }

        auto bad_physical = compact;
        bad_physical.physical_size = logical_size + 1;

        const auto expect_kzg_reader_failure =
            [&](const TraceArtifacts::SpilledEvaluationPolynomial &candidate,
                const std::string &message)
        {
            const gatzk::crypto::OpeningSemanticSourceResolver
                semantic_resolver =
                    [&](const gatzk::crypto::Commitment &)
                    -> std::optional<
                        gatzk::crypto::OpeningSemanticSourceInfo>
                {
                    return gatzk::crypto::OpeningSemanticSourceInfo{
                        .key = "trace-spill-semantic-source",
                        .domain = domain,
                        .value_count = candidate.size,
                        .stored_value_count =
                            candidate.zero_tail_compacted
                                ? candidate.physical_size
                                : candidate.size,
                        .zero_tail_compacted =
                            candidate.zero_tail_compacted,
                        .zero_domain_blinder =
                            candidate.zero_domain_blinder,
                        .file_path = candidate.path,
                        .file_backed = true,
                        .lease = spill_lease,
                    };
                };
            const gatzk::crypto::OpeningPolynomialResolver
                opening_resolver =
                    [](const gatzk::crypto::Commitment &)
                    -> std::shared_ptr<const Polynomial>
                {
                    return nullptr;
                };
            const std::vector<gatzk::crypto::Commitment>
                commitments = {
                    gatzk::crypto::Commitment{.name = "tail-a"},
                    gatzk::crypto::Commitment{.name = "tail-b"},
                };
            const std::vector<FieldElement> claimed_values = {
                FieldElement::zero(),
                FieldElement::zero(),
            };
            gatzk::crypto::KZGKeyPair dummy_key;
            const auto expected_fragment =
                candidate.physical_size > candidate.size
                    ? std::string("physical count")
                    : std::string("length mismatch");
            expect_failure_message(
                [&]()
                {
                    (void)gatzk::crypto::KZG::open_point_group(
                        commitments,
                        points.front(),
                        claimed_values,
                        FieldElement(5),
                        dummy_key,
                        nullptr,
                        nullptr,
                        &opening_resolver,
                        &semantic_resolver);
                },
                expected_fragment,
                message);
        };

        const gatzk::crypto::OpeningSemanticSourceResolver
            noncanonical_semantic_resolver =
                [&](const gatzk::crypto::Commitment &)
                -> std::optional<
                    gatzk::crypto::OpeningSemanticSourceInfo>
        {
            return gatzk::crypto::OpeningSemanticSourceInfo{
                .key = "noncanonical-spill-source",
                .domain = domain,
                .value_count = noncanonical.size,
                .stored_value_count = noncanonical.physical_size,
                .zero_tail_compacted = true,
                .file_path = noncanonical.path,
                .file_backed = true,
                .lease = spill_lease,
            };
        };
        const gatzk::crypto::OpeningPolynomialResolver
            noncanonical_opening_resolver =
                [](const gatzk::crypto::Commitment &)
                -> std::shared_ptr<const Polynomial>
        {
            return nullptr;
        };
        expect_failure_message(
            [&]()
            {
                (void)gatzk::crypto::KZG::open_point_group(
                    {gatzk::crypto::Commitment{.name = "noncanonical"}},
                    points.front(),
                    {FieldElement::zero()},
                    FieldElement(5),
                    gatzk::crypto::KZGKeyPair{},
                    nullptr,
                    nullptr,
                    &noncanonical_opening_resolver,
                    &noncanonical_semantic_resolver);
            },
            "non-canonical field encoding",
            "KZG accepted same-length noncanonical spill encoding");

        expect_kzg_reader_failure(
            bad_physical,
            "KZG accepted a physical count larger than logical size");
        expect_failure(
            [&]()
            {
                (void)evaluate(
                    bad_physical,
                    domain,
                    points,
                    weights);
            },
            "physical count larger than logical size was accepted");

        write_values(
            compact_path,
            tail_values,
            physical_size - 1);
        expect_kzg_reader_failure(
            compact,
            "KZG accepted a truncated compact spill");
        expect_failure(
            [&]()
            {
                (void)evaluate(
                    compact,
                    domain,
                    points,
                    weights);
            },
            "truncated compact spill was accepted");

        write_values(compact_path, tail_values, physical_size);
        append_byte(compact_path);
        expect_kzg_reader_failure(
            compact,
            "KZG accepted a compact spill with trailing bytes");
        expect_failure(
            [&]()
            {
                (void)evaluate(
                    compact,
                    domain,
                    points,
                    weights);
            },
            "compact spill with trailing bytes was accepted");

        write_values(compact_path, tail_values, physical_size);

        const auto alternate_path = spill_root / "compact_alt.bin";
        write_values(
            alternate_path,
            tail_values,
            physical_size - 1);
        const auto alternate_compact = descriptor(
            alternate_path,
            domain,
            logical_size,
            physical_size - 1,
            true,
            spill_lease);
        const gatzk::crypto::OpeningSemanticSourceResolver
            conflicting_semantic_resolver =
                [&](const gatzk::crypto::Commitment &commitment)
                -> std::optional<
                    gatzk::crypto::OpeningSemanticSourceInfo>
            {
                const auto &source =
                    commitment.name == "tail-a"
                        ? compact
                        : alternate_compact;
                return gatzk::crypto::OpeningSemanticSourceInfo{
                    .key = "trace-spill-semantic-source",
                    .domain = domain,
                    .value_count = source.size,
                    .stored_value_count = source.physical_size,
                    .zero_tail_compacted = true,
                    .file_path = source.path,
                    .file_backed = true,
                    .lease = spill_lease,
                };
            };
        const gatzk::crypto::OpeningPolynomialResolver
            conflicting_opening_resolver =
                [](const gatzk::crypto::Commitment &)
                -> std::shared_ptr<const Polynomial>
            {
                return nullptr;
            };
        const std::vector<gatzk::crypto::Commitment>
            conflicting_commitments = {
                gatzk::crypto::Commitment{.name = "tail-a"},
                gatzk::crypto::Commitment{.name = "tail-b"},
            };
        const std::vector<FieldElement> conflicting_claims = {
            FieldElement::zero(),
            FieldElement::zero(),
        };
        gatzk::crypto::KZGKeyPair conflicting_dummy_key;
        expect_failure_message(
            [&]()
            {
                (void)gatzk::crypto::KZG::open_point_group(
                    conflicting_commitments,
                    points.front(),
                    conflicting_claims,
                    FieldElement(5),
                    conflicting_dummy_key,
                    nullptr,
                    nullptr,
                    &conflicting_opening_resolver,
                    &conflicting_semantic_resolver);
            },
            "physical metadata",
            "KZG accepted conflicting compact physical metadata");

        // A singleton id-zero descriptor must still use the production
        // streamed opening path.  The synthetic key is intentionally empty:
        // the assertion is made after the fold planner has consumed the
        // source, before the later witness MSM rejects the absent SRS.  A
        // full-vector opening resolver would increment this counter.
        std::size_t singleton_opening_resolver_calls = 0;
        const gatzk::crypto::OpeningPolynomialResolver
            singleton_opening_resolver =
                [&](const gatzk::crypto::Commitment &)
                -> std::shared_ptr<const Polynomial>
        {
            ++singleton_opening_resolver_calls;
            return nullptr;
        };
        const gatzk::crypto::OpeningSemanticSourceResolver
            singleton_semantic_resolver =
                [&](const gatzk::crypto::Commitment &)
                -> std::optional<
                    gatzk::crypto::OpeningSemanticSourceInfo>
        {
            return gatzk::crypto::OpeningSemanticSourceInfo{
                .key = "singleton-id-zero-source",
                .domain = domain,
                .value_count = compact.size,
                .stored_value_count = compact.physical_size,
                .zero_tail_compacted = true,
                .zero_domain_blinder = blinder,
                .file_path = compact.path,
                .file_backed = true,
                .lease = spill_lease,
            };
        };
        gatzk::crypto::BatchOpeningProfile singleton_profile;
        try
        {
            (void)gatzk::crypto::KZG::open_point_group(
                {gatzk::crypto::Commitment{.name = "singleton-id-zero"}},
                points.front(),
                {FieldElement(17)},
                FieldElement(5),
                gatzk::crypto::KZGKeyPair{},
                &singleton_profile,
                nullptr,
                &singleton_opening_resolver,
                &singleton_semantic_resolver);
        }
        catch (const std::exception &)
        {
            // The empty synthetic key is expected to fail only after the
            // streamed fold has been assembled.
        }
        expect(
            singleton_opening_resolver_calls == 0,
            "singleton id-zero opening fell back to full-vector resolver");
        expect(
            singleton_profile.streamed_semantic_source_scan_count == 1 &&
                singleton_profile.streamed_semantic_source_scan_bytes ==
                    gatzk::algebra::field_elements_binary_size(
                        compact.physical_size),
            "singleton id-zero opening did not perform one bounded source scan");

        // Two independent point groups may share the authenticated source
        // bytes even when their commitments/blinders differ.  The paired
        // planner must fuse only the physical traversal, leaving each fold's
        // claims and blinder correction independent.
        std::size_t pair_opening_resolver_calls = 0;
        const gatzk::crypto::OpeningPolynomialResolver
            pair_opening_resolver =
                [&](const gatzk::crypto::Commitment &)
                -> std::shared_ptr<const Polynomial>
        {
            ++pair_opening_resolver_calls;
            return nullptr;
        };
        const gatzk::crypto::OpeningSemanticSourceResolver
            pair_semantic_resolver =
                [&](const gatzk::crypto::Commitment &commitment)
                -> std::optional<
                    gatzk::crypto::OpeningSemanticSourceInfo>
        {
            return gatzk::crypto::OpeningSemanticSourceInfo{
                .key = "paired-shared-source",
                .domain = domain,
                .value_count = compact.size,
                .stored_value_count = compact.physical_size,
                .zero_tail_compacted = true,
                .zero_domain_blinder =
                    commitment.name == "pair-a"
                        ? std::vector<FieldElement>{FieldElement(7)}
                        : std::vector<FieldElement>{FieldElement(11)},
                .file_path = compact.path,
                .file_backed = true,
                .lease = spill_lease,
            };
        };
        std::array<gatzk::crypto::BatchOpeningProfile, 2>
            pair_profiles;
        try
        {
            (void)gatzk::crypto::KZG::open_point_group_pair_shared_stream(
                std::array<std::vector<gatzk::crypto::Commitment>, 2>{
                    std::vector<gatzk::crypto::Commitment>{
                        gatzk::crypto::Commitment{.name = "pair-a"}},
                    std::vector<gatzk::crypto::Commitment>{
                        gatzk::crypto::Commitment{.name = "pair-b"}}},
                std::array<FieldElement, 2>{points[0], points[1]},
                std::array<std::vector<FieldElement>, 2>{
                    std::vector<FieldElement>{FieldElement(17)},
                    std::vector<FieldElement>{FieldElement(19)}},
                std::array<FieldElement, 2>{FieldElement(5), FieldElement(7)},
                gatzk::crypto::KZGKeyPair{},
                &pair_profiles,
                nullptr,
                &pair_opening_resolver,
                &pair_semantic_resolver);
        }
        catch (const std::exception &)
        {
            // As above, the empty synthetic SRS is expected to fail after
            // streamed fold construction.
        }
        expect(
            pair_opening_resolver_calls == 0,
            "paired streamed opening invoked the full-vector resolver");
        expect(
            pair_profiles[0].streamed_semantic_source_scan_count == 1 &&
                pair_profiles[1].streamed_semantic_source_scan_count == 0 &&
                pair_profiles[0].streamed_semantic_source_scan_bytes ==
                    gatzk::algebra::field_elements_binary_size(
                        compact.physical_size),
            "paired streamed opening did not fuse one physical source scan");

        TraceArtifacts trace;
        trace.polynomial_domains["tail"] = domain->name;
        trace.spilled_polynomials.emplace(
            "tail",
            compact);
        trace.commitments.emplace(
            "tail",
            gatzk::crypto::Commitment{.name = "tail"});
        const auto resolved =
            gatzk::protocol::resolve_trace_opening_polynomial(
                trace,
                "tail",
                domain);
        expect(
            resolved->values() == tail_values,
            "opening materialization did not restore the logical zero tail");

        const auto before = usage();
        for (std::size_t repeat = 0; repeat < 3; ++repeat)
        {
            const auto full_ms = timed_scan(
                full,
                domain,
                points,
                weights,
                expected_full);
            const auto compact_ms = timed_scan(
                compact,
                domain,
                points,
                weights,
                expected_compact);
            const auto after = usage();
            std::cout << "spill_compaction_runtime: repeat=" << repeat
                      << " full_ms=" << full_ms
                      << " compact_ms=" << compact_ms
                      << " full_bytes=" << std::filesystem::file_size(full_path)
                      << " compact_bytes=" << std::filesystem::file_size(compact_path)
                      << " maxrss_kb=" << after.max_rss_kb
                      << " read_blocks_delta=" << (after.in_block - before.in_block)
                      << " write_blocks_delta=" << (after.out_block - before.out_block)
                      << '\n';
        }

        std::filesystem::remove_all(root, cleanup_error);
        std::cout << "spill_compaction_runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "spill_compaction_runtime: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
