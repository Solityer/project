#include "gatzk/util/trace_spill.hpp"

#include <array>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{

    using gatzk::util::TraceSpillLease;

    void expect(
        bool condition,
        const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename Fn>
    void expect_failure(
        Fn &&fn,
        const std::string &message)
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

    mode_t mode_of(
        const std::filesystem::path &path)
    {
        struct stat status{};
        expect(
            ::lstat(path.c_str(), &status) == 0,
            "lstat failed for " + path.string());
        return status.st_mode & 0777;
    }

    std::set<std::filesystem::path> proof_directories(
        const std::filesystem::path &root)
    {
        std::set<std::filesystem::path> result;
        std::error_code error;
        for (std::filesystem::directory_iterator it(root, error), end;
             !error && it != end;
             it.increment(error))
        {
            const auto path = it->path();
            const auto name = path.filename().string();
            if (name.size() != 38 || name.substr(0, 6) != "proof-")
            {
                continue;
            }
            struct stat status{};
            if (::lstat(path.c_str(), &status) == 0 &&
                S_ISDIR(status.st_mode))
            {
                result.insert(path);
            }
        }
        expect(!error, "failed to scan trace spill test root");
        return result;
    }

    void write_payload(
        const std::shared_ptr<const TraceSpillLease> &lease,
        const std::string &name)
    {
        auto stream = lease->create_payload_stream(name);
        (*stream) << "private trace payload";
        stream->flush();
        expect(
            static_cast<bool>(*stream),
            "failed to flush trace spill test payload");
    }

    pid_t fork_crashed_creator(
        const std::filesystem::path &root,
        const std::string &payload_name)
    {
        const auto child = ::fork();
        expect(child >= 0, "fork failed for crashed trace spill creator");
        if (child == 0)
        {
            try
            {
                const auto lease = TraceSpillLease::create(root);
                write_payload(lease, payload_name);
                // Deliberately bypass C++ destruction.  The kernel must
                // release the per-lease flock so the next collector can
                // distinguish this crash from an active lease.
                ::_exit(0);
            }
            catch (const std::exception &)
            {
                ::_exit(2);
            }
        }
        return child;
    }

    pid_t fork_active_creator(
        const std::filesystem::path &root,
        int *release_fd)
    {
        int ready[2] = {-1, -1};
        int release[2] = {-1, -1};
        expect(::pipe(ready) == 0, "ready pipe creation failed");
        expect(::pipe(release) == 0, "release pipe creation failed");
        const auto child = ::fork();
        expect(child >= 0, "fork failed for active trace spill creator");
        if (child == 0)
        {
            (void)::close(ready[0]);
            (void)::close(release[1]);
            try
            {
                const auto lease = TraceSpillLease::create(root);
                write_payload(lease, "cross-process-active.bin");
                const char marker = 'R';
                if (::write(ready[1], &marker, 1) != 1)
                {
                    ::_exit(2);
                }
                char stop = 0;
                if (::read(release[0], &stop, 1) != 1)
                {
                    ::_exit(2);
                }
                ::_exit(0);
            }
            catch (const std::exception &)
            {
                ::_exit(2);
            }
        }
        (void)::close(ready[1]);
        (void)::close(release[0]);
        char marker = 0;
        expect(::read(ready[0], &marker, 1) == 1 && marker == 'R',
               "active trace spill creator did not become ready");
        (void)::close(ready[0]);
        *release_fd = release[1];
        return child;
    }

    void wait_success(
        pid_t child,
        const std::string &message)
    {
        int status = 0;
        expect(::waitpid(child, &status, 0) == child, "waitpid failed");
        expect(
            WIFEXITED(status) && WEXITSTATUS(status) == 0,
            message);
    }

    pid_t fork_before_create(
        const std::filesystem::path &root)
    {
        const auto child = ::fork();
        expect(child >= 0, "fork failed before trace spill create");
        if (child == 0)
        {
            try
            {
                {
                    const auto lease = TraceSpillLease::create(root);
                    write_payload(lease, "fork-before-create.bin");
                }
                ::_exit(0);
            }
            catch (const std::exception &)
            {
                ::_exit(2);
            }
        }
        return child;
    }

    pid_t fork_after_create_and_reset(
        std::shared_ptr<const TraceSpillLease> lease)
    {
        const auto child = ::fork();
        expect(child >= 0, "fork failed after trace spill create");
        if (child == 0)
        {
            // The inherited object is deliberately destroyed in the child.
            // Creator identity must prevent it from deleting the parent's
            // lease directory.
            lease.reset();
            ::_exit(0);
        }
        return child;
    }

    void expect_parent_lock_survives_child_destructor(
        const std::shared_ptr<const TraceSpillLease> &lease)
    {
        const auto child = ::fork();
        expect(child >= 0, "fork failed for inherited lock destructor probe");
        if (child == 0)
        {
            auto inherited = lease;
            inherited.reset();
            ::_exit(0);
        }
        wait_success(
            child,
            "child destructor lock probe failed");

        int result_pipe[2] = {-1, -1};
        expect(::pipe(result_pipe) == 0, "lock probe pipe failed");
        const auto probe = ::fork();
        expect(probe >= 0, "fork failed for third-process lock probe");
        if (probe == 0)
        {
            (void)::close(result_pipe[0]);
            const auto fd = ::open(
                (lease->path() / ".lease.lock").c_str(),
                O_RDWR | O_CLOEXEC | O_NOFOLLOW);
            char result = 'E';
            if (fd >= 0)
            {
                if (::flock(fd, LOCK_EX | LOCK_NB) == 0)
                    result = 'U';
                else if (errno == EWOULDBLOCK || errno == EAGAIN)
                    result = 'B';
                (void)::close(fd);
            }
            if (::write(result_pipe[1], &result, 1) != 1)
                ::_exit(3);
            ::_exit(result == 'B' ? 0 : 2);
        }
        (void)::close(result_pipe[1]);
        char result = 0;
        expect(::read(result_pipe[0], &result, 1) == 1,
               "lock probe did not return a result");
        (void)::close(result_pipe[0]);
        wait_success(
            probe,
            "child destructor released the parent's per-lease lock");
        expect(result == 'B', "third process acquired parent's lease lock");
    }

    pid_t fork_dead_parent_child_holder(
        const std::filesystem::path &root,
        int *release_fd,
        int *done_fd)
    {
        int ready[2] = {-1, -1};
        int release[2] = {-1, -1};
        int done[2] = {-1, -1};
        expect(::pipe(ready) == 0, "dead-parent ready pipe failed");
        expect(::pipe(release) == 0, "dead-parent release pipe failed");
        expect(::pipe(done) == 0, "dead-parent done pipe failed");
        const auto child = ::fork();
        expect(child >= 0, "fork failed for dead-parent lease holder");
        if (child == 0)
        {
            (void)::close(ready[0]);
            (void)::close(release[1]);
            (void)::close(done[0]);
            try
            {
                auto lease = TraceSpillLease::create(root);
                write_payload(lease, "dead-parent-child-holder.bin");
                const auto grandchild = ::fork();
                if (grandchild < 0)
                    ::_exit(2);
                if (grandchild == 0)
                {
                    (void)::close(ready[1]);
                    (void)::close(done[0]);
                    char stop = 0;
                    if (::read(release[0], &stop, 1) != 1)
                        ::_exit(2);
                    lease.reset();
                    const char marker = 'D';
                    if (::write(done[1], &marker, 1) != 1)
                        ::_exit(2);
                    ::_exit(0);
                }
                (void)::close(done[1]);
                const char marker = 'R';
                if (::write(ready[1], &marker, 1) != 1)
                    ::_exit(2);
                // The parent process exits without running the lease
                // destructor; the grandchild retains the inherited lock FD.
                ::_exit(0);
            }
            catch (const std::exception &)
            {
                ::_exit(3);
            }
        }
        (void)::close(ready[1]);
        (void)::close(release[0]);
        (void)::close(done[1]);
        char ready_marker = 0;
        expect(
            ::read(ready[0], &ready_marker, 1) == 1 &&
                ready_marker == 'R',
            "dead-parent holder did not become ready");
        (void)::close(ready[0]);
        *release_fd = release[1];
        *done_fd = done[0];
        return child;
    }

    pid_t fork_exec_payload_probe(
        const std::shared_ptr<const TraceSpillLease> &lease)
    {
        const auto child = ::fork();
        expect(child >= 0, "fork failed for trace spill exec probe");
        if (child == 0)
        {
            try
            {
                auto stream = lease->create_payload_stream("exec-probe.bin");
                (*stream) << "exec probe";
                stream->flush();
                int payload_fd = -1;
                const auto descriptor_limit = ::getdtablesize();
                for (int fd = 3; fd < descriptor_limit; ++fd)
                {
                    std::array<char, 512> target{};
                    const auto count = ::readlink(
                        (std::string("/proc/self/fd/") +
                         std::to_string(fd)).c_str(),
                        target.data(),
                        target.size() - 1);
                    if (count <= 0)
                    {
                        continue;
                    }
                    target[static_cast<std::size_t>(count)] = '\0';
                    if (std::string_view(target.data(), static_cast<std::size_t>(count))
                            .find("/exec-probe.bin") != std::string_view::npos)
                    {
                        payload_fd = fd;
                        break;
                    }
                }
                if (payload_fd < 0)
                {
                    ::_exit(2);
                }
                const auto fd_text = std::to_string(payload_fd);
                ::execl(
                    "/proc/self/exe",
                    "gatzk_trace_spill_lease_test",
                    "--fd-probe",
                    fd_text.c_str(),
                    static_cast<char *>(nullptr));
            }
            catch (const std::exception &)
            {
            }
            ::_exit(3);
        }
        return child;
    }

} // namespace

int main(int argc, char **argv)
{
    if (argc == 3 && std::string_view(argv[1]) == "--fd-probe")
    {
        const auto fd = std::stoi(argv[2]);
        errno = 0;
        return ::fcntl(fd, F_GETFD) < 0 && errno == EBADF ? 0 : 1;
    }
    const auto root =
        std::filesystem::temp_directory_path() /
        "gatzk_trace_spill_lease_runtime";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    try
    {
        std::filesystem::create_directories(root);
        const auto root_alias = root.parent_path() /
            "gatzk_trace_spill_lease_root_alias";
        std::filesystem::remove(root_alias, cleanup_error);
        std::filesystem::create_directory_symlink(root, root_alias);
        expect_failure(
            [&]()
            {
                (void)TraceSpillLease::create(root_alias);
            },
            "symlinked spill root was accepted");
        std::filesystem::remove(root_alias, cleanup_error);
        auto active = TraceSpillLease::create(root);
        const auto active_path = active->path();
        write_payload(active, "active.bin");

        expect(mode_of(root) == 0700, "spill root is not mode 0700");
        expect(mode_of(active_path) == 0700, "lease directory is not mode 0700");
        expect(mode_of(active_path / ".lease.meta") == 0600,
               "lease metadata is not mode 0600");
        expect(mode_of(active_path / ".lease.lock") == 0600,
               "lease lock is not mode 0600");
        expect(mode_of(active_path / "active.bin") == 0600,
               "payload is not mode 0600");

        const auto before_fork_create = proof_directories(root);
        const auto before_fork_child = fork_before_create(root);
        wait_success(
            before_fork_child,
            "fork-before-create trace spill child failed");
        expect(
            proof_directories(root) == before_fork_create,
            "normal child lease destructor changed the parent's lease set");

        const auto after_create_child = fork_after_create_and_reset(active);
        wait_success(
            after_create_child,
            "fork-after-create child destructor failed");
        expect(
            std::filesystem::exists(active_path),
            "fork child destructor removed the parent's active lease");
        expect_parent_lock_survives_child_destructor(active);

        const auto exec_probe_child = fork_exec_payload_probe(active);
        wait_success(
            exec_probe_child,
            "trace spill payload FD leaked across exec");
        expect(
            std::filesystem::exists(active_path / "exec-probe.bin"),
            "exec probe payload was not created through the lease");

        TraceSpillLease::collect_stale(root);
        expect(
            std::filesystem::exists(active_path),
            "active lease was collected while its FD was held");

        int active_release_fd = -1;
        const auto cross_process_before = proof_directories(root);
        const auto active_child =
            fork_active_creator(root, &active_release_fd);
        const auto cross_process_active = proof_directories(root);
        TraceSpillLease::collect_stale(root);
        expect(
            proof_directories(root) == cross_process_active,
            "cross-process active lease was collected");
        const char stop = 'S';
        expect(::write(active_release_fd, &stop, 1) == 1,
               "failed to release active child lease");
        (void)::close(active_release_fd);
        wait_success(
            active_child,
            "active trace spill creator failed");
        TraceSpillLease::collect_stale(root);
        expect(
            proof_directories(root) == cross_process_before,
            "released cross-process lease was not collected");

        // Explicitly cover the parent-dead/child-held-FD case: the metadata
        // creator is gone, but the inherited lock keeps GC fail-closed until
        // the surviving child exits.
        int dead_parent_release_fd = -1;
        int dead_parent_done_fd = -1;
        const auto dead_parent_before = proof_directories(root);
        const auto dead_parent_child = fork_dead_parent_child_holder(
            root,
            &dead_parent_release_fd,
            &dead_parent_done_fd);
        int dead_parent_status = 0;
        expect(
            ::waitpid(dead_parent_child, &dead_parent_status, 0) ==
                dead_parent_child &&
                WIFEXITED(dead_parent_status) &&
                WEXITSTATUS(dead_parent_status) == 0,
            "dead-parent lease creator failed");
        TraceSpillLease::collect_stale(root);
        expect(
            proof_directories(root).size() == dead_parent_before.size() + 1,
            "GC removed a lease whose inherited child lock was active");
        const char dead_parent_release = 'S';
        expect(
            ::write(
                dead_parent_release_fd,
                &dead_parent_release,
                1) == 1,
            "failed to release dead-parent child holder");
        (void)::close(dead_parent_release_fd);
        char dead_parent_done = 0;
        expect(
            ::read(dead_parent_done_fd, &dead_parent_done, 1) == 1 &&
                dead_parent_done == 'D',
            "dead-parent child holder did not release");
        (void)::close(dead_parent_done_fd);
        TraceSpillLease::collect_stale(root);
        expect(
            proof_directories(root) == dead_parent_before,
            "released dead-parent child lease was not collected");

        expect_failure(
            [&]()
            {
                (void)active->create_payload_stream("../escape.bin");
            },
            "payload traversal was accepted");
        expect_failure(
            [&]()
            {
                (void)active->create_payload_stream("payload.txt");
            },
            "non-binary payload name was accepted");
        write_payload(active, "collision.bin");
        expect_failure(
            [&]()
            {
                (void)active->create_payload_stream("collision.bin");
            },
            "payload collision was silently truncated");

        const auto outside = root.parent_path() / "trace_spill_lease_outside";
        std::filesystem::remove(outside, cleanup_error);
        {
            std::ofstream stream(outside);
            stream << "outside";
        }
        const auto symlink_payload = active_path / "symlink.bin";
        std::filesystem::create_symlink(outside, symlink_payload);
        expect_failure(
            [&]()
            {
                (void)active->create_payload_stream("symlink.bin");
            },
            "symlink payload collision was accepted");
        expect(
            std::filesystem::exists(outside),
            "symlink collision damaged an outside file");

        const auto before_crash = proof_directories(root);
        const auto crashed_child =
            fork_crashed_creator(root, "crashed.bin");
        wait_success(
            crashed_child,
            "crashed trace spill creator failed");
        const auto after_crash = proof_directories(root);
        expect(
            after_crash.size() == before_crash.size() + 1,
            "crashed creator did not leave exactly one lease");
        TraceSpillLease::collect_stale(root);
        const auto after_crash_gc = proof_directories(root);
        expect(
            after_crash_gc == before_crash,
            "stale crashed lease was not safely collected");

        const auto before_hardlink = proof_directories(root);
        const auto hardlink_child =
            fork_crashed_creator(root, "hardlink.bin");
        wait_success(
            hardlink_child,
            "hardlink lease creator failed");
        std::filesystem::path hardlink_lease;
        for (const auto &candidate : proof_directories(root))
        {
            if (!before_hardlink.contains(candidate) &&
                candidate != active_path)
            {
                hardlink_lease = candidate;
            }
        }
        expect(!hardlink_lease.empty(), "hardlink lease was not created");
        const auto hardlink_outside =
            root.parent_path() / "trace_spill_hardlink_outside.bin";
        std::filesystem::remove(hardlink_outside, cleanup_error);
        std::filesystem::create_hard_link(
            hardlink_lease / "hardlink.bin",
            hardlink_outside);
        TraceSpillLease::collect_stale(root);
        expect(
            std::filesystem::exists(hardlink_lease),
            "GC removed a lease containing a hardlinked payload");
        std::filesystem::remove(hardlink_outside, cleanup_error);
        TraceSpillLease::collect_stale(root);
        expect(
            !std::filesystem::exists(hardlink_lease),
            "authenticated hardlink lease was not collected after repair");

        const auto staging_name =
            std::string(".staging-proof-") +
            "0123456789abcdef0123456789abcdef-999999999-1";
        const auto stale_staging = root / staging_name;
        std::filesystem::create_directory(stale_staging);
        expect(::chmod(stale_staging.c_str(), 0700) == 0,
               "failed to secure stale staging fixture");
        TraceSpillLease::collect_stale(root);
        expect(
            !std::filesystem::exists(stale_staging),
            "authenticated stale staging directory was not collected");

        const auto before_replacement = proof_directories(root);
        const auto replacement_child =
            fork_crashed_creator(root, "replacement.bin");
        wait_success(
            replacement_child,
            "metadata replacement lease creator failed");
        std::filesystem::path replacement_lease;
        for (const auto &candidate : proof_directories(root))
        {
            if (!before_replacement.contains(candidate) &&
                candidate != active_path)
            {
                replacement_lease = candidate;
            }
        }
        expect(!replacement_lease.empty(), "replacement lease was not created");
        const auto metadata_backup = replacement_lease / ".lease.meta.backup";
        std::filesystem::rename(
            replacement_lease / ".lease.meta",
            metadata_backup);
        TraceSpillLease::collect_stale(root);
        expect(
            std::filesystem::exists(replacement_lease),
            "GC modified a lease while metadata was concurrently replaced");
        std::filesystem::rename(
            metadata_backup,
            replacement_lease / ".lease.meta");
        const auto lock_hardlink =
            root.parent_path() / "trace_spill_lock_hardlink";
        std::filesystem::remove(lock_hardlink, cleanup_error);
        std::filesystem::create_hard_link(
            replacement_lease / ".lease.lock",
            lock_hardlink);
        TraceSpillLease::collect_stale(root);
        expect(
            std::filesystem::exists(replacement_lease),
            "GC accepted a hardlinked lease lock");
        std::filesystem::remove(lock_hardlink, cleanup_error);
        TraceSpillLease::collect_stale(root);
        expect(
            !std::filesystem::exists(replacement_lease),
            "repaired replacement lease was not collected");

        const auto root_race_child =
            fork_crashed_creator(root, "root-race.bin");
        wait_success(
            root_race_child,
            "root entry race lease creator failed");
        std::filesystem::path root_race_lease;
        for (const auto &candidate : proof_directories(root))
        {
            if (candidate != active_path &&
                candidate != replacement_lease)
            {
                root_race_lease = candidate;
            }
        }
        expect(!root_race_lease.empty(), "root race lease was not created");
        const auto root_race_alias = root / "root-race-alias";
        std::filesystem::rename(root_race_lease, root_race_alias);
        std::filesystem::create_directory_symlink(
            root_race_alias,
            root_race_lease);
        TraceSpillLease::collect_stale(root);
        expect(
            std::filesystem::exists(root_race_alias),
            "GC followed a symlinked root lease entry");
        std::filesystem::remove(root_race_lease, cleanup_error);
        std::filesystem::rename(root_race_alias, root_race_lease);
        TraceSpillLease::collect_stale(root);
        expect(
            !std::filesystem::exists(root_race_lease),
            "restored root race lease was not collected");

        const auto legacy = root /
            "proof-0123456789abcdef0123456789abcdef";
        std::filesystem::create_directory(legacy);
        {
            std::ofstream stream(legacy / "legacy.bin");
            stream << "legacy";
        }
        const auto legacy_before = mode_of(legacy / "legacy.bin");
        TraceSpillLease::collect_stale(root);
        expect(
            std::filesystem::exists(legacy),
            "legacy spill directory was collected");
        expect(
            mode_of(legacy / "legacy.bin") == legacy_before,
            "legacy payload was modified by GC");

        const auto invalid_child =
            fork_crashed_creator(root, "invalid.bin");
        wait_success(
            invalid_child,
            "invalid-metadata creator failed");
        const auto candidates = proof_directories(root);
        std::filesystem::path invalid_lease;
        for (const auto &candidate : candidates)
        {
            if (candidate != active_path && candidate != legacy)
            {
                invalid_lease = candidate;
            }
        }
        expect(!invalid_lease.empty(), "invalid metadata lease was not created");
        expect(
            ::chmod(
                (invalid_lease / ".lease.meta").c_str(),
                0644) == 0,
            "failed to make metadata unverifiable");
        TraceSpillLease::collect_stale(root);
        expect(
            std::filesystem::exists(invalid_lease),
            "unverifiable metadata lease was collected");

        constexpr int kConcurrentCreators = 4;
        std::vector<pid_t> creators;
        creators.reserve(kConcurrentCreators);
        for (int index = 0; index < kConcurrentCreators; ++index)
        {
            creators.push_back(
                fork_crashed_creator(
                    root,
                    "concurrent-" + std::to_string(index) + ".bin"));
        }
        for (const auto child : creators)
        {
            wait_success(child, "concurrent trace spill creator failed");
        }
        TraceSpillLease::collect_stale(root);
        const auto remaining = proof_directories(root);
        expect(
            remaining.count(active_path) == 1 &&
                remaining.count(legacy) == 1 &&
                remaining.count(invalid_lease) == 1,
            "concurrent GC removed an active, legacy, or unverifiable lease");

        std::filesystem::path normal_path;
        {
            const auto normal = TraceSpillLease::create(root);
            normal_path = normal->path();
            write_payload(normal, "normal.bin");
        }
        expect(
            !std::filesystem::exists(normal_path),
            "normal lease destruction did not remove its payload directory");

        active.reset();
        std::filesystem::remove_all(root, cleanup_error);
        std::filesystem::remove(outside, cleanup_error);
        std::cout << "trace_spill_lease_runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::filesystem::remove_all(root, cleanup_error);
        std::cerr << "trace_spill_lease_runtime: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
