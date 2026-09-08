#pragma once

// Crash-safe, private execution-only storage for large trace evaluation
// columns.  The lease is deliberately independent of the proof schema: it
// only owns a directory, its lock and the temporary payload files.

#include <cstdint>
#include <filesystem>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <sys/stat.h>

namespace gatzk::util
{

    class TraceSpillLease
    {
    public:
        static std::filesystem::path default_root();

        // Create a new proof-* lease below root.  Creation and stale-lease
        // collection are serialized by root/.gc.lock.  The returned object
        // keeps the per-lease lock held until its last owner is destroyed.
        static std::shared_ptr<const TraceSpillLease> create(
            const std::filesystem::path &root);

        // Collect only authenticated, inactive leases created by this
        // implementation.  Legacy directories and any unverifiable entry are
        // intentionally left untouched.
        static void collect_stale(
            const std::filesystem::path &root);

        TraceSpillLease(const TraceSpillLease &) = delete;
        TraceSpillLease &operator=(const TraceSpillLease &) = delete;

        ~TraceSpillLease() noexcept;

        const std::filesystem::path &path() const noexcept
        {
            return path_;
        }

        // Validate that payload_path is exactly one payload beneath this
        // lease, then return its safe basename.  This rejects traversal,
        // symlink aliases and paths outside the authenticated lease.
        std::string payload_name(
            const std::filesystem::path &payload_path) const;

        // Create an immutable payload file with O_EXCL/O_NOFOLLOW and mode
        // 0600, then expose it as a stream.  The caller must flush and check
        // the stream before returning the descriptor to the protocol.
        std::unique_ptr<std::ostream> create_payload_stream(
            std::string_view name) const;

        // Open an existing payload through the lease directory FD.  No path
        // component is followed by the open operation; the returned stream
        // supports the seek operations used by direct-domain openings.
        std::unique_ptr<std::istream> open_payload_stream(
            std::string_view name) const;

        // Return the authenticated payload size from fstat(2).
        std::uintmax_t payload_size(
            std::string_view name) const;

        // Remove one failed/incomplete payload through the lease directory
        // FD.  This is intentionally basename-only and never follows a path.
        void remove_payload(
            std::string_view name) const noexcept;

    private:
        TraceSpillLease(
            int root_fd,
            int lease_fd,
            int lock_fd,
            std::filesystem::path root,
            std::filesystem::path path,
            std::string name,
            std::uint64_t device,
            std::uint64_t inode,
            std::uint64_t creator_pid,
            std::uint64_t creator_start_ticks,
            std::string creator_nonce,
            struct stat metadata_status) noexcept;

        int root_fd_ = -1;
        int lease_fd_ = -1;
        int lock_fd_ = -1;
        std::filesystem::path root_;
        std::filesystem::path path_;
        std::string name_;
        std::uint64_t device_ = 0;
        std::uint64_t inode_ = 0;
        std::uint64_t creator_pid_ = 0;
        std::uint64_t creator_start_ticks_ = 0;
        std::string creator_nonce_;
        struct stat metadata_status_{};
    };

} // namespace gatzk::util
