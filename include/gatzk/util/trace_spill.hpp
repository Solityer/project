#pragma once

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

        static std::shared_ptr<const TraceSpillLease> create(const std::filesystem::path &root);

        static void collect_stale(const std::filesystem::path &root);

        TraceSpillLease(const TraceSpillLease &) = delete;
        TraceSpillLease &operator=(const TraceSpillLease &) = delete;

        ~TraceSpillLease() noexcept;

        const std::filesystem::path &path() const noexcept { return path_; }

        std::string payload_name(const std::filesystem::path &payload_path) const;

        std::unique_ptr<std::ostream> create_payload_stream(std::string_view name) const;

        std::unique_ptr<std::istream> open_payload_stream(std::string_view name) const;

        std::uintmax_t payload_size(std::string_view name) const;

        void remove_payload(std::string_view name) const noexcept;

    private:
        TraceSpillLease(int root_fd, int lease_fd, int lock_fd, std::filesystem::path root, std::filesystem::path path, std::string name, std::uint64_t device,
                        std::uint64_t inode, std::uint64_t creator_pid, std::uint64_t creator_start_ticks, std::string creator_nonce,
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
        struct stat metadata_status_
        {
        };
    };

}
