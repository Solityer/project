#include "gatzk/util/trace_spill.hpp"

#include <array>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <ext/stdio_filebuf.h>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <string_view>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <openssl/rand.h>

namespace gatzk::util
{
    namespace
    {

        constexpr mode_t kPrivateDirectoryMode = S_IRWXU;
        constexpr mode_t kPrivateFileMode = S_IRUSR | S_IWUSR;
        constexpr char kRootLockName[] = ".gc.lock";
        constexpr char kMetadataName[] = ".lease.meta";
        constexpr char kLeaseLockName[] = ".lease.lock";
        constexpr char kStagingPrefix[] = ".staging-proof-";
        constexpr std::size_t kMetadataLimit = 1024;

        [[noreturn]] void throw_errno(
            const char *operation)
        {
            throw std::system_error(
                errno,
                std::generic_category(),
                operation);
        }

        void close_noexcept(int *fd) noexcept
        {
            if (fd != nullptr && *fd >= 0)
            {
                (void)::close(*fd);
                *fd = -1;
            }
        }

        bool owner_and_mode(
            const struct stat &status,
            mode_t mode)
        {
            return status.st_uid == ::geteuid() &&
                   (status.st_mode & 0777) == mode;
        }

        bool is_directory_component(
            const std::filesystem::path &component)
        {
            const auto value = component.string();
            return !value.empty() && value != "." && value != ".." &&
                   value.find('/') == std::string::npos;
        }

        std::filesystem::path normalized_absolute_root(
            const std::filesystem::path &root)
        {
            if (root.empty())
            {
                throw std::runtime_error(
                    "trace spill root is empty");
            }

            std::error_code absolute_error;
            const auto absolute =
                std::filesystem::absolute(root, absolute_error);
            if (absolute_error)
            {
                throw std::runtime_error(
                    "failed to resolve trace spill root: " +
                    absolute_error.message());
            }

            const auto normalized = absolute.lexically_normal();
            if (!normalized.is_absolute() || normalized == normalized.root_path())
            {
                throw std::runtime_error(
                    "trace spill root must be a non-root absolute directory");
            }

            for (const auto &component : normalized.relative_path())
            {
                if (!is_directory_component(component))
                {
                    throw std::runtime_error(
                        "trace spill root contains an unsafe path component");
                }
            }
            return normalized;
        }

        int open_private_directory_tree(
            const std::filesystem::path &root)
        {
            const auto normalized =
                normalized_absolute_root(root);
            int current = ::open(
                normalized.root_path().c_str(),
                O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            if (current < 0)
            {
                throw_errno("failed to open trace spill filesystem root");
            }

            try
            {
                const auto relative = normalized.relative_path();
                for (auto component_it = relative.begin();
                     component_it != relative.end();
                     ++component_it)
                {
                    const auto &component = *component_it;
                    const auto name = component.string();
                    int next = ::openat(
                        current,
                        name.c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
                    if (next < 0 && errno == ENOENT)
                    {
                        if (::mkdirat(
                                current,
                                name.c_str(),
                                kPrivateDirectoryMode) != 0 &&
                            errno != EEXIST)
                        {
                            throw_errno(
                                "failed to create trace spill directory component");
                        }
                        next = ::openat(
                            current,
                            name.c_str(),
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
                    }
                    if (next < 0)
                    {
                        throw_errno(
                            "failed to open trace spill directory component");
                    }

                    struct stat status{};
                    if (::fstat(next, &status) != 0 ||
                        !S_ISDIR(status.st_mode))
                    {
                        const auto saved_errno =
                            errno != 0 ? errno : ENOTDIR;
                        (void)::close(next);
                        errno = saved_errno;
                        throw_errno(
                            "trace spill path component is not a directory");
                    }

                    const bool is_final =
                        std::next(component_it) == relative.end();
                    if (is_final)
                    {
                        if (::fchmod(next, kPrivateDirectoryMode) != 0)
                        {
                            (void)::close(next);
                            throw_errno(
                                "failed to secure trace spill root permissions");
                        }
                        if (::fstat(next, &status) != 0 ||
                            !owner_and_mode(status, kPrivateDirectoryMode))
                        {
                            (void)::close(next);
                            throw_errno(
                                "trace spill root ownership or permissions are invalid");
                        }
                    }

                    (void)::close(current);
                    current = next;
                }
                return current;
            }
            catch (...)
            {
                (void)::close(current);
                throw;
            }
        }

        int duplicate_cloexec(int fd)
        {
            const auto duplicate =
                ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
            if (duplicate < 0)
            {
                throw_errno("failed to duplicate trace spill directory FD");
            }
            return duplicate;
        }

        std::string read_fd_bounded(
            int fd,
            std::size_t limit,
            const char *operation)
        {
            std::string result;
            result.reserve(limit);
            std::array<char, 256> buffer{};
            for (;;)
            {
                const auto count =
                    ::read(fd, buffer.data(), buffer.size());
                if (count < 0)
                {
                    throw_errno(operation);
                }
                if (count == 0)
                {
                    break;
                }
                if (result.size() > limit -
                                        static_cast<std::size_t>(count))
                {
                    throw std::runtime_error(
                        std::string(operation) +
                        " exceeds the metadata size limit");
                }
                result.append(buffer.data(), static_cast<std::size_t>(count));
            }
            return result;
        }

        void write_all(
            int fd,
            std::string_view bytes,
            const char *operation)
        {
            std::size_t offset = 0;
            while (offset < bytes.size())
            {
                const auto written = ::write(
                    fd,
                    bytes.data() + offset,
                    bytes.size() - offset);
                if (written < 0)
                {
                    throw_errno(operation);
                }
                if (written == 0)
                {
                    throw std::runtime_error(
                        std::string(operation) + " made no progress");
                }
                offset += static_cast<std::size_t>(written);
            }
        }

        bool hex_lower(
            char value)
        {
            return (value >= '0' && value <= '9') ||
                   (value >= 'a' && value <= 'f');
        }

        bool valid_lease_name(
            std::string_view value)
        {
            if (value.size() != 38 ||
                value.substr(0, 6) != "proof-")
            {
                return false;
            }
            for (std::size_t i = 6; i < value.size(); ++i)
            {
                if (!hex_lower(value[i]))
                {
                    return false;
                }
            }
            return true;
        }

        std::uint64_t parse_u64(
            std::string_view value,
            const char *field);

        struct StagingIdentity
        {
            std::string final_name;
            std::uint64_t pid = 0;
            std::uint64_t start_ticks = 0;
        };

        std::optional<StagingIdentity> parse_staging_name(
            std::string_view value)
        {
            if (value.substr(0, sizeof(kStagingPrefix) - 1) !=
                kStagingPrefix)
            {
                return std::nullopt;
            }
            const auto suffix = value.substr(sizeof(kStagingPrefix) - 1);
            if (suffix.size() <= 33 || suffix[32] != '-')
            {
                return std::nullopt;
            }
            const auto nonce = suffix.substr(0, 32);
            if (nonce.size() != 32)
            {
                return std::nullopt;
            }
            for (const auto ch : nonce)
            {
                if (!hex_lower(ch))
                {
                    return std::nullopt;
                }
            }
            const auto pid_separator = suffix.find('-', 33);
            if (pid_separator == std::string_view::npos ||
                pid_separator == 33 ||
                pid_separator + 1 >= suffix.size())
            {
                return std::nullopt;
            }
            try
            {
                const auto pid = parse_u64(
                    suffix.substr(33, pid_separator - 33),
                    "staging pid");
                const auto start_ticks = parse_u64(
                    suffix.substr(pid_separator + 1),
                    "staging start ticks");
                if (pid == 0 || pid > static_cast<std::uint64_t>(INT_MAX) ||
                    start_ticks == 0)
                {
                    return std::nullopt;
                }
                return StagingIdentity{
                    .final_name = std::string("proof-") + std::string(nonce),
                    .pid = pid,
                    .start_ticks = start_ticks,
                };
            }
            catch (const std::exception &)
            {
                return std::nullopt;
            }
        }

        bool valid_payload_name(
            std::string_view value)
        {
            if (value.empty() || value.size() > 255 ||
                value == kMetadataName || value == kLeaseLockName ||
                value == kRootLockName ||
                value.find('/') != std::string_view::npos ||
                value.find('\\') != std::string_view::npos ||
                value == "." || value == "..")
            {
                return false;
            }
            if (value.size() < 4 || value.substr(value.size() - 4) != ".bin")
            {
                return false;
            }
            for (const auto ch : value.substr(0, value.size() - 4))
            {
                if (!(std::isalnum(static_cast<unsigned char>(ch)) ||
                      ch == '_' || ch == '-'))
                {
                    return false;
                }
            }
            return true;
        }

        std::uint64_t parse_u64(
            std::string_view value,
            const char *field)
        {
            if (value.empty())
            {
                throw std::runtime_error(
                    std::string("trace spill metadata has empty ") + field);
            }
            std::uint64_t result = 0;
            for (const auto ch : value)
            {
                if (ch < '0' || ch > '9')
                {
                    throw std::runtime_error(
                        std::string("trace spill metadata has invalid ") + field);
                }
                const auto digit =
                    static_cast<std::uint64_t>(ch - '0');
                if (result >
                    (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
                {
                    throw std::runtime_error(
                        std::string("trace spill metadata ") + field +
                        " overflows");
                }
                result = result * 10 + digit;
            }
            return result;
        }

        struct LeaseMetadata
        {
            std::string name;
            std::uint64_t pid = 0;
            std::uint64_t start_ticks = 0;
            std::string nonce;
        };

        LeaseMetadata parse_metadata(
            std::string_view text,
            std::string_view expected_name)
        {
            std::vector<std::string_view> lines;
            std::size_t begin = 0;
            while (begin <= text.size())
            {
                const auto end = text.find('\n', begin);
                const auto line_end =
                    end == std::string_view::npos ? text.size() : end;
                lines.push_back(text.substr(begin, line_end - begin));
                if (end == std::string_view::npos)
                {
                    break;
                }
                begin = end + 1;
            }
            if (lines.size() != 6 || !lines.back().empty())
            {
                throw std::runtime_error(
                    "trace spill metadata line count is invalid");
            }

            const auto expect_field =
                [&](std::string_view line,
                    std::string_view prefix)
                -> std::string_view
            {
                if (line.substr(0, prefix.size()) != prefix)
                {
                    throw std::runtime_error(
                        "trace spill metadata field order is invalid");
                }
                return line.substr(prefix.size());
            };

            if (expect_field(lines[0], "version=") != "1")
            {
                throw std::runtime_error(
                    "trace spill metadata version is unsupported");
            }
            const auto name = expect_field(lines[1], "name=");
            if (name != expected_name || !valid_lease_name(name))
            {
                throw std::runtime_error(
                    "trace spill metadata lease name is invalid");
            }
            const auto pid = parse_u64(
                expect_field(lines[2], "pid="),
                "pid");
            const auto start_ticks = parse_u64(
                expect_field(lines[3], "start_ticks="),
                "start_ticks");
            const auto nonce = expect_field(lines[4], "nonce=");
            if (pid == 0 || pid > static_cast<std::uint64_t>(INT_MAX) ||
                start_ticks == 0 ||
                nonce.size() != 32 ||
                nonce != name.substr(6))
            {
                throw std::runtime_error(
                    "trace spill metadata numeric fields are invalid");
            }
            for (const auto ch : nonce)
            {
                if (!hex_lower(ch))
                {
                    throw std::runtime_error(
                        "trace spill metadata nonce is invalid");
                }
            }

            return LeaseMetadata{
                .name = std::string(name),
                .pid = pid,
                .start_ticks = start_ticks,
                .nonce = std::string(nonce),
            };
        }

        bool same_metadata_identity(
            const LeaseMetadata &a,
            const LeaseMetadata &b)
        {
            return a.name == b.name &&
                   a.pid == b.pid &&
                   a.start_ticks == b.start_ticks &&
                   a.nonce == b.nonce;
        }

        struct ProcStartResult
        {
            bool exists = false;
            bool readable = false;
            std::uint64_t start_ticks = 0;
        };

        ProcStartResult process_start_ticks(
            std::uint64_t pid)
        {
            if (pid == 0 || pid > static_cast<std::uint64_t>(INT_MAX))
            {
                return {};
            }

            const auto proc_path =
                std::string("/proc/") + std::to_string(pid) + "/stat";
            const auto fd = ::open(
                proc_path.c_str(),
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            if (fd < 0)
            {
                if (errno == ENOENT || errno == ESRCH)
                {
                    errno = 0;
                    if (::kill(
                            static_cast<pid_t>(pid),
                            0) == 0 ||
                        errno == EPERM)
                    {
                        return ProcStartResult{
                            .exists = true,
                            .readable = false,
                        };
                    }
                    if (errno == ESRCH)
                    {
                        return ProcStartResult{.exists = false};
                    }
                }
                return ProcStartResult{.exists = true, .readable = false};
            }

            std::string text;
            try
            {
                text = read_fd_bounded(
                    fd,
                    8192,
                    "failed to read process identity");
            }
            catch (...)
            {
                (void)::close(fd);
                return ProcStartResult{.exists = true, .readable = false};
            }
            (void)::close(fd);

            const auto close_paren = text.rfind(')');
            if (close_paren == std::string::npos ||
                close_paren + 2 >= text.size())
            {
                return ProcStartResult{.exists = true, .readable = false};
            }

            std::istringstream fields(
                text.substr(close_paren + 2));
            std::string token;
            // The first token after the command is field 3 (state); field 22
            // is therefore the twentieth token in this suffix.
            for (int field = 3; field <= 22; ++field)
            {
                if (!(fields >> token))
                {
                    return ProcStartResult{.exists = true, .readable = false};
                }
                if (field == 22)
                {
                    try
                    {
                        const auto start = parse_u64(token, "process start");
                        return ProcStartResult{
                            .exists = true,
                            .readable = true,
                            .start_ticks = start,
                        };
                    }
                    catch (const std::exception &)
                    {
                        return ProcStartResult{.exists = true, .readable = false};
                    }
                }
            }
            return ProcStartResult{.exists = true, .readable = false};
        }

        std::uint64_t current_process_start_ticks()
        {
            const auto result = process_start_ticks(
                static_cast<std::uint64_t>(::getpid()));
            if (!result.exists || !result.readable || result.start_ticks == 0)
            {
                throw std::runtime_error(
                    "cannot authenticate current process identity for trace spill lease");
            }
            return result.start_ticks;
        }

        int open_root_lock(int root_fd)
        {
            const auto fd = ::openat(
                root_fd,
                kRootLockName,
                O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
                kPrivateFileMode);
            if (fd < 0)
            {
                throw_errno("failed to open trace spill root lock");
            }
            struct stat status{};
            if (::fstat(fd, &status) != 0 ||
                !S_ISREG(status.st_mode) ||
                !owner_and_mode(status, kPrivateFileMode) ||
                status.st_nlink != 1)
            {
                const auto saved_errno = errno != 0 ? errno : EACCES;
                (void)::close(fd);
                errno = saved_errno;
                throw_errno("trace spill root lock permissions are invalid");
            }
            if (::flock(fd, LOCK_EX) != 0)
            {
                const auto saved_errno = errno;
                (void)::close(fd);
                errno = saved_errno;
                throw_errno("failed to acquire trace spill root lock");
            }
            return fd;
        }

        void unlock_close(int *fd) noexcept
        {
            if (fd != nullptr && *fd >= 0)
            {
                (void)::flock(*fd, LOCK_UN);
                (void)::close(*fd);
                *fd = -1;
            }
        }

        bool same_directory(
            const struct stat &a,
            const struct stat &b)
        {
            return a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
                   S_ISDIR(a.st_mode) && S_ISDIR(b.st_mode);
        }

        bool same_regular_file(
            const struct stat &a,
            const struct stat &b)
        {
            return a.st_dev == b.st_dev &&
                   a.st_ino == b.st_ino &&
                   a.st_uid == b.st_uid &&
                   a.st_gid == b.st_gid &&
                   a.st_mode == b.st_mode &&
                   a.st_nlink == b.st_nlink &&
                   a.st_size == b.st_size &&
                   S_ISREG(a.st_mode);
        }

        bool valid_private_regular(
            const struct stat &status)
        {
            return S_ISREG(status.st_mode) &&
                   owner_and_mode(status, kPrivateFileMode) &&
                   status.st_nlink == 1;
        }

        std::vector<std::string> lease_entries(
            int lease_fd)
        {
            const auto duplicate = duplicate_cloexec(lease_fd);
            if (::lseek(duplicate, 0, SEEK_SET) < 0)
            {
                const auto saved_errno = errno;
                (void)::close(duplicate);
                errno = saved_errno;
                throw_errno("failed to rewind trace spill lease scan");
            }
            DIR *directory = ::fdopendir(duplicate);
            if (directory == nullptr)
            {
                const auto saved_errno = errno;
                (void)::close(duplicate);
                errno = saved_errno;
                throw_errno("failed to scan trace spill lease");
            }

            std::vector<std::string> entries;
            errno = 0;
            while (auto *entry = ::readdir(directory))
            {
                const std::string name(entry->d_name);
                if (name == "." || name == "..")
                {
                    continue;
                }
                entries.push_back(name);
            }
            const auto scan_error = errno;
            (void)::closedir(directory);
            if (scan_error != 0)
            {
                errno = scan_error;
                throw_errno("failed to scan trace spill lease entries");
            }
            return entries;
        }

        struct MetadataHandle
        {
            int fd = -1;
            LeaseMetadata metadata;
            struct stat status{};
        };

        std::optional<MetadataHandle> open_metadata_handle(
            int lease_fd,
            std::string_view expected_name)
        {
            const auto metadata_fd = ::openat(
                lease_fd,
                kMetadataName,
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            if (metadata_fd < 0)
            {
                return std::nullopt;
            }
            struct stat status{};
            if (::fstat(metadata_fd, &status) != 0 ||
                !valid_private_regular(status) ||
                status.st_size < 0 ||
                static_cast<std::uintmax_t>(status.st_size) > kMetadataLimit)
            {
                (void)::close(metadata_fd);
                return std::nullopt;
            }
            try
            {
                const auto text = read_fd_bounded(
                    metadata_fd,
                    kMetadataLimit,
                    "failed to read trace spill metadata");
                return MetadataHandle{
                    .fd = metadata_fd,
                    .metadata = parse_metadata(text, expected_name),
                    .status = status,
                };
            }
            catch (const std::exception &)
            {
                (void)::close(metadata_fd);
                return std::nullopt;
            }
        }

        bool safe_remove_lease(
            int root_fd,
            std::string_view name,
            int lease_fd,
            int lock_fd,
            const struct stat &expected_directory,
            const LeaseMetadata &expected_metadata,
            const struct stat &expected_metadata_status)
        {
            if (!valid_lease_name(name))
            {
                return false;
            }

            struct stat current_directory{};
            if (::fstatat(
                    root_fd,
                    std::string(name).c_str(),
                    &current_directory,
                    AT_SYMLINK_NOFOLLOW) != 0 ||
                !same_directory(expected_directory, current_directory))
            {
                return false;
            }

            auto metadata = open_metadata_handle(lease_fd, name);
            if (!metadata.has_value())
            {
                return false;
            }
            const auto close_metadata = [&]() noexcept
            {
                int fd = metadata->fd;
                close_noexcept(&fd);
                metadata->fd = -1;
            };

            // The metadata used to decide that this lease is stale is an
            // authenticated identity, not merely a hint.  Re-opened content
            // and inode identity must both match it before anything can be
            // unlinked.  This closes the replacement window between the
            // stale-process check and the destructive phase.
            if (!same_metadata_identity(
                    metadata->metadata,
                    expected_metadata) ||
                !same_regular_file(
                    metadata->status,
                    expected_metadata_status))
            {
                close_metadata();
                return false;
            }

            std::vector<std::string> entries;
            try
            {
                entries = lease_entries(lease_fd);
            }
            catch (const std::exception &)
            {
                return false;
            }

            struct EntrySnapshot
            {
                std::string name;
                struct stat status{};
            };
            std::vector<EntrySnapshot> snapshots;
            snapshots.reserve(entries.size());
            bool saw_metadata = false;
            bool saw_lock = false;
            for (const auto &entry : entries)
            {
                if (entry != kMetadataName && entry != kLeaseLockName &&
                    !valid_payload_name(entry))
                {
                    close_metadata();
                    return false;
                }
                struct stat status{};
                if (::fstatat(
                        lease_fd,
                        entry.c_str(),
                        &status,
                        AT_SYMLINK_NOFOLLOW) != 0 ||
                    !valid_private_regular(status))
                {
                    close_metadata();
                    return false;
                }
                saw_metadata = saw_metadata || entry == kMetadataName;
                saw_lock = saw_lock || entry == kLeaseLockName;
                snapshots.push_back(
                    EntrySnapshot{
                        .name = entry,
                        .status = status,
                    });
            }
            if (!saw_metadata || !saw_lock)
            {
                close_metadata();
                return false;
            }

            struct stat lock_status{};
            if (lock_fd < 0 ||
                ::fstat(lock_fd, &lock_status) != 0 ||
                !valid_private_regular(lock_status))
            {
                close_metadata();
                return false;
            }
            const auto metadata_it = std::find_if(
                snapshots.begin(),
                snapshots.end(),
                [](const auto &item)
                {
                    return item.name == kMetadataName;
                });
            const auto lock_it = std::find_if(
                snapshots.begin(),
                snapshots.end(),
                [](const auto &item)
                {
                    return item.name == kLeaseLockName;
                });
            if (metadata_it == snapshots.end() ||
                lock_it == snapshots.end() ||
                !same_regular_file(metadata->status, metadata_it->status) ||
                !same_regular_file(lock_status, lock_it->status))
            {
                close_metadata();
                return false;
            }

            // Re-scan and compare the complete directory view immediately
            // before any unlink.  A replacement of metadata, lock, payload,
            // or a root entry therefore fails closed without changing the
            // unverifiable directory.
            std::vector<std::string> second_entries;
            try
            {
                second_entries = lease_entries(lease_fd);
            }
            catch (const std::exception &)
            {
                close_metadata();
                return false;
            }
            auto sort_entries = [](auto &items)
            {
                std::sort(items.begin(), items.end());
            };
            sort_entries(entries);
            sort_entries(second_entries);
            if (entries != second_entries)
            {
                close_metadata();
                return false;
            }
            for (const auto &snapshot : snapshots)
            {
                struct stat status{};
                if (::fstatat(
                        lease_fd,
                        snapshot.name.c_str(),
                        &status,
                        AT_SYMLINK_NOFOLLOW) != 0 ||
                    !same_regular_file(snapshot.status, status))
                {
                    close_metadata();
                    return false;
                }
            }

            struct stat metadata_after_read{};
            if (::fstat(metadata->fd, &metadata_after_read) != 0 ||
                !same_regular_file(metadata->status, metadata_after_read) ||
                !same_regular_file(
                    expected_metadata_status,
                    metadata_after_read))
            {
                close_metadata();
                return false;
            }

            // Read and parse the bytes again immediately before the final
            // unlink.  A same-UID writer can change file contents without
            // changing its stat identity; such a replacement is therefore
            // fail-closed even when all directory entries still match.
            LeaseMetadata metadata_before_unlink;
            try
            {
                if (::lseek(metadata->fd, 0, SEEK_SET) < 0)
                {
                    close_metadata();
                    return false;
                }
                metadata_before_unlink = parse_metadata(
                    read_fd_bounded(
                        metadata->fd,
                        kMetadataLimit,
                        "failed to reread trace spill metadata"),
                    name);
            }
            catch (const std::exception &)
            {
                close_metadata();
                return false;
            }
            if (!same_metadata_identity(
                    metadata_before_unlink,
                    expected_metadata))
            {
                close_metadata();
                return false;
            }

            if (::fstatat(
                    root_fd,
                    std::string(name).c_str(),
                    &current_directory,
                    AT_SYMLINK_NOFOLLOW) != 0 ||
                !same_directory(expected_directory, current_directory))
            {
                close_metadata();
                return false;
            }

            for (const auto &entry : entries)
            {
                if (::unlinkat(lease_fd, entry.c_str(), 0) != 0)
                {
                    close_metadata();
                    return false;
                }
            }
            close_metadata();
            if (::fstatat(
                    root_fd,
                    std::string(name).c_str(),
                    &current_directory,
                    AT_SYMLINK_NOFOLLOW) != 0 ||
                !same_directory(expected_directory, current_directory))
            {
                return false;
            }
            return ::unlinkat(
                       root_fd,
                       std::string(name).c_str(),
                       AT_REMOVEDIR) == 0;
        }

        bool safe_remove_staging(
            int root_fd,
            std::string_view name,
            int staging_fd,
            int lock_fd,
            const struct stat &expected_directory,
            const StagingIdentity &identity)
        {
            struct stat current_directory{};
            if (::fstatat(
                    root_fd,
                    std::string(name).c_str(),
                    &current_directory,
                    AT_SYMLINK_NOFOLLOW) != 0 ||
                !same_directory(expected_directory, current_directory))
            {
                return false;
            }

            std::optional<MetadataHandle> metadata;
            const auto metadata_probe = ::openat(
                staging_fd,
                kMetadataName,
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            if (metadata_probe >= 0)
            {
                (void)::close(metadata_probe);
                metadata = open_metadata_handle(
                    staging_fd,
                    identity.final_name);
                if (!metadata.has_value() ||
                    metadata->metadata.pid != identity.pid ||
                    metadata->metadata.start_ticks != identity.start_ticks)
                {
                    if (metadata.has_value())
                    {
                        int fd = metadata->fd;
                        close_noexcept(&fd);
                    }
                    return false;
                }
            }

            std::vector<std::string> entries;
            try
            {
                entries = lease_entries(staging_fd);
            }
            catch (const std::exception &)
            {
                if (metadata.has_value())
                {
                    int fd = metadata->fd;
                    close_noexcept(&fd);
                }
                return false;
            }

            struct EntrySnapshot
            {
                std::string name;
                struct stat status{};
            };
            std::vector<EntrySnapshot> snapshots;
            snapshots.reserve(entries.size());
            bool saw_lock = false;
            bool saw_metadata = false;
            for (const auto &entry : entries)
            {
                if (entry != kMetadataName && entry != kLeaseLockName)
                {
                    if (metadata.has_value())
                    {
                        int fd = metadata->fd;
                        close_noexcept(&fd);
                    }
                    return false;
                }
                struct stat status{};
                if (::fstatat(
                        staging_fd,
                        entry.c_str(),
                        &status,
                        AT_SYMLINK_NOFOLLOW) != 0 ||
                    !valid_private_regular(status))
                {
                    if (metadata.has_value())
                    {
                        int fd = metadata->fd;
                        close_noexcept(&fd);
                    }
                    return false;
                }
                saw_lock = saw_lock || entry == kLeaseLockName;
                saw_metadata = saw_metadata || entry == kMetadataName;
                snapshots.push_back(
                    EntrySnapshot{
                        .name = entry,
                        .status = status,
                    });
            }
            if (saw_metadata != metadata.has_value() ||
                (saw_lock && lock_fd < 0) ||
                (!saw_lock && !entries.empty()))
            {
                if (metadata.has_value())
                {
                    int fd = metadata->fd;
                    close_noexcept(&fd);
                }
                return false;
            }

            struct stat lock_status{};
            if (saw_lock &&
                (::fstat(lock_fd, &lock_status) != 0 ||
                 !valid_private_regular(lock_status)))
            {
                if (metadata.has_value())
                {
                    int fd = metadata->fd;
                    close_noexcept(&fd);
                }
                return false;
            }
            for (const auto &snapshot : snapshots)
            {
                if (snapshot.name == kLeaseLockName &&
                    !same_regular_file(snapshot.status, lock_status))
                {
                    if (metadata.has_value())
                    {
                        int fd = metadata->fd;
                        close_noexcept(&fd);
                    }
                    return false;
                }
                if (snapshot.name == kMetadataName &&
                    !metadata.has_value())
                {
                    return false;
                }
                if (snapshot.name == kMetadataName &&
                    !same_regular_file(snapshot.status, metadata->status))
                {
                    int fd = metadata->fd;
                    close_noexcept(&fd);
                    return false;
                }
            }

            std::vector<std::string> second_entries;
            try
            {
                second_entries = lease_entries(staging_fd);
            }
            catch (const std::exception &)
            {
                if (metadata.has_value())
                {
                    int fd = metadata->fd;
                    close_noexcept(&fd);
                }
                return false;
            }
            std::sort(entries.begin(), entries.end());
            std::sort(second_entries.begin(), second_entries.end());
            if (entries != second_entries)
            {
                if (metadata.has_value())
                {
                    int fd = metadata->fd;
                    close_noexcept(&fd);
                }
                return false;
            }
            for (const auto &snapshot : snapshots)
            {
                struct stat status{};
                if (::fstatat(
                        staging_fd,
                        snapshot.name.c_str(),
                        &status,
                        AT_SYMLINK_NOFOLLOW) != 0 ||
                    !same_regular_file(snapshot.status, status))
                {
                    if (metadata.has_value())
                    {
                        int fd = metadata->fd;
                        close_noexcept(&fd);
                    }
                    return false;
                }
            }
            if (::fstatat(
                    root_fd,
                    std::string(name).c_str(),
                    &current_directory,
                    AT_SYMLINK_NOFOLLOW) != 0 ||
                !same_directory(expected_directory, current_directory))
            {
                if (metadata.has_value())
                {
                    int fd = metadata->fd;
                    close_noexcept(&fd);
                }
                return false;
            }

            for (const auto &entry : entries)
            {
                if (::unlinkat(staging_fd, entry.c_str(), 0) != 0)
                {
                    if (metadata.has_value())
                    {
                        int fd = metadata->fd;
                        close_noexcept(&fd);
                    }
                    return false;
                }
            }
            if (metadata.has_value())
            {
                int fd = metadata->fd;
                close_noexcept(&fd);
            }
            if (::fstatat(
                    root_fd,
                    std::string(name).c_str(),
                    &current_directory,
                    AT_SYMLINK_NOFOLLOW) != 0 ||
                !same_directory(expected_directory, current_directory))
            {
                return false;
            }
            return ::unlinkat(
                       root_fd,
                       std::string(name).c_str(),
                       AT_REMOVEDIR) == 0;
        }

        void remove_owned_lease_best_effort(
            int root_fd,
            const std::string &name,
            int lease_fd,
            int lock_fd,
            const struct stat &expected_directory,
            const LeaseMetadata &expected_metadata,
            const struct stat &expected_metadata_status) noexcept
        {
            try
            {
                int root_lock_fd = open_root_lock(root_fd);
                try
                {
                    (void)safe_remove_lease(
                        root_fd,
                        name,
                        lease_fd,
                        lock_fd,
                        expected_directory,
                        expected_metadata,
                        expected_metadata_status);
                }
                catch (...)
                {
                    // Destruction is best-effort and must never terminate the
                    // process on a replacement race or allocation failure.
                }
                unlock_close(&root_lock_fd);
            }
            catch (...)
            {
                return;
            }
        }

        std::optional<MetadataHandle> read_valid_metadata(
            int lease_fd,
            std::string_view expected_name)
        {
            const auto metadata_fd = ::openat(
                lease_fd,
                kMetadataName,
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            if (metadata_fd < 0)
            {
                return std::nullopt;
            }
            struct stat status{};
            if (::fstat(metadata_fd, &status) != 0 ||
                !S_ISREG(status.st_mode) ||
                !owner_and_mode(status, kPrivateFileMode) ||
                status.st_nlink != 1 ||
                status.st_size < 0 ||
                static_cast<std::uintmax_t>(status.st_size) > kMetadataLimit)
            {
                (void)::close(metadata_fd);
                return std::nullopt;
            }

            try
            {
                const auto text = read_fd_bounded(
                    metadata_fd,
                    kMetadataLimit,
                    "failed to read trace spill metadata");
                return MetadataHandle{
                    .fd = metadata_fd,
                    .metadata = parse_metadata(text, expected_name),
                    .status = status,
                };
            }
            catch (const std::exception &)
            {
                (void)::close(metadata_fd);
                return std::nullopt;
            }
        }

        void collect_stale_locked(
            int root_fd)
        {
            std::vector<std::string> candidates;
            try
            {
                candidates = lease_entries(root_fd);
            }
            catch (const std::exception &)
            {
                // A directory scan failure is fail-closed: no lease is
                // removed when the complete root view is not available.
                return;
            }

            for (const auto &name : candidates)
            {
                const auto staging = parse_staging_name(name);
                if (!valid_lease_name(name))
                {
                    if (!staging.has_value())
                    {
                        continue;
                    }
                    const auto staging_fd = ::openat(
                        root_fd,
                        name.c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
                    if (staging_fd < 0)
                    {
                        continue;
                    }
                    struct stat staging_status{};
                    if (::fstat(staging_fd, &staging_status) != 0 ||
                        !S_ISDIR(staging_status.st_mode) ||
                        !owner_and_mode(
                            staging_status,
                            kPrivateDirectoryMode))
                    {
                        (void)::close(staging_fd);
                        continue;
                    }
                    const auto process = process_start_ticks(staging->pid);
                    const bool active_or_unverifiable =
                        (process.exists && !process.readable) ||
                        (process.exists &&
                         process.start_ticks == staging->start_ticks);
                    if (active_or_unverifiable)
                    {
                        (void)::close(staging_fd);
                        continue;
                    }

                    int staging_lock_fd = ::openat(
                        staging_fd,
                        kLeaseLockName,
                        O_RDWR | O_CLOEXEC | O_NOFOLLOW);
                    if (staging_lock_fd >= 0)
                    {
                        struct stat lock_status{};
                        if (::fstat(staging_lock_fd, &lock_status) != 0 ||
                            !valid_private_regular(lock_status) ||
                            ::flock(
                                staging_lock_fd,
                                LOCK_EX | LOCK_NB) != 0)
                        {
                            (void)::close(staging_lock_fd);
                            (void)::close(staging_fd);
                            continue;
                        }
                    }
                    (void)safe_remove_staging(
                        root_fd,
                        name,
                        staging_fd,
                        staging_lock_fd,
                        staging_status,
                        *staging);
                    if (staging_lock_fd >= 0)
                    {
                        (void)::flock(staging_lock_fd, LOCK_UN);
                        (void)::close(staging_lock_fd);
                    }
                    (void)::close(staging_fd);
                    continue;
                }

                const auto lease_fd = ::openat(
                    root_fd,
                    name.c_str(),
                    O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
                if (lease_fd < 0)
                {
                    continue;
                }

                struct stat directory_status{};
                if (::fstat(lease_fd, &directory_status) != 0 ||
                    !S_ISDIR(directory_status.st_mode) ||
                    !owner_and_mode(directory_status, kPrivateDirectoryMode))
                {
                    (void)::close(lease_fd);
                    continue;
                }

                auto metadata =
                    read_valid_metadata(lease_fd, name);
                if (!metadata.has_value())
                {
                    // This is either a legacy directory or an incomplete
                    // creation.  Never infer staleness from its mtime/PID.
                    (void)::close(lease_fd);
                    continue;
                }

                const auto expected_metadata = metadata->metadata;
                const auto expected_metadata_status = metadata->status;

                const auto lock_fd = ::openat(
                    lease_fd,
                    kLeaseLockName,
                    O_RDWR | O_CLOEXEC | O_NOFOLLOW);
                if (lock_fd < 0)
                {
                    close_noexcept(&metadata->fd);
                    (void)::close(lease_fd);
                    continue;
                }
                struct stat lock_status{};
                if (::fstat(lock_fd, &lock_status) != 0 ||
                    !valid_private_regular(lock_status) ||
                    ::flock(lock_fd, LOCK_EX | LOCK_NB) != 0)
                {
                    close_noexcept(&metadata->fd);
                    (void)::close(lock_fd);
                    (void)::close(lease_fd);
                    continue;
                }

                const auto process = process_start_ticks(metadata->metadata.pid);
                const bool active_or_unverifiable =
                    (process.exists && !process.readable) ||
                    (process.exists &&
                     process.start_ticks == metadata->metadata.start_ticks);
                if (!active_or_unverifiable)
                {
                    (void)safe_remove_lease(
                        root_fd,
                        name,
                        lease_fd,
                        lock_fd,
                        directory_status,
                        expected_metadata,
                        expected_metadata_status);
                }

                close_noexcept(&metadata->fd);
                (void)::flock(lock_fd, LOCK_UN);
                (void)::close(lock_fd);
                (void)::close(lease_fd);
            }
        }

        std::string random_hex_nonce()
        {
            std::array<std::uint8_t, 16> bytes{};
            if (RAND_bytes(
                    bytes.data(),
                    static_cast<int>(bytes.size())) != 1)
            {
                throw std::runtime_error(
                    "CSPRNG failed while creating trace spill lease");
            }
            std::ostringstream output;
            for (const auto byte : bytes)
            {
                output << std::hex;
                output.width(2);
                output.fill('0');
                output << static_cast<unsigned>(byte);
            }
            return output.str();
        }

        class FdOutputStream final : public std::ostream
        {
        public:
            explicit FdOutputStream(int fd)
                : std::ostream(nullptr),
                  buffer_(std::make_unique<__gnu_cxx::stdio_filebuf<char>>(
                      fd,
                      std::ios::binary | std::ios::out | std::ios::app))
            {
                if (!buffer_->is_open())
                {
                    setstate(std::ios::badbit);
                    return;
                }
                rdbuf(buffer_.get());
            }

            bool is_open() const noexcept
            {
                return buffer_ != nullptr && buffer_->is_open();
            }

        private:
            std::unique_ptr<__gnu_cxx::stdio_filebuf<char>> buffer_;
        };

        class FdInputStream final : public std::istream
        {
        public:
            explicit FdInputStream(int fd)
                : std::istream(nullptr),
                  buffer_(std::make_unique<__gnu_cxx::stdio_filebuf<char>>(
                      fd,
                      std::ios::binary | std::ios::in))
            {
                if (!buffer_->is_open())
                {
                    setstate(std::ios::badbit);
                    return;
                }
                rdbuf(buffer_.get());
            }

            bool is_open() const noexcept
            {
                return buffer_ != nullptr && buffer_->is_open();
            }

        private:
            std::unique_ptr<__gnu_cxx::stdio_filebuf<char>> buffer_;
        };

        int create_payload_fd(
            int lease_fd,
            std::string_view name)
        {
            if (!valid_payload_name(name))
            {
                throw std::runtime_error(
                    "trace spill payload name is unsafe");
            }
            const std::string owned_name(name);
            const auto fd = ::openat(
                lease_fd,
                owned_name.c_str(),
                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                kPrivateFileMode);
            if (fd < 0)
            {
                throw_errno("failed to create trace spill payload");
            }
            struct stat status{};
            if (::fchmod(fd, kPrivateFileMode) != 0 ||
                ::fstat(fd, &status) != 0 ||
                !S_ISREG(status.st_mode) ||
                !owner_and_mode(status, kPrivateFileMode) ||
                status.st_nlink != 1)
            {
                const auto saved_errno = errno != 0 ? errno : EACCES;
                (void)::close(fd);
                (void)::unlinkat(lease_fd, owned_name.c_str(), 0);
                errno = saved_errno;
                throw_errno("trace spill payload permissions are invalid");
            }
            return fd;
        }

        int open_payload_fd(
            int lease_fd,
            std::string_view name)
        {
            if (!valid_payload_name(name))
            {
                throw std::runtime_error(
                    "trace spill payload name is unsafe");
            }
            const std::string owned_name(name);
            const auto fd = ::openat(
                lease_fd,
                owned_name.c_str(),
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            if (fd < 0)
            {
                throw_errno("failed to open trace spill payload");
            }
            struct stat status{};
            if (::fstat(fd, &status) != 0 ||
                !S_ISREG(status.st_mode) ||
                !owner_and_mode(status, kPrivateFileMode) ||
                status.st_nlink != 1)
            {
                const auto saved_errno = errno != 0 ? errno : EACCES;
                (void)::close(fd);
                errno = saved_errno;
                throw_errno("trace spill payload permissions are invalid");
            }
            return fd;
        }

    } // namespace

    std::filesystem::path TraceSpillLease::default_root()
    {
        return (std::filesystem::current_path() / "runs" / ".trace_spill")
            .lexically_normal();
    }

    std::shared_ptr<const TraceSpillLease> TraceSpillLease::create(
        const std::filesystem::path &root)
    {
        const auto normalized = normalized_absolute_root(root);
        const auto root_fd = open_private_directory_tree(normalized);
        int root_lock_fd = -1;
        try
        {
            root_lock_fd = open_root_lock(root_fd);
            const auto creator_pid =
                static_cast<std::uint64_t>(::getpid());
            const auto creator_start_ticks =
                current_process_start_ticks();

            for (std::size_t attempt = 0; attempt < 32; ++attempt)
            {
                const auto nonce = random_hex_nonce();
                const auto name = std::string("proof-") + nonce;
                const LeaseMetadata expected_metadata{
                    .name = name,
                    .pid = creator_pid,
                    .start_ticks = creator_start_ticks,
                    .nonce = nonce,
                };
                const auto staging_name =
                    std::string(kStagingPrefix) +
                    nonce + "-" +
                    std::to_string(creator_pid) + "-" +
                    std::to_string(creator_start_ticks);
                if (::mkdirat(
                        root_fd,
                        staging_name.c_str(),
                        kPrivateDirectoryMode) != 0)
                {
                    if (errno == EEXIST)
                    {
                        continue;
                    }
                    throw_errno("failed to create trace spill lease directory");
                }

                auto lease_fd = ::openat(
                    root_fd,
                    staging_name.c_str(),
                    O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
                if (lease_fd < 0)
                {
                    (void)::unlinkat(root_fd, staging_name.c_str(), AT_REMOVEDIR);
                    throw_errno("failed to open trace spill lease directory");
                }

                struct stat directory_status{};
                if (::fstat(lease_fd, &directory_status) != 0 ||
                    !owner_and_mode(directory_status, kPrivateDirectoryMode))
                {
                    (void)::close(lease_fd);
                    (void)::unlinkat(root_fd, staging_name.c_str(), AT_REMOVEDIR);
                    throw std::runtime_error(
                        "trace spill lease directory permissions are invalid");
                }

                int lock_fd = ::openat(
                    lease_fd,
                    kLeaseLockName,
                    O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                    kPrivateFileMode);
                if (lock_fd < 0)
                {
                    (void)::close(lease_fd);
                    (void)::unlinkat(root_fd, staging_name.c_str(), AT_REMOVEDIR);
                    throw_errno("failed to create trace spill lease lock");
                }
                struct stat lock_status{};
                if (::fchmod(lock_fd, kPrivateFileMode) != 0 ||
                    ::fstat(lock_fd, &lock_status) != 0 ||
                    !valid_private_regular(lock_status) ||
                    ::flock(lock_fd, LOCK_EX | LOCK_NB) != 0)
                {
                    const auto saved_errno = errno != 0 ? errno : EACCES;
                    (void)::close(lock_fd);
                    (void)::unlinkat(
                        lease_fd,
                        kLeaseLockName,
                        0);
                    (void)::close(lease_fd);
                    (void)::unlinkat(root_fd, staging_name.c_str(), AT_REMOVEDIR);
                    errno = saved_errno;
                    throw_errno("failed to lock trace spill lease");
                }

                int metadata_fd = ::openat(
                    lease_fd,
                    kMetadataName,
                    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                    kPrivateFileMode);
                if (metadata_fd < 0)
                {
                    const auto saved_errno = errno;
                    (void)::flock(lock_fd, LOCK_UN);
                    (void)::close(lock_fd);
                    (void)::unlinkat(
                        lease_fd,
                        kLeaseLockName,
                        0);
                    (void)::close(lease_fd);
                    (void)::unlinkat(root_fd, staging_name.c_str(), AT_REMOVEDIR);
                    errno = saved_errno;
                    throw_errno("failed to create trace spill lease metadata");
                }

                bool published = false;
                struct stat metadata_status{};
                try
                {
                    const auto metadata =
                        std::string("version=1\n") +
                        "name=" + name + "\n" +
                        "pid=" + std::to_string(creator_pid) + "\n" +
                        "start_ticks=" + std::to_string(creator_start_ticks) + "\n" +
                        "nonce=" + nonce + "\n";
                    write_all(
                        metadata_fd,
                        metadata,
                        "failed to write trace spill lease metadata");
                    if (::fsync(metadata_fd) != 0 ||
                        ::fchmod(metadata_fd, kPrivateFileMode) != 0)
                    {
                        throw_errno(
                            "failed to persist trace spill lease metadata");
                    }
                    if (::fstat(metadata_fd, &metadata_status) != 0 ||
                        !S_ISREG(metadata_status.st_mode) ||
                        !owner_and_mode(metadata_status, kPrivateFileMode) ||
                        metadata_status.st_nlink != 1)
                    {
                        throw std::runtime_error(
                            "trace spill lease metadata permissions are invalid");
                    }
                    (void)::close(metadata_fd);
                    metadata_fd = -1;
                    if (::fsync(lease_fd) != 0)
                    {
                        throw_errno(
                            "failed to persist trace spill lease directory");
                    }
                    if (::renameat(
                            root_fd,
                            staging_name.c_str(),
                            root_fd,
                            name.c_str()) != 0)
                    {
                        throw_errno(
                            "failed to publish trace spill lease directory");
                    }
                    published = true;
                    if (::fsync(root_fd) != 0)
                    {
                        throw_errno(
                            "failed to persist trace spill lease publication");
                    }
                }
                catch (...)
                {
                    const auto saved_errno = errno;
                    close_noexcept(&metadata_fd);
                    (void)::flock(lock_fd, LOCK_UN);
                    (void)::close(lock_fd);
                    (void)::unlinkat(
                        lease_fd,
                        kMetadataName,
                        0);
                    (void)::unlinkat(
                        lease_fd,
                        kLeaseLockName,
                        0);
                    (void)::close(lease_fd);
                    (void)::unlinkat(
                        root_fd,
                        (published ? name : staging_name).c_str(),
                        AT_REMOVEDIR);
                    errno = saved_errno;
                    throw;
                }

                std::unique_ptr<TraceSpillLease> owned;
                try
                {
                    owned = std::unique_ptr<TraceSpillLease>(
                        new TraceSpillLease(
                            root_fd,
                            lease_fd,
                            lock_fd,
                            normalized,
                            normalized / name,
                            name,
                            static_cast<std::uint64_t>(directory_status.st_dev),
                            static_cast<std::uint64_t>(directory_status.st_ino),
                            creator_pid,
                            creator_start_ticks,
                            nonce,
                            metadata_status));
                }
                catch (...)
                {
                    // Publication preceded the object allocation.  Complete
                    // the same authenticated cleanup while the root lock is
                    // still held so a bad_alloc cannot strand an active
                    // creator record in this process.
                    try
                    {
                        (void)safe_remove_lease(
                            root_fd,
                            name,
                            lease_fd,
                            lock_fd,
                            directory_status,
                            expected_metadata,
                            metadata_status);
                    }
                    catch (...)
                    {
                    }
                    close_noexcept(&lease_fd);
                    unlock_close(&lock_fd);
                    throw;
                }
                unlock_close(&root_lock_fd);
                auto result =
                    std::shared_ptr<const TraceSpillLease>(
                        std::move(owned));
                return result;
            }

            throw std::runtime_error(
                "failed to allocate a unique trace spill lease");
        }
        catch (...)
        {
            unlock_close(&root_lock_fd);
            (void)::close(root_fd);
            throw;
        }
    }

    void TraceSpillLease::collect_stale(
        const std::filesystem::path &root)
    {
        const auto root_fd = open_private_directory_tree(root);
        int root_lock_fd = -1;
        try
        {
            root_lock_fd = open_root_lock(root_fd);
            collect_stale_locked(root_fd);
            unlock_close(&root_lock_fd);
            (void)::close(root_fd);
        }
        catch (...)
        {
            unlock_close(&root_lock_fd);
            (void)::close(root_fd);
            throw;
        }
    }

    TraceSpillLease::TraceSpillLease(
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
        struct stat metadata_status) noexcept
        : root_fd_(root_fd),
          lease_fd_(lease_fd),
          lock_fd_(lock_fd),
          root_(std::move(root)),
          path_(std::move(path)),
          name_(std::move(name)),
          device_(device),
          inode_(inode),
          creator_pid_(creator_pid),
          creator_start_ticks_(creator_start_ticks),
          creator_nonce_(std::move(creator_nonce)),
          metadata_status_(metadata_status)
    {
    }

    TraceSpillLease::~TraceSpillLease() noexcept
    {
        bool creator_process = false;
        try
        {
            const auto current_identity = process_start_ticks(
                static_cast<std::uint64_t>(::getpid()));
            creator_process =
                current_identity.exists && current_identity.readable &&
                static_cast<std::uint64_t>(::getpid()) == creator_pid_ &&
                current_identity.start_ticks == creator_start_ticks_;
            if (creator_process && root_fd_ >= 0 && lease_fd_ >= 0)
            {
                struct stat expected_directory{};
                expected_directory.st_dev = static_cast<dev_t>(device_);
                expected_directory.st_ino = static_cast<ino_t>(inode_);
                expected_directory.st_mode = S_IFDIR | kPrivateDirectoryMode;
                const LeaseMetadata expected_metadata{
                    .name = name_,
                    .pid = creator_pid_,
                    .start_ticks = creator_start_ticks_,
                    .nonce = creator_nonce_,
                };
                remove_owned_lease_best_effort(
                    root_fd_,
                    name_,
                    lease_fd_,
                    lock_fd_,
                    expected_directory,
                    expected_metadata,
                    metadata_status_);
            }
        }
        catch (...)
        {
            // All cleanup below is descriptor-only and noexcept.  In
            // particular, a failed process-identity allocation must not turn
            // a normal destructor into std::terminate.
        }
        close_noexcept(&lease_fd_);
        // A fork child inherits the parent's open file description.  Calling
        // flock(LOCK_UN) here would unlock the parent's lease, so a
        // non-creator process may only close its inherited descriptors.
        if (creator_process)
        {
            unlock_close(&lock_fd_);
        }
        else
        {
            close_noexcept(&lock_fd_);
        }
        close_noexcept(&root_fd_);
    }

    std::string TraceSpillLease::payload_name(
        const std::filesystem::path &payload_path) const
    {
        std::error_code absolute_error;
        const auto absolute =
            std::filesystem::absolute(payload_path, absolute_error)
                .lexically_normal();
        if (absolute_error || absolute.parent_path() != path_)
        {
            throw std::runtime_error(
                "trace spill payload path escapes its lease");
        }
        const auto name = absolute.filename().string();
        if (!valid_payload_name(name))
        {
            throw std::runtime_error(
                "trace spill payload path has an unsafe basename");
        }
        return name;
    }

    std::unique_ptr<std::ostream> TraceSpillLease::create_payload_stream(
        std::string_view name) const
    {
        auto fd = create_payload_fd(lease_fd_, name);
        bool transferred = false;
        try
        {
            auto stream = std::make_unique<FdOutputStream>(fd);
            transferred = true;
            if (!stream->is_open() || !stream->good())
            {
                throw std::runtime_error(
                    "failed to open trace spill payload stream");
            }
            return stream;
        }
        catch (...)
        {
            if (!transferred)
            {
                close_noexcept(&fd);
            }
            remove_payload(name);
            throw;
        }
    }

    std::unique_ptr<std::istream> TraceSpillLease::open_payload_stream(
        std::string_view name) const
    {
        auto fd = open_payload_fd(lease_fd_, name);
        bool transferred = false;
        try
        {
            auto stream = std::make_unique<FdInputStream>(fd);
            transferred = true;
            if (!stream->is_open() || !stream->good())
            {
                throw std::runtime_error(
                    "failed to open trace spill payload stream");
            }
            return stream;
        }
        catch (...)
        {
            if (!transferred)
            {
                close_noexcept(&fd);
            }
            throw;
        }
    }

    std::uintmax_t TraceSpillLease::payload_size(
        std::string_view name) const
    {
        const auto fd = open_payload_fd(lease_fd_, name);
        struct stat status{};
        const auto result =
            ::fstat(fd, &status) == 0 && status.st_size >= 0
                ? static_cast<std::uintmax_t>(status.st_size)
                : std::numeric_limits<std::uintmax_t>::max();
        const auto saved_errno = errno;
        (void)::close(fd);
        if (result == std::numeric_limits<std::uintmax_t>::max())
        {
            errno = saved_errno != 0 ? saved_errno : EIO;
            throw_errno("failed to stat trace spill payload");
        }
        return result;
    }

    void TraceSpillLease::remove_payload(
        std::string_view name) const noexcept
    {
        try
        {
            if (lease_fd_ < 0 || !valid_payload_name(name))
            {
                return;
            }
            const std::string owned_name(name);
            struct stat status{};
            if (::fstatat(
                    lease_fd_,
                    owned_name.c_str(),
                    &status,
                    AT_SYMLINK_NOFOLLOW) != 0 ||
                !valid_private_regular(status))
            {
                return;
            }
            (void)::unlinkat(lease_fd_, owned_name.c_str(), 0);
        }
        catch (...)
        {
            // Failure to remove a failed payload is intentionally fail-closed;
            // the authenticated lease destructor remains the sole owner of
            // final cleanup.
            return;
        }
    }

} // namespace gatzk::util
