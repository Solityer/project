#include "gatzk/util/storage_reservation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <signal.h>
#include <system_error>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

#include <openssl/rand.h>

namespace gatzk::util
{
    namespace
    {

        constexpr std::size_t kMinimumHeadroom = 256ULL << 20;
        constexpr std::size_t kMaximumHeadroom = 4ULL << 30;
        constexpr std::size_t kHeadroomDivisor = 20;
        // Filesystem free-space accounting can move by a few blocks while
        // the ledger is being atomically rewritten. Keep a bounded mutation
        // margin so two near-half reservations cannot cross the cap merely
        // because the ledger rename returned a small amount of space.
        constexpr std::size_t kLedgerMutationHeadroom = 64ULL << 20;
        constexpr std::size_t kLedgerReadLimit = 8ULL << 20;
        constexpr mode_t kPrivateDirectoryMode = S_IRWXU;
        constexpr mode_t kPrivateFileMode = S_IRUSR | S_IWUSR;
        constexpr char kLedgerDirectoryName[] = ".gatzk-storage-ledger";

        std::atomic<std::uint64_t> ledger_nonce{0};

        [[noreturn]] void throw_errno(const char *operation)
        {
            throw std::system_error(errno, std::generic_category(), operation);
        }

        void close_noexcept(int *fd) noexcept
        {
            if (fd != nullptr && *fd >= 0)
            {
                (void)::close(*fd);
                *fd = -1;
            }
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

        bool owner_and_mode(const struct stat &status, mode_t mode)
        {
            return status.st_uid == ::geteuid() && (status.st_mode & 0777) == mode;
        }

        bool valid_private_regular(const struct stat &status)
        {
            return S_ISREG(status.st_mode) && owner_and_mode(status, kPrivateFileMode) && status.st_nlink == 1;
        }

        void write_all(int fd, std::string_view bytes, const char *operation)
        {
            std::size_t offset = 0;
            while (offset < bytes.size())
            {
                const auto written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
                if (written < 0)
                {
                    throw_errno(operation);
                }
                if (written == 0)
                {
                    throw std::runtime_error(std::string(operation) + " made no progress");
                }
                offset += static_cast<std::size_t>(written);
            }
        }

        std::string read_fd_bounded(int fd, std::size_t limit, const char *operation)
        {
            if (::lseek(fd, 0, SEEK_SET) < 0)
            {
                throw_errno(operation);
            }
            std::string result;
            result.reserve(std::min<std::size_t>(limit, 4096));
            std::array<char, 4096> buffer{};
            for (;;)
            {
                const auto count = ::read(fd, buffer.data(), buffer.size());
                if (count < 0)
                {
                    throw_errno(operation);
                }
                if (count == 0)
                {
                    break;
                }
                if (result.size() > limit - static_cast<std::size_t>(count))
                {
                    throw std::runtime_error(std::string(operation) + " exceeds ledger size limit");
                }
                result.append(buffer.data(), static_cast<std::size_t>(count));
            }
            return result;
        }

        std::uint64_t parse_u64(std::string_view value, const char *field)
        {
            if (value.empty())
                throw std::runtime_error(std::string("storage ledger has empty ") + field);
            std::uint64_t result = 0;
            for (const auto ch : value)
            {
                if (ch < '0' || ch > '9')
                    throw std::runtime_error(std::string("storage ledger has invalid ") + field);
                const auto digit = static_cast<std::uint64_t>(ch - '0');
                if (result > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
                    throw std::runtime_error(std::string("storage ledger ") + field + " overflows");
                result = result * 10 + digit;
            }
            return result;
        }

        struct ProcessIdentity
        {
            bool exists = false;
            bool readable = false;
            std::uint64_t start_ticks = 0;
        };

        ProcessIdentity process_identity(std::uint64_t pid)
        {
            if (pid == 0 || pid > static_cast<std::uint64_t>(INT_MAX))
                return {};
            const auto proc_path = std::string("/proc/") + std::to_string(pid) + "/stat";
            const auto fd = ::open(proc_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            if (fd < 0)
            {
                if (errno == ENOENT || errno == ESRCH)
                {
                    errno = 0;
                    if (::kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM)
                        return {.exists = true, .readable = false};
                    if (errno == ESRCH)
                        return {};
                }
                return {.exists = true, .readable = false};
            }
            std::string text;
            try
            {
                text = read_fd_bounded(fd, 8192, "failed to read process identity");
            }
            catch (const std::exception &)
            {
                (void)::close(fd);
                return {.exists = true, .readable = false};
            }
            (void)::close(fd);
            const auto close_paren = text.rfind(')');
            if (close_paren == std::string::npos || close_paren + 2 >= text.size())
                return {.exists = true, .readable = false};
            std::istringstream fields(text.substr(close_paren + 2));
            std::string token;
            for (int field = 3; field <= 22; ++field)
            {
                if (!(fields >> token))
                    return {.exists = true, .readable = false};
                if (field == 22)
                {
                    try
                    {
                        return {
                            .exists = true,
                            .readable = true,
                            .start_ticks = parse_u64(token, "process start"),
                        };
                    }
                    catch (const std::exception &)
                    {
                        return {.exists = true, .readable = false};
                    }
                }
            }
            return {.exists = true, .readable = false};
        }

        std::uint64_t current_process_start_ticks()
        {
            const auto result = process_identity(static_cast<std::uint64_t>(::getpid()));
            if (!result.exists || !result.readable || result.start_ticks == 0)
                throw std::runtime_error("cannot authenticate current process identity for storage reservation");
            return result.start_ticks;
        }

        struct FilesystemBudget
        {
            std::filesystem::path canonical_root;
            std::uint64_t device = 0;
            std::size_t remaining = 0;
        };

        FilesystemBudget inspect_budget(const std::filesystem::path &root, std::string_view label)
        {
            std::error_code absolute_error;
            const auto absolute = std::filesystem::absolute(root, absolute_error);
            if (absolute_error || absolute.empty())
                throw std::runtime_error("unable to establish storage root for " + std::string(label) + ": " +
                                         (absolute_error ? absolute_error.message() : "empty path"));
            std::error_code canonical_error;
            const auto canonical = std::filesystem::weakly_canonical(absolute, canonical_error);
            if (canonical_error || canonical.empty() || !canonical.is_absolute())
                throw std::runtime_error("unable to establish canonical storage root for " + std::string(label) + ": " +
                                         (canonical_error ? canonical_error.message() : "non-absolute path"));
            const auto fd = ::open(canonical.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            if (fd < 0)
                throw_errno("failed to open storage root");
            try
            {
                struct stat status{};
                if (::fstat(fd, &status) != 0 || !S_ISDIR(status.st_mode))
                {
                    const auto saved_errno = errno != 0 ? errno : ENOTDIR;
                    errno = saved_errno;
                    throw_errno("storage root is not a directory");
                }

                // Query the filesystem through the authenticated directory FD
                // while it remains open.  A path-based space() call after the
                // close would re-resolve a potentially replaced root.
                struct statvfs filesystem_status{};
                if (::fstatvfs(fd, &filesystem_status) != 0)
                    throw_errno("failed to inspect storage filesystem");
                const auto blocks_to_bytes = [&](auto blocks)
                {
                    const auto count = static_cast<std::uintmax_t>(blocks);
                    const auto unit = static_cast<std::uintmax_t>(filesystem_status.f_frsize != 0 ? filesystem_status.f_frsize : filesystem_status.f_bsize);
                    if (unit == 0 || count > std::numeric_limits<std::uintmax_t>::max() / unit)
                        throw std::runtime_error("filesystem storage capacity overflows");
                    return count * unit;
                };
                const auto capacity_u = blocks_to_bytes(filesystem_status.f_blocks);
                const auto available_u = blocks_to_bytes(filesystem_status.f_bavail);
                if (capacity_u == 0 || available_u > capacity_u || capacity_u > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()))
                    throw std::runtime_error("unable to establish filesystem storage budget for " + std::string(label) + ": invalid filesystem capacity");
                const auto capacity = static_cast<std::size_t>(capacity_u);
                const auto available = static_cast<std::size_t>(available_u);
                auto headroom = std::max(kMinimumHeadroom, std::min(kMaximumHeadroom, capacity / kHeadroomDivisor));
                headroom = std::min(headroom, capacity / 2);
                if (headroom < capacity / 2)
                {
                    const auto mutation_headroom = std::min(kLedgerMutationHeadroom, std::numeric_limits<std::size_t>::max() - headroom);
                    headroom = std::min(capacity / 2, headroom + mutation_headroom);
                }
                if (headroom == 0 || capacity < headroom)
                    throw std::runtime_error("filesystem storage budget has no safe headroom for " + std::string(label));
                const auto used = capacity - available;
                const auto total_budget = capacity - headroom;
                if (used > total_budget)
                    throw std::runtime_error("filesystem storage budget is exhausted for " + std::string(label));
                const auto result = FilesystemBudget{
                    .canonical_root = canonical.lexically_normal(),
                    .device = static_cast<std::uint64_t>(status.st_dev),
                    .remaining = total_budget - used,
                };
                (void)::close(fd);
                return result;
            }
            catch (...)
            {
                (void)::close(fd);
                throw;
            }
        }

        struct LedgerRecord
        {
            std::string token;
            std::uint64_t pid = 0;
            std::uint64_t start_ticks = 0;
            std::size_t bytes = 0;
        };

        struct LedgerHandles
        {
            int directory_fd = -1;
            int lock_fd = -1;
            int ledger_fd = -1;
            std::string lock_name;
            std::string ledger_name;
        };

        struct LedgerTempIdentity
        {
            std::uint64_t pid = 0;
            std::uint64_t start_ticks = 0;
            std::string nonce;
        };

        std::string ledger_temp_prefix(const LedgerHandles &handles)
        {
            return "." + handles.ledger_name + ".tmp-";
        }

        std::string ledger_random_nonce()
        {
            std::array<std::uint8_t, 16> bytes{};
            if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1)
                throw std::runtime_error("CSPRNG failed while creating storage ledger staging file");
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

        std::optional<LedgerTempIdentity> parse_ledger_temp_name(std::string_view name, std::string_view prefix)
        {
            if (name.substr(0, prefix.size()) != prefix)
                return std::nullopt;
            const auto suffix = name.substr(prefix.size());
            const auto first = suffix.find('-');
            const auto second = first == std::string_view::npos ? std::string_view::npos : suffix.find('-', first + 1);
            if (first == std::string_view::npos || second == std::string_view::npos || first == 0 || second <= first + 1 || second + 1 >= suffix.size())
                return std::nullopt;
            const auto nonce = suffix.substr(second + 1);
            if (nonce.size() != 32)
                return std::nullopt;
            for (const auto ch : nonce)
            {
                if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
                    return std::nullopt;
            }
            try
            {
                const auto pid = parse_u64(suffix.substr(0, first), "ledger temp pid");
                const auto start_ticks = parse_u64(suffix.substr(first + 1, second - first - 1), "ledger temp start ticks");
                if (pid == 0 || pid > static_cast<std::uint64_t>(INT_MAX) || start_ticks == 0)
                    return std::nullopt;
                return LedgerTempIdentity{
                    .pid = pid,
                    .start_ticks = start_ticks,
                    .nonce = std::string(nonce),
                };
            }
            catch (const std::exception &)
            {
                return std::nullopt;
            }
        }

        int open_ledger_directory()
        {
            std::error_code temp_error;
            const auto temp = std::filesystem::weakly_canonical(std::filesystem::temp_directory_path(temp_error), temp_error);
            if (temp_error || temp.empty() || !temp.is_absolute())
                throw std::runtime_error("unable to establish storage ledger parent: " + (temp_error ? temp_error.message() : "invalid path"));
            const auto parent_fd = ::open(temp.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            if (parent_fd < 0)
                throw_errno("failed to open storage ledger parent");
            if (::mkdirat(parent_fd, kLedgerDirectoryName, kPrivateDirectoryMode) != 0 && errno != EEXIST)
            {
                const auto saved_errno = errno;
                (void)::close(parent_fd);
                errno = saved_errno;
                throw_errno("failed to create storage ledger directory");
            }
            const auto directory_fd = ::openat(parent_fd, kLedgerDirectoryName, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            (void)::close(parent_fd);
            if (directory_fd < 0)
                throw_errno("failed to open storage ledger directory");
            struct stat status{};
            if (::fchmod(directory_fd, kPrivateDirectoryMode) != 0 || ::fstat(directory_fd, &status) != 0 || !S_ISDIR(status.st_mode) ||
                !owner_and_mode(status, kPrivateDirectoryMode))
            {
                const auto saved_errno = errno != 0 ? errno : EACCES;
                (void)::close(directory_fd);
                errno = saved_errno;
                throw_errno("storage ledger directory permissions are invalid");
            }
            return directory_fd;
        }

        bool same_regular_file_identity(const struct stat &a, const struct stat &b)
        {
            return valid_private_regular(a) && valid_private_regular(b) && a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
                   a.st_nlink == b.st_nlink && a.st_size == b.st_size;
        }

        std::vector<std::string> ledger_directory_entries(int directory_fd)
        {
            const auto duplicate = ::fcntl(directory_fd, F_DUPFD_CLOEXEC, 0);
            if (duplicate < 0)
                throw_errno("failed to duplicate storage ledger directory FD");
            if (::lseek(duplicate, 0, SEEK_SET) < 0)
            {
                const auto saved_errno = errno;
                (void)::close(duplicate);
                errno = saved_errno;
                throw_errno("failed to rewind storage ledger directory scan");
            }
            DIR *directory = ::fdopendir(duplicate);
            if (directory == nullptr)
            {
                const auto saved_errno = errno;
                (void)::close(duplicate);
                errno = saved_errno;
                throw_errno("failed to scan storage ledger directory");
            }
            std::vector<std::string> entries;
            errno = 0;
            while (auto *entry = ::readdir(directory))
            {
                const std::string name(entry->d_name);
                if (name != "." && name != "..")
                    entries.push_back(name);
            }
            const auto scan_error = errno;
            (void)::closedir(directory);
            if (scan_error != 0)
            {
                errno = scan_error;
                throw_errno("failed to scan storage ledger entries");
            }
            return entries;
        }

        void collect_stale_ledger_temps(const LedgerHandles &handles)
        {
            const auto prefix = ledger_temp_prefix(handles);
            std::vector<std::string> entries;
            try
            {
                entries = ledger_directory_entries(handles.directory_fd);
            }
            catch (const std::exception &)
            {
                // A complete authenticated view is required before cleanup;
                // leave every staging file untouched when scanning fails.
                return;
            }

            for (const auto &name : entries)
            {
                const auto identity = parse_ledger_temp_name(name, prefix);
                if (!identity.has_value())
                    continue;
                const auto fd = ::openat(handles.directory_fd, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
                if (fd < 0)
                    continue;
                struct stat snapshot{};
                if (::fstat(fd, &snapshot) != 0 || !valid_private_regular(snapshot))
                {
                    (void)::close(fd);
                    continue;
                }
                const auto process = process_identity(identity->pid);
                const bool active_or_unverifiable = (process.exists && !process.readable) || (process.exists && process.start_ticks == identity->start_ticks);
                if (active_or_unverifiable)
                {
                    (void)::close(fd);
                    continue;
                }

                // Revalidate the exact directory entry after process
                // authentication.  Any replacement, symlink or hardlink is
                // preserved rather than guessed at.
                struct stat current{};
                const bool unchanged =
                    ::fstatat(handles.directory_fd, name.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 && same_regular_file_identity(snapshot, current);
                (void)::close(fd);
                if (!unchanged)
                    continue;
                (void)::unlinkat(handles.directory_fd, name.c_str(), 0);
            }
        }

        LedgerHandles open_ledger(std::uint64_t device)
        {
            LedgerHandles handles;
            handles.directory_fd = open_ledger_directory();
            handles.lock_name = "dev-" + std::to_string(device) + ".lock";
            handles.ledger_name = "dev-" + std::to_string(device) + ".ledger";
            handles.lock_fd = ::openat(handles.directory_fd, handles.lock_name.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, kPrivateFileMode);
            if (handles.lock_fd < 0)
            {
                const auto saved_errno = errno;
                close_noexcept(&handles.directory_fd);
                errno = saved_errno;
                throw_errno("failed to open storage ledger lock");
            }
            struct stat lock_status{};
            if (::fstat(handles.lock_fd, &lock_status) != 0 || !valid_private_regular(lock_status) || ::flock(handles.lock_fd, LOCK_EX) != 0)
            {
                const auto saved_errno = errno != 0 ? errno : EACCES;
                close_noexcept(&handles.lock_fd);
                close_noexcept(&handles.directory_fd);
                errno = saved_errno;
                throw_errno("storage ledger lock is invalid");
            }
            // The per-device lock is the ledger/root serialization point.
            // Perform stale staging GC only while it is held, so a crashed
            // writer's temp cannot be mistaken for a concurrent writer.
            collect_stale_ledger_temps(handles);
            handles.ledger_fd = ::openat(handles.directory_fd, handles.ledger_name.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, kPrivateFileMode);
            if (handles.ledger_fd < 0)
            {
                const auto saved_errno = errno;
                unlock_close(&handles.lock_fd);
                close_noexcept(&handles.directory_fd);
                errno = saved_errno;
                throw_errno("failed to open storage ledger");
            }
            struct stat ledger_status{};
            if (::fstat(handles.ledger_fd, &ledger_status) != 0 || !valid_private_regular(ledger_status))
            {
                const auto saved_errno = errno != 0 ? errno : EACCES;
                close_noexcept(&handles.ledger_fd);
                unlock_close(&handles.lock_fd);
                close_noexcept(&handles.directory_fd);
                errno = saved_errno;
                throw_errno("storage ledger permissions are invalid");
            }
            return handles;
        }

        void close_ledger(LedgerHandles *handles) noexcept
        {
            if (handles == nullptr)
                return;
            close_noexcept(&handles->ledger_fd);
            unlock_close(&handles->lock_fd);
            close_noexcept(&handles->directory_fd);
        }

        std::vector<LedgerRecord> read_ledger(int fd)
        {
            const auto text = read_fd_bounded(fd, kLedgerReadLimit, "failed to read storage ledger");
            std::vector<LedgerRecord> records;
            std::unordered_set<std::string> tokens;
            std::size_t begin = 0;
            while (begin < text.size())
            {
                const auto end = text.find('\n', begin);
                if (end == std::string::npos)
                    throw std::runtime_error("storage ledger has an unterminated record");
                const auto line = text.substr(begin, end - begin);
                begin = end + 1;
                if (line.empty())
                    throw std::runtime_error("storage ledger has an empty record");
                std::istringstream fields(line);
                std::string version;
                std::string pid_text;
                std::string start_text;
                std::string bytes_text;
                LedgerRecord record;
                std::string extra;
                if (!(fields >> version >> record.token >> pid_text >> start_text >> bytes_text) || (fields >> extra) || version != "1" ||
                    record.token.empty() || record.token.size() > 128 || tokens.contains(record.token))
                    throw std::runtime_error("storage ledger record is invalid");
                record.pid = parse_u64(pid_text, "pid");
                record.start_ticks = parse_u64(start_text, "start_ticks");
                const auto bytes = parse_u64(bytes_text, "bytes");
                if (record.pid == 0 || record.pid > static_cast<std::uint64_t>(INT_MAX) || record.start_ticks == 0 || bytes == 0 ||
                    bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
                    throw std::runtime_error("storage ledger numeric fields are invalid");
                record.bytes = static_cast<std::size_t>(bytes);
                tokens.insert(record.token);
                records.push_back(std::move(record));
            }
            return records;
        }

        bool prune_stale(std::vector<LedgerRecord> *records)
        {
            bool changed = false;
            auto it = records->begin();
            while (it != records->end())
            {
                const auto process = process_identity(it->pid);
                const bool stale = !process.exists || (process.readable && process.start_ticks != it->start_ticks);
                if (stale)
                {
                    it = records->erase(it);
                    changed = true;
                }
                else
                    ++it;
            }
            return changed;
        }

        std::string serialize_ledger(const std::vector<LedgerRecord> &records)
        {
            std::ostringstream output;
            for (const auto &record : records)
                output << "1 " << record.token << ' ' << record.pid << ' ' << record.start_ticks << ' ' << record.bytes << '\n';
            return output.str();
        }

        void write_ledger(const LedgerHandles &handles, const std::vector<LedgerRecord> &records)
        {
            // Materialize all potentially-throwing state before publishing
            // the new ledger entry.  If publication itself reports an error,
            // callers can rewrite their pre-commit snapshot while retaining
            // the same per-device lock.
            const auto serialized = serialize_ledger(records);
            const auto creator_pid = static_cast<std::uint64_t>(::getpid());
            const auto creator_start_ticks = current_process_start_ticks();
            const auto prefix = ledger_temp_prefix(handles);
            for (std::size_t attempt = 0; attempt < 64; ++attempt)
            {
                const auto temporary_name = prefix + std::to_string(creator_pid) + "-" + std::to_string(creator_start_ticks) + "-" + ledger_random_nonce();
                const auto fd = ::openat(handles.directory_fd, temporary_name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, kPrivateFileMode);
                if (fd < 0)
                {
                    if (errno == EEXIST)
                        continue;
                    throw_errno("failed to create storage ledger staging file");
                }
                bool closed = false;
                bool published = false;
                try
                {
                    struct stat status{};
                    if (::fstat(fd, &status) != 0 || !valid_private_regular(status))
                        throw std::runtime_error("storage ledger staging permissions are invalid");
                    write_all(fd, serialized, "failed to write storage ledger");
                    if (::fsync(fd) != 0)
                        throw_errno("failed to persist storage ledger");
                    (void)::close(fd);
                    closed = true;
                    if (::renameat(handles.directory_fd, temporary_name.c_str(), handles.directory_fd, handles.ledger_name.c_str()) != 0)
                        throw_errno("failed to publish storage ledger");
                    published = true;
                    if (::fsync(handles.directory_fd) != 0)
                        throw_errno("failed to persist storage ledger publication");
                    return;
                }
                catch (...)
                {
                    if (!closed)
                        (void)::close(fd);
                    if (!published)
                        (void)::unlinkat(handles.directory_fd, temporary_name.c_str(), 0);
                    throw;
                }
            }
            throw std::runtime_error("failed to allocate a unique storage ledger staging file");
        }

        void write_ledger_with_rollback(const LedgerHandles &handles, const std::vector<LedgerRecord> &records,
                                        const std::vector<LedgerRecord> &committed_records)
        {
            try
            {
                write_ledger(handles, records);
            }
            catch (...)
            {
                // A failure after rename can leave the new snapshot visible
                // even though the caller cannot finish object construction.
                // Roll back under the same ledger lock.  If rollback itself
                // fails, retaining the authenticated old/new ledger is safer
                // than deleting or guessing at either snapshot.
                try
                {
                    write_ledger(handles, committed_records);
                }
                catch (...)
                {
                }
                throw;
            }
        }

        std::size_t sum_reserved(const std::vector<LedgerRecord> &records)
        {
            std::size_t total = 0;
            for (const auto &record : records)
            {
                if (record.bytes > std::numeric_limits<std::size_t>::max() - total)
                    throw std::runtime_error("storage ledger reservation total overflows");
                total += record.bytes;
            }
            return total;
        }

        std::string make_token(std::uint64_t pid, std::uint64_t start_ticks)
        {
            return std::to_string(pid) + "-" + std::to_string(start_ticks) + "-" + std::to_string(ledger_nonce.fetch_add(1, std::memory_order_relaxed));
        }

    } // namespace

    std::size_t FilesystemSpaceReservation::available_budget(const std::filesystem::path &root)
    {
        const auto budget = inspect_budget(root, "budget query");
        auto handles = open_ledger(budget.device);
        try
        {
            auto records = read_ledger(handles.ledger_fd);
            const auto committed_records = records;
            if (prune_stale(&records))
                write_ledger_with_rollback(handles, records, committed_records);
            const auto reserved = sum_reserved(records);
            if (reserved > budget.remaining)
                throw std::runtime_error("filesystem storage reservation accounting exceeds budget");
            close_ledger(&handles);
            return budget.remaining - reserved;
        }
        catch (...)
        {
            close_ledger(&handles);
            throw;
        }
    }

    FilesystemSpaceReservation::FilesystemSpaceReservation(const std::filesystem::path &root, std::size_t bytes, std::string_view label) : bytes_(bytes)
    {
        if (bytes_ == 0)
            return;
        const auto budget = inspect_budget(root, label);
        auto canonical_key = budget.canonical_root.string();
        const auto creator_pid = static_cast<std::uint64_t>(::getpid());
        const auto creator_start_ticks = current_process_start_ticks();
        auto handles = open_ledger(budget.device);
        try
        {
            auto records = read_ledger(handles.ledger_fd);
            const auto committed_records = records;
            const auto changed = prune_stale(&records);
            const auto reserved = sum_reserved(records);
            if (reserved > budget.remaining || bytes_ > budget.remaining - reserved)
                throw std::runtime_error("filesystem storage reservation exceeds safe budget for " + std::string(label) +
                                         " (requested=" + std::to_string(bytes_) + ", available=" + std::to_string(budget.remaining) +
                                         ", reserved=" + std::to_string(reserved) + ")");
            token_ = make_token(creator_pid, creator_start_ticks);
            records.push_back(LedgerRecord{
                .token = token_,
                .pid = creator_pid,
                .start_ticks = creator_start_ticks,
                .bytes = bytes_,
            });
            // The rewrite is intentionally atomic even when no stale record
            // was found.  It keeps each reservation update bounded and makes
            // the ledger a single authoritative cross-process snapshot.
            (void)changed;
            write_ledger_with_rollback(handles, records, committed_records);
            key_ = std::move(canonical_key);
            device_ = budget.device;
            creator_pid_ = creator_pid;
            creator_start_ticks_ = creator_start_ticks;
            active_ = true;
            close_ledger(&handles);
        }
        catch (...)
        {
            close_ledger(&handles);
            throw;
        }
    }

    FilesystemSpaceReservation::~FilesystemSpaceReservation()
    {
        if (!active_)
            return;
        LedgerHandles handles;
        try
        {
            const auto current = process_identity(static_cast<std::uint64_t>(::getpid()));
            if (!current.exists || !current.readable || static_cast<std::uint64_t>(::getpid()) != creator_pid_ || current.start_ticks != creator_start_ticks_)
                return;
            handles = open_ledger(device_);
            auto records = read_ledger(handles.ledger_fd);
            const auto committed_records = records;
            const auto changed = prune_stale(&records);
            const auto it = std::find_if(
                records.begin(), records.end(), [&](const auto &record)
                { return record.token == token_ && record.pid == creator_pid_ && record.start_ticks == creator_start_ticks_ && record.bytes == bytes_; });
            const bool found = it != records.end();
            if (found)
                records.erase(it);
            if (changed || found)
                write_ledger_with_rollback(handles, records, committed_records);
            close_ledger(&handles);
        }
        catch (...)
        {
            close_ledger(&handles);
            // A destructor cannot report failure.  Leaving the authenticated
            // record is fail-closed; the next process reclaims it after this
            // process is gone and its PID/start tuple no longer exists.
        }
    }

} // namespace gatzk::util
