#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace gatzk::util
{

    // Cross-process, fail-closed reservation for bounded temporary storage.
    // Reservations are keyed by the real filesystem device identity (st_dev),
    // so trace and quotient spill roots on the same filesystem share one
    // locked ledger.
    // The reservation is execution metadata only; it never enters a proof or
    // changes the logical object represented by a spill file.
    class FilesystemSpaceReservation
    {
    public:
        FilesystemSpaceReservation(
            const std::filesystem::path &root,
            std::size_t bytes,
            std::string_view label);

        FilesystemSpaceReservation(
            const FilesystemSpaceReservation &) = delete;
        FilesystemSpaceReservation &operator=(
            const FilesystemSpaceReservation &) = delete;

        ~FilesystemSpaceReservation();

        // Return the currently reservable budget for a filesystem.  This is
        // an execution diagnostic used by short resource-regression tests and
        // runner-side scheduling; it includes reservations already held by
        // this process and remains fail-closed on stat/capacity errors.
        static std::size_t available_budget(
            const std::filesystem::path &root);

    private:
        std::string key_;
        std::size_t bytes_ = 0;
        std::uint64_t device_ = 0;
        std::uint64_t creator_pid_ = 0;
        std::uint64_t creator_start_ticks_ = 0;
        std::string token_;
        bool active_ = false;
    };

} // namespace gatzk::util
