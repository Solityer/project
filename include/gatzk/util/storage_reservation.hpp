#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace gatzk::util
{

    class FilesystemSpaceReservation
    {
    public:
        FilesystemSpaceReservation(const std::filesystem::path &root, std::size_t bytes, std::string_view label);

        FilesystemSpaceReservation(const FilesystemSpaceReservation &) = delete;
        FilesystemSpaceReservation &operator=(const FilesystemSpaceReservation &) = delete;

        ~FilesystemSpaceReservation();

        static std::size_t available_budget(const std::filesystem::path &root);

    private:
        std::string key_;
        std::size_t bytes_ = 0;
        std::uint64_t device_ = 0;
        std::uint64_t creator_pid_ = 0;
        std::uint64_t creator_start_ticks_ = 0;
        std::string token_;
        bool active_ = false;
    };

}
