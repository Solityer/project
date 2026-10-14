#include "/home/pzh/project/comparison/zkGPT/gat_adapter/native_storage.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    void* pointer = nullptr;
    try {
        const std::size_t page = zkgpt::native_storage::page_size();
        const std::size_t file_bytes = (std::size_t(16) << 20) + 1U;
        const std::size_t active_bytes = page;
        pointer = zkgpt::native_storage::allocate(file_bytes, alignof(std::max_align_t));
        const std::uint64_t before = zkgpt::native_storage::physical_bytes(pointer, file_bytes);
        const auto reclaimed = zkgpt::native_storage::reclaim_tail(pointer, active_bytes, file_bytes);
        const std::uint64_t after = zkgpt::native_storage::physical_bytes(pointer, file_bytes);
        if (!reclaimed.mapped || reclaimed.live_bytes != active_bytes || after >= before)
            throw std::runtime_error("scratch page-boundary reclaim postcondition failed");
        std::cout << "SCRATCH_PAGE_BOUNDARY_OK page_bytes=" << page
                  << " file_bytes=" << file_bytes << " live_bytes=" << active_bytes
                  << " allocated_before=" << before << " allocated_after=" << after << '\n';
        zkgpt::native_storage::deallocate(pointer, alignof(std::max_align_t));
        return 0;
    } catch (const std::exception& error) {
        if (pointer != nullptr) zkgpt::native_storage::deallocate(pointer, alignof(std::max_align_t));
        std::cerr << "SCRATCH_PAGE_BOUNDARY_FAILED: " << error.what() << '\n';
        return 1;
    }
}
