#include <iostream>

#include "icicle/runtime.h"
#include "icicle/device_api.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"

using namespace icicle;

extern "C" eIcicleError icicle_set_device(const Device& device) { return DeviceAPI::set_thread_local_device(device); }

extern "C" eIcicleError icicle_set_default_device(const Device& device)
{
  return DeviceAPI::set_default_device(device);
}

extern "C" eIcicleError icicle_get_active_device(icicle::Device& device)
{
  const Device& active_device = DeviceAPI::get_thread_local_device();
  device = active_device;
  return eIcicleError::SUCCESS;
}

extern "C" eIcicleError icicle_is_host_memory(const void* ptr)
{
  auto it = DeviceAPI::get_global_memory_tracker().identify(ptr);
  return (it == std::nullopt) ? eIcicleError::SUCCESS : eIcicleError::INVALID_POINTER;
}

extern "C" eIcicleError icicle_is_active_device_memory(const void* ptr)
{
  auto it = DeviceAPI::get_global_memory_tracker().identify(ptr);
  if (it == std::nullopt) { return eIcicleError::INVALID_POINTER; }
  auto [device, offset] = *it;
  return (*device == DeviceAPI::get_thread_local_device()) ? eIcicleError::SUCCESS : eIcicleError::INVALID_POINTER;
}

extern "C" eIcicleError icicle_get_device_count(int& device_count /*OUT*/)
{
  return DeviceAPI::get_thread_local_deviceAPI()->get_device_count(device_count);
}

extern "C" eIcicleError icicle_malloc(void** ptr, size_t size)
{
  if (size == 0) {
    static uint64_t dummy = 0;
    *ptr = static_cast<void*>(&dummy);
    DeviceAPI::get_global_memory_tracker().add_allocation(*ptr, size, DeviceAPI::get_thread_local_device());
    return eIcicleError::SUCCESS;
  }

  auto err = DeviceAPI::get_thread_local_deviceAPI()->allocate_memory(ptr, size);
  if (err == eIcicleError::SUCCESS) {
    DeviceAPI::get_global_memory_tracker().add_allocation(*ptr, size, DeviceAPI::get_thread_local_device());
  }
  return err;
}

extern "C" eIcicleError icicle_malloc_async(void** ptr, size_t size, icicleStreamHandle stream)
{
  if (size == 0) {
    static uint64_t dummy = 0;
    *ptr = static_cast<void*>(&dummy);
    DeviceAPI::get_global_memory_tracker().add_allocation(*ptr, size, DeviceAPI::get_thread_local_device());
    return eIcicleError::SUCCESS;
  }

  auto err = DeviceAPI::get_thread_local_deviceAPI()->allocate_memory_async(ptr, size, stream);
  if (err == eIcicleError::SUCCESS) {
    DeviceAPI::get_global_memory_tracker().add_allocation(*ptr, size, DeviceAPI::get_thread_local_device());
  }
  return err;
}

extern "C" eIcicleError icicle_free(void* ptr)
{
  // If releasing memory of non-active device, switch device, release and switch back
  // Alternatively I would have to consider it an error but it means that user has to switch device before dropping
  // (when scope is closed)

  auto& tracker = DeviceAPI::get_global_memory_tracker();
  const auto& cur_device = DeviceAPI::get_thread_local_device();

  const auto& it = tracker.identify(ptr);
  if (it == std::nullopt) {
    ICICLE_LOG_ERROR << "Trying to release host memory from device " << cur_device.type << " " << cur_device.id;
    return eIcicleError::INVALID_DEVICE;
  }
  auto [ptr_dev, offset] = *it;
  const bool is_active_device = *ptr_dev == cur_device;
  if (is_active_device) {
    auto err = DeviceAPI::get_thread_local_deviceAPI()->free_memory(ptr);
    if (err == eIcicleError::SUCCESS) { DeviceAPI::get_global_memory_tracker().remove_allocation(ptr); }
    return err;
  }

  // Getting here means memory does not belong to active device
  auto err = icicle_set_device(*ptr_dev);
  if (err == eIcicleError::SUCCESS) err = icicle_free(ptr);
  err = icicle_set_device(cur_device);
  return err;
}

extern "C" eIcicleError icicle_free_async(void* ptr, icicleStreamHandle stream)
{
  auto& tracker = DeviceAPI::get_global_memory_tracker();
  const auto& it = tracker.identify(ptr);
  const auto& cur_device = DeviceAPI::get_thread_local_device();

  if (it == std::nullopt) {
    ICICLE_LOG_ERROR << "Trying to release host memory from device " << cur_device.type << " " << cur_device.id;
    return eIcicleError::INVALID_DEVICE;
  }
  auto [device, offset] = *it;

  // Note that in that case, not switching device, since the stream may be wrong too. User has to handle it.
  const bool is_active_device = *device == cur_device;
  if (!is_active_device) {
    ICICLE_LOG_ERROR << "Trying to release memory allocated by " << device->type << "(" << device->id
                     << ") from device " << cur_device.type << "(" << cur_device.id << ")";
    return eIcicleError::INVALID_DEVICE;
  }

  auto err = DeviceAPI::get_thread_local_deviceAPI()->free_memory_async(ptr, stream);
  if (err == eIcicleError::SUCCESS) { DeviceAPI::get_global_memory_tracker().remove_allocation(ptr); }
  return err;
}

extern "C" eIcicleError icicle_get_available_memory(size_t& total /*OUT*/, size_t& free /*OUT*/)
{
  return DeviceAPI::get_thread_local_deviceAPI()->get_available_memory(total, free);
}

extern "C" eIcicleError icicle_memset(void* ptr, int value, size_t size)
{
  if (eIcicleError::SUCCESS == icicle_is_active_device_memory(ptr)) {
    return DeviceAPI::get_thread_local_deviceAPI()->memset(ptr, value, size);
  }
  ICICLE_LOG_ERROR << "icicle_memset API not expecting host memory";
  return eIcicleError::INVALID_POINTER;
}

extern "C" eIcicleError icicle_memset_async(void* ptr, int value, size_t size, icicleStreamHandle stream)
{
  if (eIcicleError::SUCCESS == icicle_is_active_device_memory(ptr)) {
    return DeviceAPI::get_thread_local_deviceAPI()->memset_async(ptr, value, size, stream);
  }
  ICICLE_LOG_ERROR << "icicle_memset_async API not expecting host memory";
  return eIcicleError::INVALID_POINTER;
}

/**
 * @brief Enum for specifying the type of memory.
 */
enum class MemoryType {
  Untracked,       ///< Memory is not tracked, assumed to be host memory
  ActiveDevice,    ///< Memory is on the active device
  NonActiveDevice, ///< Memory is on a non-active device
};

static MemoryType _get_memory_type(const void* ptr)
{
  auto it = DeviceAPI::get_global_memory_tracker().identify(ptr);
  // Untracked address assumed to be host memory but could be invalid
  if (it == std::nullopt) { return MemoryType::Untracked; }
  auto [device, offset] = *it;

  const bool is_active_device_ptr = (*device == DeviceAPI::get_thread_local_device());
  return is_active_device_ptr ? MemoryType::ActiveDevice : MemoryType::NonActiveDevice;
}

static eIcicleError _determine_copy_direction(void* dst, const void* src, eCopyDirection& direction)
{
  // Determine the type of memory for dst and src
  MemoryType dstType = _get_memory_type(dst);
  MemoryType srcType = _get_memory_type(src);

  if (dstType == MemoryType::Untracked && srcType == MemoryType::Untracked) {
    direction = HostToHost;
    return eIcicleError::SUCCESS;
  }
  if (dstType == MemoryType::NonActiveDevice || srcType == MemoryType::NonActiveDevice) {
    ICICLE_LOG_ERROR << "Either dst or src is on a non-active device memory";
    return eIcicleError::INVALID_POINTER;
  }

  direction = srcType == MemoryType::ActiveDevice && dstType == MemoryType::Untracked   ? eCopyDirection::DeviceToHost
              : srcType == MemoryType::Untracked && dstType == MemoryType::ActiveDevice ? eCopyDirection::HostToDevice
              : srcType == MemoryType::ActiveDevice && dstType == MemoryType::ActiveDevice
                ? eCopyDirection::DeviceToDevice
                : eCopyDirection::HostToDevice; // This line should never be reached

  return eIcicleError::SUCCESS;
}

extern "C" eIcicleError icicle_copy(void* dst, const void* src, size_t size)
{
  // NOTE: memory allocated outside of icicle APIs is considered host memory. Do not use it with memory allocated by
  // external libs (e.g. a cudaMalloc() call)

  eCopyDirection direction;
  auto err = _determine_copy_direction(dst, src, direction);
  if (eIcicleError::SUCCESS != err) { return err; }
  if (eCopyDirection::HostToHost == direction) {
    ICICLE_LOG_VERBOSE
      << "Host to Host copy, falling back to std::memcpy(). NOTE: memory allocated outside of icicle APIs is "
         "considered host memory. Do not use icicle_copy() with memory allocated by external libs (e.g. a cudaMalloc() "
         "call)";
    std::memcpy(dst, src, size);
    return eIcicleError::SUCCESS;
  }
  // Call the appropriate copy method
  return DeviceAPI::get_thread_local_deviceAPI()->copy(dst, src, size, direction);
}

extern "C" eIcicleError icicle_copy_async(void* dst, const void* src, size_t size, icicleStreamHandle stream)
{
  eCopyDirection direction;
  auto err = _determine_copy_direction(dst, src, direction);
  if (eIcicleError::SUCCESS != err) { return err; }
  if (eCopyDirection::HostToHost == direction) {
    ICICLE_LOG_VERBOSE
      << "Host to Host copy, falling back to std::memcpy(). NOTE: memory allocated outside of icicle APIs is "
         "considered host memory. Do not use icicle_copy() with memory allocated by external libs (e.g. a cudaMalloc() "
         "call)";
    std::memcpy(dst, src, size);
    return eIcicleError::SUCCESS;
  }
  // Call the appropriate copy method
  return DeviceAPI::get_thread_local_deviceAPI()->copy_async(dst, src, size, direction, stream);
}

extern "C" eIcicleError icicle_copy_to_host(void* dst, const void* src, size_t size)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy(dst, src, size, eCopyDirection::DeviceToHost);
}

extern "C" eIcicleError icicle_copy_to_host_async(void* dst, const void* src, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy_async(dst, src, size, eCopyDirection::DeviceToHost, stream);
}

extern "C" eIcicleError icicle_copy_to_device(void* dst, const void* src, size_t size)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy(dst, src, size, eCopyDirection::HostToDevice);
}

extern "C" eIcicleError icicle_copy_to_device_async(void* dst, const void* src, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy_async(dst, src, size, eCopyDirection::HostToDevice, stream);
}

extern "C" eIcicleError icicle_stream_synchronize(icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->synchronize(stream);
}

extern "C" eIcicleError icicle_device_synchronize()
{
  return DeviceAPI::get_thread_local_deviceAPI()->synchronize(nullptr);
}

extern "C" eIcicleError icicle_get_device_properties(DeviceProperties& properties)
{
  return DeviceAPI::get_thread_local_deviceAPI()->get_device_properties(properties);
}

extern "C" eIcicleError icicle_is_device_available(const Device& dev)
{
  return is_device_registered(dev.type) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
}

extern "C" eIcicleError icicle_get_registered_devices(char* output, size_t output_size)
{
  return get_registered_devices(output, output_size);
}

extern "C" eIcicleError icicle_create_stream(icicleStreamHandle* stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->create_stream(stream);
}

extern "C" eIcicleError icicle_destroy_stream(icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->destroy_stream(stream);
}
