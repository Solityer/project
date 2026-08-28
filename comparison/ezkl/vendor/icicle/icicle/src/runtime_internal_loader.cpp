#include <dlfcn.h>
#include <string>
#include <sys/stat.h>

#include "icicle/utils/log.h"
#include "runtime_internal.h"

namespace {

#ifdef __linux__
const std::string SHARED_LIB_EXTENSION = ".so";
#elif __APPLE__
const std::string SHARED_LIB_EXTENSION = ".dylib";
#else
#error "Unsupported operating system"
#endif

bool is_single_link_regular_file(const char* path)
{
  if (path == nullptr) return false;
  struct stat path_stat;
  if (lstat(path, &path_stat) != 0) return false;
  return S_ISREG(path_stat.st_mode) && path_stat.st_nlink == 1;
}

bool is_shared_library(const std::string& filename)
{
  return filename.size() >= SHARED_LIB_EXTENSION.size() &&
         filename.compare(filename.size() - SHARED_LIB_EXTENSION.size(), SHARED_LIB_EXTENSION.size(),
                          SHARED_LIB_EXTENSION) == 0;
}

bool load_one_library(const char* file_path)
{
  if (!is_single_link_regular_file(file_path)) return false;
  const std::string path(file_path);
  const std::string file_name = path.substr(path.find_last_of("/\\") + 1);
  if (file_name.find("icicle_backend") == std::string::npos || !is_shared_library(file_name)) return false;

  // The device DSO must export registration symbols globally for the
  // field/curve/hash DSOs; all other authenticated DSOs remain local.
  // Resolve all imports while loading. Lazy relocation could make an
  // apparently successful load fail only after the authenticated wrapper had
  // returned, which is not acceptable for the formal GPU path.
  const int flags = (file_name.find("device") != std::string::npos) ? (RTLD_NOW | RTLD_GLOBAL)
                                                                       : (RTLD_NOW | RTLD_LOCAL);
  ICICLE_LOG_VERBOSE << "Attempting to load authenticated backend file: " << file_path;
  void* handle = dlopen(file_path, flags);
  if (handle == nullptr) {
    ICICLE_LOG_ERROR << "Failed to load " << file_path << ": " << dlerror();
    return false;
  }
  return true;
}

}  // namespace

extern "C" ICICLE_RUNTIME_INTERNAL_EXPORT icicle::eIcicleError
icicle_load_backend_file_internal(const char* path)
{
  return load_one_library(path) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::BACKEND_LOAD_FAILED;
}
