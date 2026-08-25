/*
 * Scoped CUDA kernel telemetry for the formal ezkl path.
 *
 * This is deliberately a tiny C shim around CUPTI's activity API.  The Rust
 * crate has unsafe_code denied, and keeping the ABI boundary in this file
 * prevents a guessed device marker from being treated as proof of GPU
 * execution.  Only kernel activity records whose device timestamps fall
 * inside the requested scope are counted.
 */

#define _POSIX_C_SOURCE 200809L

#include <cuda.h>
#include <cupti.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EZKL_GPU_BUFFER_BYTES (1u << 20)
#define EZKL_GPU_MAX_KERNELS 256u
#define EZKL_GPU_MAX_NAME 256u
#define EZKL_GPU_MAX_CATEGORY 48u
#define EZKL_GPU_MAX_ACTIVITY_KEYS 4096u

typedef struct {
  char name[EZKL_GPU_MAX_NAME];
  char category[EZKL_GPU_MAX_CATEGORY];
  uint64_t count;
  uint64_t first_start;
  uint64_t last_end;
  uint32_t device_id;
} ezkl_kernel_entry;

typedef struct {
  uint64_t start;
  uint64_t end;
  int64_t grid_id;
  uint32_t correlation_id;
  uint32_t stream_id;
  uint32_t device_id;
} ezkl_activity_key;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_initialized = 0;
static _Atomic int g_scope_active = 0;
static _Atomic int g_callback_error = 0;
static uint64_t g_scope_start = 0;
static uint64_t g_scope_end = 0;
static uint64_t g_scope_nonce = 0;
static char g_scope_name[64];
static uint64_t g_kernel_count = 0;
static uint64_t g_dropped_records = 0;
static uint32_t g_device_id = UINT32_MAX;
static ezkl_kernel_entry g_kernels[EZKL_GPU_MAX_KERNELS];
static ezkl_activity_key g_activity_keys[EZKL_GPU_MAX_ACTIVITY_KEYS];
static size_t g_activity_key_count = 0;

static void ezkl_set_callback_error(void) {
  atomic_store_explicit(&g_callback_error, 1, memory_order_release);
}

static void CUPTIAPI ezkl_buffer_requested(uint8_t **buffer, size_t *size,
                                           size_t *max_num_records) {
  *size = EZKL_GPU_BUFFER_BYTES;
  *buffer = (uint8_t *)malloc(*size);
  *max_num_records = 0;
  if (*buffer == NULL) {
    *size = 0;
    ezkl_set_callback_error();
  }
}

static const char *ezkl_kernel_category(const char *name) {
  if (name == NULL || name[0] == '\0') {
    return "unnamed_cuda_kernel";
  }
  if (strstr(name, "msm") != NULL || strstr(name, "MSM") != NULL) {
    return "msm";
  }
  if (strstr(name, "ntt") != NULL || strstr(name, "NTT") != NULL ||
      strstr(name, "fft") != NULL || strstr(name, "FFT") != NULL) {
    return "ntt_or_fft";
  }
  if (strstr(name, "icicle") != NULL || strstr(name, "Icicle") != NULL) {
    return "icicle";
  }
  if (strstr(name, "halo2") != NULL || strstr(name, "Halo2") != NULL) {
    return "halo2";
  }
  return "cuda_kernel";
}

static void ezkl_record_kernel(const CUpti_ActivityKernel10 *kernel) {
  if (kernel == NULL || kernel->name == NULL) {
    ezkl_set_callback_error();
    return;
  }
  if (kernel->start == 0 || kernel->end == 0 || kernel->end < kernel->start) {
    ezkl_set_callback_error();
    return;
  }
  if (kernel->start < g_scope_start || kernel->end > g_scope_end) {
    return;
  }
  if (g_device_id != UINT32_MAX && g_device_id != kernel->deviceId) {
    ezkl_set_callback_error();
    return;
  }
  for (size_t key = 0; key < g_activity_key_count; ++key) {
    const ezkl_activity_key *seen = &g_activity_keys[key];
    if (seen->start == kernel->start && seen->end == kernel->end &&
        seen->grid_id == kernel->gridId &&
        seen->correlation_id == kernel->correlationId &&
        seen->stream_id == kernel->streamId &&
        seen->device_id == kernel->deviceId) {
      // CUPTI may expose the same launch through KERNEL and
      // CONCURRENT_KERNEL.  Count one physical activity only.
      return;
    }
  }
  if (g_activity_key_count == EZKL_GPU_MAX_ACTIVITY_KEYS) {
    ezkl_set_callback_error();
    return;
  }
  g_activity_keys[g_activity_key_count++] = (ezkl_activity_key){
      .start = kernel->start,
      .end = kernel->end,
      .grid_id = kernel->gridId,
      .correlation_id = kernel->correlationId,
      .stream_id = kernel->streamId,
      .device_id = kernel->deviceId,
  };
  g_kernel_count++;
  g_device_id = kernel->deviceId;
  size_t index = 0;
  for (; index < EZKL_GPU_MAX_KERNELS; ++index) {
    if (g_kernels[index].name[0] == '\0') {
      break;
    }
    if (strncmp(g_kernels[index].name, kernel->name, EZKL_GPU_MAX_NAME - 1) == 0 &&
        g_kernels[index].device_id == kernel->deviceId) {
      g_kernels[index].count++;
      if (kernel->end > g_kernels[index].last_end) {
        g_kernels[index].last_end = kernel->end;
      }
      return;
    }
  }
  if (index == EZKL_GPU_MAX_KERNELS) {
    return;
  }
  strncpy(g_kernels[index].name, kernel->name, EZKL_GPU_MAX_NAME - 1);
  g_kernels[index].name[EZKL_GPU_MAX_NAME - 1] = '\0';
  strncpy(g_kernels[index].category, ezkl_kernel_category(kernel->name),
          EZKL_GPU_MAX_CATEGORY - 1);
  g_kernels[index].category[EZKL_GPU_MAX_CATEGORY - 1] = '\0';
  g_kernels[index].count = 1;
  g_kernels[index].first_start = kernel->start;
  g_kernels[index].last_end = kernel->end;
  g_kernels[index].device_id = kernel->deviceId;
}

static void CUPTIAPI ezkl_buffer_completed(CUcontext context, uint32_t stream_id,
                                            uint8_t *buffer, size_t size,
                                            size_t valid_size) {
  (void)size;
  CUpti_Activity *record = NULL;
  CUptiResult status;
  pthread_mutex_lock(&g_lock);
  while ((status = cuptiActivityGetNextRecord(buffer, valid_size, &record)) ==
         CUPTI_SUCCESS) {
    if (record == NULL) {
      ezkl_set_callback_error();
      break;
    }
    if (atomic_load_explicit(&g_scope_active, memory_order_acquire) &&
        (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
         record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) {
      ezkl_record_kernel((const CUpti_ActivityKernel10 *)record);
    }
  }
  if (status != CUPTI_ERROR_MAX_LIMIT_REACHED && status != CUPTI_SUCCESS) {
    ezkl_set_callback_error();
  }
  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped) ==
      CUPTI_SUCCESS) {
    g_dropped_records += (uint64_t)dropped;
  } else {
    ezkl_set_callback_error();
  }
  pthread_mutex_unlock(&g_lock);
  free(buffer);
}

static int ezkl_cupti_error(CUptiResult result) {
  return result == CUPTI_SUCCESS ? 0 : -(int)result;
}

int ezkl_gpu_telemetry_init(void) {
  if (atomic_load_explicit(&g_initialized, memory_order_acquire)) {
    return 0;
  }
  pthread_mutex_lock(&g_lock);
  if (!atomic_load_explicit(&g_initialized, memory_order_relaxed)) {
    CUptiResult status = cuptiActivityRegisterCallbacks(
        ezkl_buffer_requested, ezkl_buffer_completed);
    if (status != CUPTI_SUCCESS) {
      pthread_mutex_unlock(&g_lock);
      return ezkl_cupti_error(status);
    }
    status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    if (status != CUPTI_SUCCESS) {
      pthread_mutex_unlock(&g_lock);
      return ezkl_cupti_error(status);
    }
    status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (status != CUPTI_SUCCESS) {
      pthread_mutex_unlock(&g_lock);
      return ezkl_cupti_error(status);
    }
    atomic_store_explicit(&g_initialized, 1, memory_order_release);
  }
  pthread_mutex_unlock(&g_lock);
  return 0;
}

static uint64_t ezkl_nonce(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t value = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec;
  value ^= ++g_scope_nonce * UINT64_C(0x9e3779b97f4a7c15);
  return value == 0 ? 1 : value;
}

int ezkl_gpu_scope_begin(const char *scope_name, uint64_t *nonce_out) {
  if (nonce_out == NULL) {
    return -1;
  }
  int status = ezkl_gpu_telemetry_init();
  if (status != 0) {
    return status;
  }
  status = ezkl_cupti_error(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  if (status != 0) {
    return status;
  }
  uint64_t timestamp = 0;
  status = ezkl_cupti_error(cuptiGetTimestamp(&timestamp));
  if (status != 0) {
    return status;
  }
  pthread_mutex_lock(&g_lock);
  memset(g_kernels, 0, sizeof(g_kernels));
  g_kernel_count = 0;
  g_dropped_records = 0;
  g_device_id = UINT32_MAX;
  g_activity_key_count = 0;
  memset(g_activity_keys, 0, sizeof(g_activity_keys));
  g_scope_start = timestamp;
  g_scope_end = UINT64_MAX;
  strncpy(g_scope_name, scope_name == NULL ? "formal" : scope_name,
          sizeof(g_scope_name) - 1);
  g_scope_name[sizeof(g_scope_name) - 1] = '\0';
  uint64_t nonce = ezkl_nonce();
  g_scope_nonce = nonce;
  atomic_store_explicit(&g_callback_error, 0, memory_order_release);
  atomic_store_explicit(&g_scope_active, 1, memory_order_release);
  pthread_mutex_unlock(&g_lock);
  *nonce_out = nonce;
  return 0;
}

int ezkl_gpu_scope_end(uint64_t nonce, char *output, size_t output_size) {
  if (output == NULL || output_size == 0 || nonce == 0 || nonce != g_scope_nonce ||
      !atomic_load_explicit(&g_scope_active, memory_order_acquire)) {
    return -1;
  }
  CUresult cuda_status = cuCtxSynchronize();
  if (cuda_status != CUDA_SUCCESS) {
    atomic_store_explicit(&g_scope_active, 0, memory_order_release);
    return -(int)cuda_status;
  }
  uint64_t timestamp = 0;
  int status = ezkl_cupti_error(cuptiGetTimestamp(&timestamp));
  if (status != 0) {
    atomic_store_explicit(&g_scope_active, 0, memory_order_release);
    return status;
  }
  g_scope_end = timestamp;
  status = ezkl_cupti_error(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  atomic_store_explicit(&g_scope_active, 0, memory_order_release);
  if (status != 0) {
    return status;
  }
  if (atomic_load_explicit(&g_callback_error, memory_order_acquire)) {
    return -2;
  }
  pthread_mutex_lock(&g_lock);
  int written = snprintf(output, output_size,
      "{\"schema\":\"ezkl_cuda_kernel_telemetry_v1\","
      "\"scope\":\"%s\","
      "\"scope_nonce\":\"%" PRIu64 "\","
      "\"scope_start_ns\":%" PRIu64 ",\"scope_end_ns\":%" PRIu64 ","
      "\"kernel_count\":%" PRIu64 ",\"dropped_records\":%" PRIu64 ","
      "\"active_device_id\":%u,\"kernel_names\":[",
      g_scope_name, nonce, g_scope_start, g_scope_end, g_kernel_count, g_dropped_records,
      g_device_id == UINT32_MAX ? 0 : g_device_id);
  if (written < 0 || (size_t)written >= output_size) {
    pthread_mutex_unlock(&g_lock);
    return -3;
  }
  size_t used = (size_t)written;
  int first = 1;
  for (size_t index = 0; index < EZKL_GPU_MAX_KERNELS; ++index) {
    if (g_kernels[index].name[0] == '\0') {
      continue;
    }
    int next = snprintf(output + used, output_size - used,
        "%s{\"name\":\"%s\",\"category\":\"%s\",\"count\":%" PRIu64
        ",\"device_id\":%u}", first ? "" : ",", g_kernels[index].name,
        g_kernels[index].category, g_kernels[index].count,
        g_kernels[index].device_id);
    if (next < 0 || (size_t)next >= output_size - used) {
      pthread_mutex_unlock(&g_lock);
      return -3;
    }
    used += (size_t)next;
    first = 0;
  }
  int tail = snprintf(output + used, output_size - used, "]}");
  pthread_mutex_unlock(&g_lock);
  if (tail < 0 || (size_t)tail >= output_size - used) {
    return -3;
  }
  return 0;
}
