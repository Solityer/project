# GPU 后端构建尝试

## 本机环境

| 项 | 值 |
| --- | --- |
| 主机架构 | `aarch64` |
| GPU | `NVIDIA GB10` |
| GPU 架构 | `Blackwell` |
| GPU 数量 | `1` |
| 驱动版本 | `580.142` |
| CUDA 工具链 | `13.0, V13.0.88` |
| `nvcc` | `/usr/local/cuda/bin/nvcc` |
| `CUDA_HOME` | 未设置 |

检查命令：

```bash
nvidia-smi
nvidia-smi -q
nvcc --version
which nvcc
echo $CUDA_HOME
uname -m
```

## 源码获取尝试

HTTPS：

```bash
git clone --depth 1 --branch main https://github.com/ingonyama-zk/icicle-cuda-backend.git .vendor_local/icicle-cuda-backend-main
```

结果：

```text
Cloning into '.vendor_local/icicle-cuda-backend-main'...
fatal: could not read Username for 'https://github.com': No such device or address
```

SSH：

```bash
git clone --depth 1 --branch main git@github.com:ingonyama-zk/icicle-cuda-backend.git .vendor_local/icicle-cuda-backend-main-ssh
```

结果：

```text
Cloning into '.vendor_local/icicle-cuda-backend-main-ssh'...
ERROR: Repository not found.
fatal: Could not read from remote repository.
```

## 本地 CUDA backend 构建入口

命令：

```bash
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle \
  -B build_gpu_backend/icicle_cuda_local \
  -DCMAKE_BUILD_TYPE=Release \
  -DCURVE=bn254 \
  -DCUDA_BACKEND=local \
  -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_local \
  -DBUILD_TESTS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
```

结果：

```text
-- Adding CUDA backend from local path: icicle/backend/cuda
CMake Error at cmake/backend_include.cmake:6 (add_subdirectory):
  add_subdirectory given source "backend/cuda" which is not an existing directory.
```

失败原因：当前 Icicle checkout 只有 `backend/cpu`，没有 `backend/cuda` 源码目录。

## CMake 远程 CUDA backend 获取入口

命令：

```bash
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle \
  -B build_gpu_backend/icicle_cuda_fetch_main \
  -DCMAKE_BUILD_TYPE=Release \
  -DCURVE=bn254 \
  -DCUDA_BACKEND=main \
  -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_fetch_main \
  -DBUILD_TESTS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
```

结果：

```text
-- Fetching cuda backend from git@github.com:ingonyama-zk/icicle-cuda-backend.git:main
ERROR: Repository not found.
fatal: Could not read from remote repository.
CMake Error at ... cuda_backend-populate-gitclone.cmake:39 (message):
  Failed to clone repository:
  'git@github.com:ingonyama-zk/icicle-cuda-backend.git'
```

失败原因：CMake 远程 backend 源码仓库需要访问权限，本机当前凭据无法读取。

## 系统 aarch64 后端搜索

命令：

```bash
find /usr /usr/local /opt /home/pzh -type f -name "libicicle_backend_cuda*.so" 2>/dev/null | head -200
```

结果只发现 Cargo checkout 中的库：

```text
/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda/libicicle_backend_cuda_device.so
/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/01c8884/icicle/backend/cuda/libicicle_backend_cuda_device.so
/home/pzh/.cargo/git/checkouts/halo2-9199b7623b70ff0b/01c8884/icicle/backend/cuda/libicicle_backend_cuda_device.so
```

对应四个关键库的 `file` 输出均为：

```text
ELF 64-bit LSB shared object, x86-64
```

未找到 `aarch64` 的 `libicicle_backend_cuda*.so`。

## 结论

本机 CUDA 13.0 工具链和 NVIDIA GB10 设备可见，但缺少可构建或可加载的 `aarch64` Icicle CUDA backend。当前无法在本机真实进入 GPU prove/verify。GPU route 保持 `blocked`，不得标记为 `verified`，也不得回退 CPU 冒充 GPU。
