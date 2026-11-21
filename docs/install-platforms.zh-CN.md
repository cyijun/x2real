# 分平台安装说明

本文整理了在 4 类机器上部署 X2Real（ManaEnv + Isaac Sim 6.0.1.0）时遇到的问题和处理办法。安装入口仍然是
仓库根目录的 `install.sh`，通用步骤见 [README](../README.zh-CN.md#install-and-evaluation-zh)。

> 文中的速度、占用等数字都是**一次实测**，测量条件写在对应位置，只作量级参考。“步/秒”指评测时仿真环境每秒推进的
> 控制步数；评测按 30 步/秒折算实时倍率。

## 目录

- [通用事项](#通用事项)
- [DGX Spark（aarch64，GB10，驱动 580）](#dgx-sparkaarch64gb10驱动-580)
- [普通 x86 Linux 服务器（例：8×L20，驱动 570，无 sudo）](#普通-x86-linux-服务器例8l20驱动-570无-sudo)
- [AutoDL 容器（例：RTX 5090，驱动 595）](#autodl-容器例rtx-5090驱动-595)
- [WSL2（不支持带相机评测）](#wsl2不支持带相机评测)

## 通用事项

### 前置条件

- Linux（x86_64 或 aarch64）、NVIDIA GPU 和驱动，`nvidia-smi` 可用。
- Conda（推荐 Miniforge）或 uv；安装器会创建 Python 3.12 环境。
- `git` 和 `git-lfs`（Arena 仓库启用了 LFS hook，找不到 git-lfs 时 checkout 等 git 操作会报错）。
- `cmake` 在 `PATH` 上，或者有 root / 免密 sudo（见下文“cmake 与 sudo”）。
- 能访问 GitHub、PyPI、`pypi.nvidia.com`、PyTorch 下载站和 Hugging Face（或其镜像）。

### 安装命令

```bash
export OMNI_KIT_ACCEPT_EULA=YES     # 无人值守安装必须设置，见下文
bash ./install.sh --dry-run         # 只打印计划和会执行的命令
source ./install.sh                 # 默认创建/使用 conda 环境 manaenv
```

### 本 fork 的安装器额外处理了什么

| 问题 | 处理 |
| --- | --- |
| aarch64 上 Isaac Sim 运行时 wheel 的哈希、libgomp 预加载 | 按架构校验 NVIDIA 原始 wheel；aarch64 上把 `LD_PRELOAD=<libgomp>` 写进环境 |
| 驱动低于 R580 时 PyPI 的 `warp-lang 1.19.0`（CUDA 13 构建）无法使用 | 安装末尾自动换成 NVIDIA 在 GitHub Release 发布的 `warp_lang-1.19.0+cu12`（校验 sha256），见 [x86 服务器](#普通-x86-linux-服务器例8l20驱动-570无-sudo) 一节 |
| IsaacLab 安装时默认设 `PIP_FIND_LINKS=https://py.mujoco.org/`，国内每个包卡约 75 秒 | 调用 Arena/IsaacLab 安装器时传入空的 `PIP_FIND_LINKS`（需要的 `mujoco~=3.11.0` 在 PyPI 上就有 x86_64 和 aarch64 的 wheel）。如果你确实需要 find-links，自己 `export PIP_FIND_LINKS=...` 后再运行，安装器会原样传下去 |
| conda-forge 的 git-lfs 3.8.x 改写 Arena 仓库 `.githooks/*` 里的 LFS hook，重跑安装器报 “Arena checkout is dirty” | Arena 干净检出检查忽略 `.githooks/` |
| Arena 安装器第 8 步往 `~/.bashrc` 追加 `# IsaacLab Arena Configuration` 一段（错误的 `ISAACLAB_PATH=${HOME}/Desktop/new/IsaacLab`、PS1、`ll`/`..` 别名） | 安装前若没有这段，安装后只删除 Arena 追加的这 12 行，其余内容（包括安装期间你对 .bashrc 的其他改动）保留；删除前把原文件复制到 `~/.bashrc.manaenv-backup.<时间>`。如果追加内容和预期形状不一致，不做改动、只打印警告 |
| 没有 `cmake` 时 IsaacLab 会执行 `sudo apt-get update` / `apt-get install cmake build-essential` | Step 3 之前检查：找不到 cmake、又不是 root、也没有免密 sudo 时直接报错退出，并给出安装 cmake 的命令 |

### EULA

无人值守（非交互终端）安装时必须先 `export OMNI_KIT_ACCEPT_EULA=YES`。否则安装第 5 步的导入验证会因为 Kit 等待
EULA 输入而失败，报 `Unable to bootstrap inner kit kernel: EOF when reading a line`。评测时同样需要。

### cmake 与 sudo

IsaacLab 的安装代码（`isaaclab/cli/commands/install.py` 的 `_install_system_deps`）在 `PATH` 里找不到 `cmake` 时，
会以非 root 身份执行 `sudo apt-get update` 和 `sudo apt-get install cmake build-essential`。没有 sudo 的机器请先装好 cmake：

```bash
sudo apt-get install -y cmake build-essential   # 有 sudo
python3 -m pip install --user cmake             # 没有 sudo，确认 ~/.local/bin 在 PATH 上
```

注意 cmake 要在激活目标 conda 环境之后仍然可见：装在 conda `base` 环境里的 cmake，`conda activate manaenv` 后通常就不在
`PATH` 上了。

### 资产下载与还原

资产在 Hugging Face 数据集 [`x-square-robot/x2real-assets`](https://huggingface.co/datasets/x-square-robot/x2real-assets)，
由 107 个独立的 TAR 分片组成，还原后约 42G（74,423 个文件）。

- **官方方式**：用数据集里的 `scripts/download_x2real_assets.py` 先下载全部分片、校验，再还原。分片和还原结果同时存在，
  官方说明约 88.3 GB，实测峰值约 84G。
- **边下边解**：逐个分片“下载 → 按 `SHA256SUMS` 校验 → `tar -xf` 解到资产目录 → 删除分片”，峰值约 43G。分片是独立归档，
  不要用 `cat` 拼接。这种方式绕过了官方还原脚本的整树校验，解完后建议对照 `bundle/inventory.jsonl` 抽查。

还原后用 `export MANAENV_ASSET_ROOT=/path/to/x2real-assets` 指向资产目录。

### 贴图串用（UJITSO 贴图缓存）

- **现象**：物体显示成别的资产的贴图，不报错，不同机器上串得还不一样。
- **根因**：Kit 110.1（Isaac Sim 6.0）的 UJITSO 贴图缓存用源文件的内容哈希做键，但文件 mtime 为 0 时跳过哈希。公开资产分片
  还原出的文件 mtime 全是 0（共 56,732 个图片文件受影响），于是处理参数相同的贴图都落到同一个缓存键上，谁先写进持久缓存，
  其他贴图就都显示成它。
- **已有修复**：提交 `a3a7a31` 在评测和预览入口启动 Kit 前注入 `--/UJITSO/textures=false`，关掉这个缓存；显式传入的
  `--/UJITSO/textures=...` 不会被覆盖。
- **可选的治本办法**：让资产文件的 mtime 不再是 0，Kit 就会正常计算内容哈希：

  ```bash
  find <资产目录> -type f ! -newermt '1970-01-02' -exec touch -c -d '2026-01-01' {} +
  ```

  代价是每台机器、每次重新还原资产后都要做一次；之后用 rsync 同步资产时，因为 mtime 变了，rsync 会把这些文件重新比对一遍。

### 网络

- 评测启动时会预加载 Franka 等机器人模型，这些文件要从 NVIDIA 的 S3 下载；网络不通时会卡在启动阶段。需要代理的机器请设置
  `http_proxy`/`https_proxy`，同时设置 `no_proxy=localhost,127.0.0.1,::1`（以及大写的 `NO_PROXY`），让到本机 Policy Space 服务的
  请求不走代理。
- 在国内访问 `pypi.nvidia.com` 会被跳转到 `pypi.nvidia.cn`，属正常现象。

## DGX Spark（aarch64，GB10，驱动 580）

**前置条件**：DGX OS 自带的驱动（实测 580.178.04）、Miniforge、git-lfs。

**安装**：按通用命令执行即可。aarch64 相关的处理（Isaac Sim 运行时 wheel 的 aarch64 哈希、`libgomp` 预加载）已经在本 fork
的安装器里；驱动是 580，不会触发 warp 替换。

**已知问题**：

- **Kit 启动偶发挂死**：日志停在 `Using Mana headless camera experience`，进程 CPU≈0，所有线程都在 futex 上等待。实测概率
  10%～50%。处理办法是杀掉整个进程树重试；批量跑评测时可以加一个看门狗：首次推理之前日志 90 秒没有增长就杀掉重来。
- **吞吐**：单个仿真就会打满 GPU，同时多开几个仿真不会增加总吞吐。

**实测数据**（一次实测，`artixon_arm6a_tabletop` 看板的 ID 评测，单环境、带相机）：仿真约 4.1 步/秒，约为实时的 0.14 倍。

## 普通 x86 Linux 服务器（例：8×L20，驱动 570，无 sudo）

**前置条件**：

- 先装好 `cmake`（见[cmake 与 sudo](#cmake-与-sudo)），否则安装器会在 Step 3 之前报错退出。
- 能访问 GitHub Release（下载 CUDA 12 版 warp，x86_64 约 165 MB）。

**安装**：按通用命令执行。

**驱动低于 R580 时的 warp 问题**：

PyPI 上的 `warp-lang 1.19.0` 是 CUDA 13 构建，需要 R580 及以上驱动。L20 上驱动是 570.211.01（最高支持 CUDA 12.8），评测启动报：

```text
Insufficient CUDA driver version. The minimum required CUDA driver version is 13.0, but the installed CUDA driver version is 12.8
ValueError: Invalid device identifier: cuda:0
```

本 fork 的安装器在 Step 4 末尾读取 `nvidia-smi` 的驱动主版本（多块卡取最小值），低于 580 且装的是 PyPI 的 `warp-lang 1.19.0`
时，会下载 NVIDIA 在 [GitHub Release](https://github.com/NVIDIA/warp/releases/tag/v1.19.0) 发布的同版本 CUDA 12 构建，校验
sha256 后执行 `pip install --no-deps --force-reinstall`：

| 架构 | 文件 | sha256 |
| --- | --- | --- |
| x86_64 | `warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl` | `f95d9581ed12663df272e45a50101b171e143fde1ef7b2433a8bfc9fba096598` |
| aarch64 | `warp_lang-1.19.0+cu12-py3-none-manylinux_2_34_aarch64.whl` | `b3d0222132811b1db8c5c5d1d04bdc92923b77b7df8556d665116569a361a5d2` |

- 下载缓存在 `${XDG_CACHE_HOME:-~/.cache}/manaenv/warp-cu12/`。机器上下载不了 GitHub 时，可以把上面的文件手工放进这个目录，
  校验通过就不再下载。
- 替换后的版本是 `1.19.0+cu12`，依赖它的包（`newton`、`mujoco-warp`、`newton-actuators`）都只要求 `warp-lang>=...`，
  `pip check` 不受影响。
- 已装好的旧环境可以执行 `source ./install.sh --skip-sources` 补上这一步，或者手工
  `pip install --no-deps --force-reinstall <上面的 wheel>`。
- 容器里看不到 `nvidia-smi` 时，可以用 `MANAENV_NVIDIA_DRIVER_VERSION=570.211.01` 告诉安装器驱动版本。
- 如果以后 warp-lang 版本变了，安装器只打印警告、不自动替换，需要到 Release 页手工下载对应的 `+cu12` wheel。

**共享机器上只占一块 GPU**：

```bash
export CUDA_VISIBLE_DEVICES=N
python manaenv/scripts/x2real_eval.py <常规参数> --enable_cameras \
    --device cuda:0 \
    --/renderer/activeGpu=N \
    --/renderer/multiGpu/enabled=false
```

- `CUDA_VISIBLE_DEVICES=N` 之后，CUDA 里这块卡的序号就是 0，所以 `--device` 填 `cuda:0`。
- `--/renderer/activeGpu` 填的是 **Vulkan 枚举序号**。Vulkan 不受 `CUDA_VISIBLE_DEVICES` 影响，仍然能看到全部卡，所以这里要填
  这块卡自己的序号 N，而不是 0；填 0 会报 `No device could be created`。这次实测填的是与 `CUDA_VISIBLE_DEVICES` 相同的 N。
  Vulkan 的枚举顺序不保证和 `nvidia-smi` 一致，换机器时请对照 Kit 日志里列出的 GPU 确认。
- 不设 `CUDA_VISIBLE_DEVICES` 时，进程会在每块卡上各占约 300MiB 显存。

**其他已知问题**：

- 共享的 `/tmp/https` 目录被别的用户先创建后，预加载 Franka 等模型会报权限错误。把 `TMPDIR` 指到自己的目录即可，例如
  `export TMPDIR=$HOME/tmp && mkdir -p "$TMPDIR"`。

**实测数据**（一次实测，带相机评测、单环境，同一块 GPU 当时也在被其他用户的任务使用）：仿真约 5.4 步/秒（折算约为实时的 0.18 倍）。

## AutoDL 容器（例：RTX 5090，驱动 595）

**前置条件**：

1. 系统库（容器默认是 root）：

   ```bash
   apt-get update && apt-get install -y libegl1 libxt6 libglu1-mesa
   ```

   - 缺 `libEGL.so.1` 时，NVIDIA 的 Vulkan 驱动拒绝创建实例，报
     `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`，只能枚举到 llvmpipe。
   - 缺 `libXt.so.6` 时，Kit 加载 `libMaterialXRenderHw.so` 失败。

2. 系统盘只有 30G，环境、缓存、资产都放数据盘 `/root/autodl-tmp`：

   ```bash
   D=/root/autodl-tmp
   export CONDA_ENVS_DIRS=$D/conda/envs CONDA_PKGS_DIRS=$D/conda/pkgs
   export PIP_CACHE_DIR=$D/cache/pip UV_CACHE_DIR=$D/cache/uv
   export HF_HOME=$D/cache/huggingface XDG_CACHE_HOME=$D/cache TMPDIR=$D/tmp
   mkdir -p "$CONDA_ENVS_DIRS" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"
   ```

   这些变量在之后每次安装、评测的 shell 里都要设置（可以写进自己的脚本）。

3. 网络：

   - `source /etc/network_turbo` 打开 AutoDL 的学术加速，访问 GitHub、PyTorch 下载站、`pypi.nvidia.cn` 都快很多。
     它是代理，记得同时设置 `no_proxy=localhost,127.0.0.1,::1`（见[网络](#网络)）。
   - pip 默认源用阿里云镜像：`pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/`。
   - Hugging Face 用 `HF_ENDPOINT=https://hf-mirror.com`。hf-mirror 实际会跳转到海外的 xethub CDN，单路只有 2～5MB/s；
     多路并行下载分片能到 30～47MB/s，例如：

     ```bash
     HF_ENDPOINT=https://hf-mirror.com hf download x-square-robot/x2real-assets --repo-type dataset \
         --include 'shards/*' --max-workers 8 --local-dir $D/x2real-assets-download
     ```

**安装**：设好上面的变量后按通用命令执行。驱动是 595，不会触发 warp 替换；容器里是 root，IsaacLab 在缺 cmake 时会直接用
`apt-get` 安装。

**实测数据**（一次实测，带相机评测、单环境，16 核 CPU 配额）：

- 仿真约 11 步/秒，约为实时的 0.37 倍。瓶颈在 CPU：进程吃到约 1300%（16 核配额），GPU 平均利用率约 59%。
- 数据盘占用约 90G：conda 环境 32G、资产 42G、pip 缓存 16G（装完可以清掉）。

## WSL2（不支持带相机评测）

WSL2 里没有 NVIDIA 的 Linux Vulkan 驱动，只有 Mesa 的 Dozen（Vulkan 转 D3D12）。Dozen 缺少
`VK_KHR_ray_tracing_pipeline`、`VK_KHR_acceleration_structure`、`VK_KHR_ray_query`，Kit 的 RTX 渲染器报
`Failed to create any GPU devices`。NVIDIA 论坛上官方也说明 Isaac Sim 暂不支持 WSL2。

安装过程本身能跑（实测同样遇到了上面 `~/.bashrc` 被追加的问题，本 fork 已处理），但带相机的评测起不来，请改用原生 Linux 机器。
