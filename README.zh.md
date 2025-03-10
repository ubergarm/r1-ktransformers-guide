ktransformers 上的 DeepSeek-R1 671B
===
[English Guide Here](README.md)

DeepSeek-R1 671B 本地推理测试
[unsloth/DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)
量化模型在 [ktransformers](https://github.com/kvcache-ai/KTransformers)
框架下的性能表现显示，相比
[llama.cpp](https://github.com/ggml-org/llama.cpp) 在至少配备
16GB+ 显存的 *CUDA* GPU 和最低 64GB 内存（推荐 256-768GB 高带宽内存
[或更高配置的双路 Intel 至强平台]）的系统上，速度提升近 ~2 倍。

现阶段如需进行纯 CPU 推理（无 GPU 加速），建议继续使用 `llama.cpp`。

尽管 ktransformers 目前仍处于早期阶段且未达到生产就绪标准，但其部分优化策略
（下文详述）已展现出超越 llama.cpp 的潜力。

## 快速开始
#### 1. 下载GGUF模型
根据您在输出质量、生成速度和系统内存之间的权衡选择量化模型规格。若不确定，建议直接下载`UD-Q2_K_XL`版本。需注意，完整版DeepS
eek-R1
671B采用的"unsloth动态量化方案"即使在2.51比特/权重(2.51bpw)的量化强度下仍表现优异。根据实测，相比小于700亿参数的小模型，在
重度量化下表现更为稳健。另有体积更小的`UD-IQ`变体，但有用户反馈其速度可能受CPU瓶颈影响。此外，原版R1模型以`fp8`格式发布，适
合GPU运行；若需CPU推理，推荐`Q8_0`量化版本，其性能接近原版。

理想情况下模型应完全载入内存运行。但ktransformers支持类似llama.cpp的`mmap()`内存映射机制，在内存不足时可依托高速NVMe固态硬
盘进行分页交换。以笔者测试平台为例：RTX 3090Ti 24GB显存 + 96GB DDR5-6400内存 + Ryzen 9950X处理器 + PCIe 5.0 T700
2TB固态硬盘组合，可实现约3 token/秒的生成速度。早期测试表明，采用四盘PCIe 5.0 NVMe
RAID0阵列并无优势，瓶颈在于Linux内核页缓存缓冲I/O机制，而非磁盘的原始O\_DIRECT访问速率。
理想情况下模型应完全载入内存运行。但ktransformers支持类似llama.cpp的`mmap()`内存映射机制，在内存不足时可依托…
3090Ti 24GB显存 + 96GB DDR5-6400内存 + Ryzen 9950X处理器 + PCIe 5.0 T700 2TB固态硬盘组合，可实现约3
token/秒的生成速度。早期测试表明，采用四盘PCIe 5.0 NVMe
RAID0阵列并无优势，瓶颈在于Linux内核页缓存缓冲I/O机制，而非磁盘的原始O\_DIRECT访问速率。

| Size | MoE bpw | Model | Quant |
| --- | --- | --- | --- |
| (GiB) | | |
| 212 | 2.51 | `unsloth/DeepSeek-R1-GGUF` | [`UD-Q2_K_XL`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-Q2_K_XL) |
| 298 | 3.00 | `unsloth/DeepSeek-R1-GGUF` | [`Q3_K_M`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q3_K_M) |
| 377 | 4.00 | `unsloth/DeepSeek-R1-GGUF` | [`Q4_K_M`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M) |
| 665 | 8.00 | `unsloth/DeepSeek-R1-GGUF` | [`Q8_0`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q8_0) |

您可直接通过点击各GGUF文件并保存至高速硬盘的目录，或通过如下命令行工具操作：

```bash
# uv pip install huggingface-hub
HF_HUB_ENABLE_HF_TRANSFER=1 \
  huggingface-cli \
    download unsloth/DeepSeek-R1-GGUF \
    --include="DeepSeek-R1-UD-Q2_K_XL/*.gguf" \
    --local-dir ./
```

#### 2. 安装ktransformers
从[预构建的二进制whl文件](https://github.com/ubergarm/ktransformers/releases)安装以启用本地聊天API端点。测试环境为`NVIDIA驱
动版本570.86.1x`和`CUDA版本12.8`。

```bash
# 按需安装系统依赖，例如：
# sudo apt-get install git
# 安装uv工具链 https://docs.astral.sh/uv/getting-started/installation/
# curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule init # submodule "third_party/llama.cpp", [submodule "third_party/pybind11"]
git submodule update
git checkout 7a19f3b
git rev-parse --short HEAD # 应显示 7a19f3b

# 创建虚拟环境(仅托管式Python)
uv venv ./venv --python 3.11 --python-preference=only-managed
source  venv/bin/activate

# 可选实验性使用flashinfer替代triton
# 除非您是已经成功上手的进阶用户，否则暂不建议使用
# 使用以下命令安装：
# $ uv pip install flashinfer-python
# 通过以下链接查询GPU计算架构： `nvidia-smi --query-gpu=compute_cap --format=csv`
# 本示例适用于RTX 3090TI和RTX 6000A
# 设置TORCH_CUDA_ARCH_LIST环境变量：
# $ export TORCH_CUDA_ARCH_LIST="8.6"
# 注意：首次推理会较慢，需要等待JIT编译完成
# 2025-02-27 12:24:22,992 - INFO - flashinfer.jit: Loading JIT ops: batch_mla_attention_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_64
# 2025-02-27 12:24:42,108 - INFO - flashinfer.jit: Finished loading JIT ops: batch_mla_attention_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_64

# 如果你希望自行构建，请跳过后续两步并转到构建说明
uv pip install https://github.com/ubergarm/ktransformers/releases/download/7a19f3b/ktransformers-0.2.2rc1+cu120torch26fancy.amd.ubergarm.7a19f3b.flashinfer-cp311-cp311-linux_x86_64.whl
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.5/flash_attn-2.6.3+cu124torch2.6-cp311-cp311-linux_x86_64.whl

# 待处理项：或可通过调整版本标签实现发布版本号变更:
# $ grep version ktransformers/__init__.py
# __version__ = "0.2.2rc1"
```

#### 3. 运行ktransformers
启动本地聊天API端点。
```bash
# 按需修改参数，使用 `--help` 查看帮助文档
# 支持多GPU配置及通过 `--optimize_config_path` 进行更细粒度的显存卸载设置
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 ktransformers/server/main.py \
    --gguf_path /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/ \
    --model_path deepseek-ai/DeepSeek-R1 \
    --model_name unsloth/DeepSeek-R1-UD-Q2_K_XL \
    --cpu_infer 16 \
    --max_new_tokens 8192 \
    --cache_lens 32768 \
    --total_context 32768 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    --force_think \
    --use_cuda_graph \
    --host 127.0.0.1 \
    --port 8080
```

## 编译ktransformers
本部分接续上述"安装ktransformers"步骤的后续操作。

```bash
# 安装额外编译依赖项，包括CUDA工具链等，例如：
# sudo apt-get install build-essential cmake ...

source venv/bin/activate
uv pip install -r requirements-local_chat.txt
uv pip install setuptools wheel packaging

# 建议跳过可选网站应用，使用`open-webui`或`litellm`等替代方案
cd ktransformers/website/
npm install @vue/cli
npm run build
cd ../..

# 如果拥有充足的CPU核心和内存资源，可显著提升构建速度
# $ export MAX_JOBS=8
# $ export CMAKE_BUILD_PARALLEL_LEVEL=8

# 安装flash_attn
uv pip install flash_attn --no-build-isolation

# 可选实验性使用flashinfer替代triton
# 除非您是已经成功上手的进阶用户，否则暂不建议使用
# 使用以下命令安装：
# $ uv pip install flashinfer-python

# 仅适用于以下情况：
# 配备Intel双路CPU且内存>1TB可容纳两份完整模型内存副本(每路CPU一份副本)
# AMD EPYC NPS0双路平台可能无需此配置？
# $ export USE_NUMA=1

# 安装ktransformers
KTRANSFORMERS_FORCE_BUILD=TRUE uv pip install . --no-build-isolation

# 完成，继续后续步骤！
# *注意* 在最新架构设备上使用最新nvcc编译时可能报错，请参考本文档末尾的错误日志

# 如需重新编译，先执行清理操作：
uv pip uninstall ktransformers
rm -rf ktransformers/ktransformers_ext/build
rm -rf ktransformers/ktransformers_ext/cuda/build
rm -rf ktransformers/ktransformers_ext/cuda/dist
rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info

# 如需构建可分发的Python wheel包(例如在Docker容器内构建以便其他环境使用)
KTRANSFORMERS_FORCE_BUILD=TRUE uv build
# uv pip install ./dist/ktransformers-0.2.2rc1+cu120torch26fancy-cp311-cp311-linux_x86_64.whl
```

## 讨论

#### 基准测试
**硬件配置**

* AMD Ryzen Threadripper PRO 7965WX 24核处理器
* 256GB 内存（约225GB/s内存带宽）
* NVIDIA RTX A6000 48GB显存
* Linux 内核 6.13.0
* Ubuntu 24.04.1 LTS (Noble Numbat)

根据R1修订建议，在聊天线程中设置两个顺序性提示（已移除<thinking>标签）。

提示1
> Count from one to ten in French.

提示2
> Now give pronunciation tips.

| 引擎 | pp | tg |
| --- | --- | --- |
| | 令牌/秒 | 令牌/秒 |
| 1 `ktransformers@94ab2de` | 19.7 | 15.0 |
| 2 `ktransformers@94ab2de` | 63.5 | 15.2 |
| 1 `ktransformers@25c5bdd` | 19.0 | 14.7 |
| 2 `ktransformers@25c5bdd` | 69.1 | 14.8 |
| 1 `llama.cpp@51f311e0` | 18.4 | 8.63 |
| 2 `llama.cpp@51f311e0` | 12.0 | 7.39 |

有趣的是，使用 `--no-use_cuda_graph` 参数的 `ktransformers` 与 llama.cpp 的性能表现接近。更多详情请见下文。

更多基准测试详情请参考：[Level1techs论坛DeepSeek深度讨论帖](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/)

#### 优化分析
ktransformers 实现了多项领先于 llama.cpp 的优化技术，包括：

* **闪存注意力(flash attention)** - 支持量化键值缓存，可能具备上下文迁移能力
* **多头潜在注意力(Multi-head Latent Attention, MLA)** - 集成[SGLang](https://github.com/sgl-project/sglang)框架
* **选择性层卸载** -
基于[路由专家策略](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/deepseek-v2-injection.md#routed-experts-)
* **实验性flashinfer** - 即将推出（商标标注）[flashinfer](https://github.com/flashinfer-ai/flashinfer)
* **Intel至强AMX扩展** - 针对大型双路系统，通过内存双副本策略规避NUMA节点瓶颈

需注意 llama.cpp 也在以下领域进行实验性开发：
* [闪存注意力PR及讨论](https://github.com/ggml-org/llama.cpp/pull/11557)
* [MLA实现](https://github.com/ggml-org/llama.cpp/pull/11446)
* [选择性层卸载](https://github.com/ggml-org/llama.cpp/pull/11397#issuecomment-2645973828)

#### CUDA Graphs

虽然可以使用 `--optimize_config_path` 将额外的专家模块卸载到显存中，但当前必须搭配
`--no-use_cuda_graph`，这会大幅降低性能。因此，若显存容量超出所需上下文支持的最小需求，并不会带来额外优势。

> 目前，在GPU上执行专家模块会与CUDA Graph冲突。禁用CUDA
Graph将导致显著性能下降。因此，除非拥有充足显存（例如DeepSeek-V3/R1单层专家模块需至少5.6GB显存），否则不建议启用此功能。我
们正在积极优化。注：KExpertsTorch未经测试。-[ktransformers
FAQ](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/FAQ.md#q-if-i-got-more-vram-than-the-models-requirement-how-can-i-fully-utilize-it)

引擎版本：`ktransformers@94ab2de`

配置规则：
- 若 `# Offload = 0`，启用 `--use_cuda_graph`
- 若 `# Offload > 0`，启用 `--no-use_cuda_graph`

| 输入 | 卸载数量 | 显存 | 预处理速率 | 生成速率 |
| --- | --- | --- | --- | --- |
| | 专家 | GB | tok/秒| tok/秒 |
| 1 | 0 `use_cuda_graph` | 14 | 21.0 | 15.1 |
| 2 | 0 `use_cuda_graph` | 14 | 74.8 | 15.2 |
| 1 | 0 `no-use_cuda_graph` | 14 | 20.35 | 9.0 |
| 2 | 0 `no-use_cuda_graph` | 14 | 69.2 | 9.0 |
| 1 | 2 `no-use_cuda_graph` | 26 | 26.9 | 8.2 |
| 2 | 2 `no-use_cuda_graph` | 26 | 66.9 | 8.3 |
| 1 | 4 `no-use_cuda_graph` | 37 | 19.4 | 7.9 |
| 2 | 4 `no-use_cuda_graph` | 37 | 80.4 | 7.9 |
| 1 | 6 `no-use_cuda_graph` | 48 | oom | oom |
| 2 | 6 `no-use_cuda_graph` | 48 | oom | oom |

#### 磁盘I/O性能
在 RTX 3090Ti 24GB 显存 + 96GB DDR5@88GB/s + Ryzen 9950X + PCIe 5.0 T700 2TB NVMe
测试平台上，即使内存不足，该模型仍能实现约 3.2 token/秒 的生成速度。值得注意的是，ktransformers 相比 llama.cpp
更能压满磁盘读取I/O操作数，并触发`kswapd0`进程高负载（测试时未启用Linux交换分区）。

然而，在使用四盘 T705 4TB NVMe 组建的 `mdadm` RAID0 条带阵列测试时，两种推理引擎相较单盘方案均未表现出性能提升。

```python
# ktransformers内存映射实现
ktransformers/util/custom_gguf.py: self.file_data_map[file_name] = np.memmap(file_name, mode = 'r')

# llama.cpp内存映射对比
src/llama-mmap.cpp: addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
gguf-py/gguf/gguf_reader.py: self.data = np.memmap(path, mode = mode)
```

#### Intel 至强 AMX 扩展支持
检查您的Intel至强处理器及Linux内核是否支持AMX扩展：
```bash
$ lscpu | grep amx
amx_bf16 avx512_fp16 amx_tile amx_int8
```

若检测到支持，建议尝试[自定义编译的ktransformers二进制版本](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html#v03-showcase)

**请注意：此版本基于存在API端点缺陷的早期代码构建，因此仅适用于本地聊天演示。**

存在一个[已知问题及临时解决方案](https://github.com/kvcache-ai/ktransformers/issues/320#issuecomment-2662274450)：
需启用环境变量 `ARCH_REQ_XCOMP_PERM`，否则即使已配置AMX扩展指令集，仍会触发报错。

你可能还需要[使用完整的bf16模型进行在线量化到fp8/fp4，以适配Intel AMX](https://github.com/kvcache-ai/ktransformers/issues/617#issuecomment-2676999515)。（该链接指向GitHub上关于此技术实现的详细讨论）

#### 长上下文配置

为支持≥20K的上下文长度，推荐使用 [矩阵吸收MLA配置](https://github.com/kvcache-ai/ktransformers/commit/03f8bc9f79d9b3915bce73ab13174f91e53a79d9)，可显著缩减 键值缓存（KV Cache）的显存占用。

#### 编译错误日志
在全新Arch Linux系统上编译ktransformers时遇到编译错误，但Ubuntu 22.04环境正常。尝试先后使用Python
3.11和3.12环境（包括升级至最新cu128 nightly版torch）均未解决。相关错误日志与[ktransformers GitHub Issues
#217](https://github.com/kvcache-ai/ktransformers/issues/217)描述可能相同。

推测问题源于过新的`nvcc --version`：

* `Build cuda_12.0.r12.0/compiler.32267302_0` 兼容CUDA 12.8环境
* `Build cuda_12.8.r12.8/compiler.35404655_0` *版本过高*将导致错误
* `Build cuda_12.8.r12.8/compiler.35404655_0` *版本过高*将导致错误

#### open-webui 集成
使用[open-webui](https://github.com/open-webui/open-webui)连接ktransformers API端点的快速配置指南

```bash
mkdir open-webui
cd open-webui
uv venv ./venv --python 3.11
source venv/bin/activate
uv pip install open-webui
```

将以下内容复制到`./go.sh`脚本并执行`chmod u+x go.sh`授权：
```bash
#!/usr/bin/env bash

source venv/bin/activate

# open-webui 不原生支持HOST和PORT环境变量，需手动传递参数
# https://docs.openwebui.com/getting-started/env-configuration/#port
# 若open-webui运行异常，可执行`rm -rf ./data`清除数据后重启服务并清理浏览器缓存

# https://docs.openwebui.com/getting-started/env-configuration/
export DATA_DIR="$(pwd)/data"
export ENABLE_OLLAMA_API=False
export ENABLE_OPENAI_API=True
export OPENAI_API_KEY="dont_change_this_cuz_openai_is_the_mcdonalds_of_ai"
export OPENAI_API_BASE_URL="http://127.0.0.1:8080/v1" # <--- 需与ktransformers/llama.cpp的API配置匹配
#export DEFAULT_MODELS="openai/foo/bar" # <--- 保留注释，此参数用于`litellm`接入
export WEBUI_AUTH=False
export DEFAULT_USER_ROLE="admin"
export HOST=127.0.0.1
export PORT=3000 # <--- open-webui网页服务端口

# 如果仅加载了R1模型，可通过禁用以下功能节省时间：
#   * 标签生成
#   * 自动补全
#   * 标题生成
# https://github.com/kvcache-ai/ktransformers/issues/618#issuecomment-2681381587
export ENABLE_TAGS_GENERATION=False
export ENABLE_AUTOCOMPLETE_GENERATION=False
# 或许目前需手动在界面禁用该功能？？？
export TITLE_GENERATION_PROMPT_TEMPLATE=""


open-webui serve \
  --host $HOST \
  --port $PORT

# 在浏览器中访问显示的URL:端口
```

## 参考文献
* [双路 Intel Xeon NUMA 节点讨论](https://github.com/ggml-org/llama.cpp/discussions/12088)
* [双路 AMD Epyc NUMA 节点讨论](https://github.com/ggml-org/llama.cpp/discussions/11733)
* [kvcache-ai/KTransformers](https://github.com/kvcache-ai/KTransformers)
* [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
* [本指南历史版本(含更多技术笔记，结构稍乱)](https://github.com/ubergarm/r1-ktransformers-guide/blob/v0.1/README.md)
* [v0.3版临时解决方案Docker指南](https://github.com/txg1550759/ktransformers-v0.3-docker.git)
* [Level1techs论坛DeepSeek深度讨论帖](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/)
