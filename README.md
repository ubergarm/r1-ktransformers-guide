DeepSeek-R1 671B on ktransformers
===
Local inferencing of DeepSeek-R1 671B
[unsloth/DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)
quants on [ktransformers](https://github.com/kvcache-ai/KTransformers)
shows almost ~2x faster performance than
[llama.cpp](https://github.com/ggml-org/llama.cpp) on systems with at
least one 16GB+ VRAM *CUDA* GPU and bare minimum of 64GB RAM (preferrably
256-768GB high bandwidth memory [or more for dual socket Intel Xeon]).

For now if you need CPU *only* inferencing without GPU then stick with `llama.cpp`.

While ktransformers is still rough around the edges and not production ready imo,
it seems to have implemented some optimizations ahead of llama.cpp discussed below.

## Quick Start
#### 1. Download GGUFs
Pick a quantized model size trading off output quality, generation speed,
and system RAM based on your preference. If you don't know for sure, just
get the `UD-Q2_K_XL` version. Keep in mind these "unsloth dynamic quants"
of the full DeepSeek-R1 671B perform fairly well even at 2.51bpw sizes,
and in my opinion, hold up better under heavy quantization than smaller <
70B models. There are smaller `UD-IQ` flavors, though some have reported
being slower possibly due to CPU bottleneck? Finally, the original R1
was released in `fp8` format suitable for GPUs, and the best CPU option
is the `Q8_0` quantization which should provide reasonable performance
compared to the original.

Ideally the entire model size will fit into RAM. However, ktransformers
does support `mmap()` similar to `llama.cpp` so in a pinch it will page
data off a fast NVMe drive even if you don't have enough RAM. My 3090TI
24GB VRAM + 96GB DDR5-6400 + 9950X + Gen 5 T700 2TB NVMe rig hits over 3
tok/sec generation like this. In early testing there is no advantage to
using a quad Gen 5 NVMe RAID0 striped array as the bottleneck appears
to be Linux Kernel Page Cache buffered i/o and not raw `O_DIRECT`
disk speeds.

| Size | MoE bpw | Model | Quant |
| --- | --- | --- | --- |
| (GiB) | | |
| 212 | 2.51 | `unsloth/DeepSeek-R1-GGUF` | [`UD-Q2_K_XL`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-Q2_K_XL) |
| 298 | 3.00 | `unsloth/r1-1776-GGUF` | [`UD-Q3_K_XL`](https://huggingface.co/unsloth/r1-1776-GGUF/tree/main/UD-Q3_K_XL) |
| 377 | 4.00 | `unsloth/r1-1776-GGUF` | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/r1-1776-GGUF/tree/main/UD-Q4_K_XL) |
| 665 | 8.00 | `unsloth/DeepSeek-R1-GGUF` | [`Q8_0`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q8_0) |

You can simply click each GGUF file and save it in a directory on your fastest drive or use the cli tool like so:

```bash
# uv pip install huggingface-hub
HF_HUB_ENABLE_HF_TRANSFER=1 \
  huggingface-cli \
    download unsloth/DeepSeek-R1-GGUF \
    --include="DeepSeek-R1-UD-Q2_K_XL/*.gguf" \
    --local-dir ./
```

#### 2. Install ktransformers
Install from [pre-built binary whl files](https://github.com/ubergarm/ktransformers/releases) for serving local chat API endpoint.
Tested on `NVIDIA Driver Version 570.86.1x` and `CUDA Version: 12.8`.
```bash
# install system dependencies as needed e.g.
# sudo apt-get install git
# install uv https://docs.astral.sh/uv/getting-started/installation/
# curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git checkout 25c5bdd
git rev-parse --short HEAD # 25c5bdd
uv venv ./venv --python 3.11 --python-preference=only-managed
source  venv/bin/activate

# If you would prefer to build it yourself, skip next two and go to build instructions
uv pip install https://github.com/ubergarm/ktransformers/releases/download/25c5bdd/ktransformers-0.2.1.post1+cu120torch26fancy-cp311-cp311-linux_x86_64.whl
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.5/flash_attn-2.6.3+cu124torch2.6-cp311-cp311-linux_x86_64.whl
```

#### 3. Run ktransformers
Start up local chat API endpoint.
```bash
# Modify as needed using `--help` for documentation
# MultiGPU and more offload customization available with `--optimize_config_path`
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

## Build ktransformers
This picks up where the above `Install ktransformers` instructions leave off above.
```
# install additional build dependencies including CUDA toolkit etc. e.g.
# sudo apt-get install build-essentials cmake ...

source venv/bin/activate
uv pip install -r requirements-local_chat.txt
uv pip install setuptools wheel packaging

# Recommend skipping optional website app and use `open-webui` or `litellm` etc.
cd ktransformers/website/
npm install @vue/cli
npm run build
cd ../..

# Install flash_attn
MAX_JOBS=4 uv pip install flash_attn --no-build-isolation

# ONLY IF you have Intel dual socket and >1TB RAM to hold 2x copies of entire model in RAM (one copy per socket)
# Dual socket AMD EPYC NPS0 probably makes this not needed?
# $ export USE_NUMA=1

# Install ktransformers
MAX_JOBS=4 KTRANSFORMERS_FORCE_BUILD=TRUE uv pip install . --no-build-isolation

# DONE, Continue below!
# *NOTE* You may get build errors on very new ARCH with latest nvcc see build error log at bottom of this document.

# If you want to rebuild again, first make clean like so
uv pip uninstall ktransformers
rm -rf ktransformers/ktransformers_ext/build
rm -rf ktransformers/ktransformers_ext/cuda/build
rm -rf ktransformers/ktransformers_ext/cuda/dist
rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info

# if you want to build a distributable python cheese .whl (e.g. from inside a Dockerfile to use elsewhere)
MAX_JOBS=4 KTRANSFORMERS_FORCE_BUILD=TRUE uv build
# uv pip install ./dist/ktransformers-0.2.1.post1+cu120torch26fancy-cp311-cp311-linux_x86_64.whl
```

## Discussions

#### Benchmarks
Short one shot prompt comparison with roughly similar configurations.

| engine | pp | tg |
| --- | --- | --- |
| | tok/sec | tok/sec |
| `ktransformers@25c5bdd` | 19.0 | 14.7 |
| `llama.cpp@51f311e0` | 18.4 | 8.63 |

**Hardware**

* AMD Ryzen Threadripper PRO 7965WX 24-Cores
* 256 GB RAM (~225GB/s memory bandwidth)
* NVIDIA RTX A6000 with 48 GB VRAM
* Linux Kernel 6.13.0
* Ubuntu 24.04.1 LTS (Noble Numbat)

Check the [level1techs forum DeepSeek Deep Dive thread](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/) for more benchmarks and details.

#### Optimizations
ktransformers implements some optimizations ahead of llama.cpp e.g.

* flash attention - can quantize kv cache and possibly supports context-shift?
* Multi-head Latent Attention (MLA) with [SGLang](https://github.com/sgl-project/sglang)
* [Selective layer offload](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/deepseek-v2-injection.md#routed-experts-)
* experimental [flashinfer](https://github.com/flashinfer-ai/flashinfer) Coming Soon :TM:
* Intel Xeon AMX extensions and copy model into RAM *twice* for big dual socket systems (as cross NUMA nodes is bottleneck)

Keep in mind llama.cpp has some as experimental branches going as well including:
* [flash attention PR and ongoing discussion](https://github.com/ggml-org/llama.cpp/pull/11557)
* [MLA](https://github.com/ggml-org/llama.cpp/pull/11446)
* [selective layer offload](https://github.com/ggml-org/llama.cpp/pull/11397#issuecomment-2645973828)

#### Disk Read IOPs

On my local 3090TI 24GB VRAM + 96GB DDR5@88GB/s + 9950X + PCIe Gen 5
T700 2TB NVMe hits around 3.2 tok/sec generation with this same model
despite not having enough RAM. Interestingly ktransformers is saturating
read IOPs more than llama.cpp and pegging `kswapd0` (without any Linux swap enabled).

However when testing on a quad 4TB T705 NVMe `mdadm` RAID0 stripped array
I didn't notice any gains over a single NVMe for either inference engine
on that machine.

```
ktransformers/util/custom_gguf.py: self.file_data_map[file_name] = np.memmap(file_name, mode = 'r')

# vs

src/llama-mmap.cpp: addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
gguf-py/gguf/gguf_reader.py: self.data = np.memmap(path, mode = mode)
```

#### Intel Xeon AMX Extensions
Check to see if your Intel Xeon processor and Linux kernel are new enough to support AMX extensions:
```
$ lscpu | grep amx
amx_bf16 avx512_fp16 amx_tile amx_int8
```

If so, you might want to try [ktransformers custom compiled binary release](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html#v03-showcase)

#### Build Error Logs
I'm getting build errors on my new ARCH Linux box when trying to build ktransformers. Seems to work okay on my Ubuntu 22.04 box though.
I tried it on Python 3.11 first, and then on 3.12 just to see including updating torch to latest cu128 nightly. No dice.
I linked some build log errors on what may be the same thing as [ktransformers GH Issues #217](https://github.com/kvcache-ai/ktransformers/issues/217)

Seems likely an issue with too new of `nvcc --version`:

* `Build cuda_12.0.r12.0/compiler.32267302_0` works with CUDA 12.8
* `Build cuda_12.8.r12.8/compiler.35404655_0` is *too new* and throws errors?

#### open-webui
Quick start for using [open-webui](https://github.com/open-webui/open-webui) with this ktransformers API Endpoint.
```
mkdir open-webui
cd open-webui
uv venv ./venv --python 3.11
source venv/bin/activate
uv pip install open-webui
```

Then copy paste this into a `./go.sh` script and `chmod u+x go.sh` or whatever you like.
```bash
#!/usr/bin/env bash

source venv/bin/activate

# open-webui does NOT honor HOST and PORT ENV VAR so pass it manually as arguments
# https://docs.openwebui.com/getting-started/env-configuration/#port
# when open-webui gets borked just do `rm -rf ./data` and restart everything after clearing browser cache

export DATA_DIR="$(pwd)/data"
export ENABLE_OLLAMA_API=False
export ENABLE_OPENAI_API=True
export OPENAI_API_KEY="dont_change_this_cuz_openai_is_the_mcdonalds_of_ai"
export OPENAI_API_BASE_URL="http://127.0.0.1:8080/v1" # <--- this must match API config above for ktransformers or llama.cpp
#export DEFAULT_MODELS="openai/foo/bar" # <--- leave this commented, i use this slug for `litellm` access
export WEBUI_AUTH=False
export DEFAULT_USER_ROLE="admin"
export HOST=127.0.0.1
export PORT=3000 # <--- this is for the open-webui server webpage

open-webui serve \
  --host $HOST \
  --port $PORT

# open browser to the url/port shown
```

## References
* [kvcache-ai/KTransformers](https://github.com/kvcache-ai/KTransformers)
* [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
* [previous version of this guide with more notes but kinda messy]()
* [level1techs forum DeepSeek Deep Dive thread](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/)
