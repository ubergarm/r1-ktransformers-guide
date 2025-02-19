DeepSeek-R1 671B unsloth quants on ktransformers
===
In early testing ktransformers is running about 2x faster than llama.cpp
right now for DeepSeek-R1 GGUF inferencing for some supported system
configurations and quants.

While ktransformers is still rough around the edges and not production ready imo,
it does seem to have implemented some optimizations ahead of llama.cpp for R1 including:

* flash attention - can quantize kv cache and possibly supports context-shift?
* Multi-head Latent Attention (MLA) with [SGLang](https://github.com/sgl-project/sglang)
* [Selective layer offload](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/deepseek-v2-injection.md#routed-experts-)
* experimental [flashinfer](https://github.com/flashinfer-ai/flashinfer) Coming Soon :TM:
* Intel Xeon AMX extensions and copy model into RAM *twice* for big dual socket systems (as cross NUMA nodes is bottleneck)

Keep in mind llama.cpp has some as experimental branches going as well including:
* [flash attention PR and ongoing discussion](https://github.com/ggml-org/llama.cpp/pull/11557)
* [MLA](https://github.com/ggml-org/llama.cpp/pull/11446)
* [selective layer offload](https://github.com/ggml-org/llama.cpp/pull/11397#issuecomment-2645973828)

## NOTES for `ktransformers@c515cc4`

1. There is a *hard runtime requirement* on at least one *CUDA* GPU with about 14GB VRAM for 4k context and [24GB for 8k context in v0.2.1](https://github.com/kvcache-ai/ktransformers?tab=readme-ov-file#-updates). If you want CPU *ONLY* inference, continue using `llama.cpp` currently.
2. The Web and API interface do not work yet. They will hallucinate.
3. Only the local chat is working. It has no memory, just one round of user prompt and assistant response.
4. You must use a single line prompt with local chat as any copy paste '\n' will trigger multiple generations leading to confusing output.

If you have an Intel Xeon with AMX extensions look closely at secton 4 below regarding the `v0.3 binary`.

## INCOMING FIX for API endpoint
I tested an incoming [ktransformers PR #382](https://github.com/kvcache-ai/ktransformers/pull/382#issuecomment-2664109713)
which seems fix the OpenAI compatible API endpoint! With luck it will land in `main` soon.

Initial new benchmark for this experimental PR looks really good at face value:

| engine | pp | tg |
| --- | --- | --- |
| | tok/sec | tok/sec |
| `ktransformers@ee24eb8` | 86.5 | 14.8 |
| `llama.cpp@90e4dba4` | 35.4 | 7.3 |

On my local 3090TI 24GB VRAM + 96GB DDR5@88GB/s + 9950X + PCIe Gen 5
T700 2TB NVMe hits around 3.2 tok/sec generation with this same model
despite not having enough RAM. Disk `mmap()` i/o discussion below.

## Guide
#### 1. Download unsloth GGUF files
Download desired
[unsloth/DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-Q2_K_XL)
files (`UD-Q2_K_XL` linked) and put everything into a single directory
on your fastest drive.

#### 2. Download .py and .json files
I copied the `.json` and `.py` files out of [this questional hugging face repo](https://huggingface.co/is210379/DeepSeek-R1-UD-IQ1_S/discussions/1)

You can get them by git cloning this repo or just download them directly.

Copy all the `.py` and `.json` files into the directory with your GGUF model files above. It should look something like this:
```
$ ls /mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/
config.json                                 DeepSeek-R1-UD-Q2_K_XL-00002-of-00005.gguf  DeepSeek-R1-UD-Q2_K_XL-00005-of-00005.gguf  tokenizer.json
configuration_deepseek.py                   DeepSeek-R1-UD-Q2_K_XL-00003-of-00005.gguf  generation_config.json
DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf  DeepSeek-R1-UD-Q2_K_XL-00004-of-00005.gguf  tokenizer_config.json
```

*NOTE* I've realized these files might be out of [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main) and you can possibly just call `ktransformers --model_path deepseek-ai/DeepSeek-R1` and it will pull the files automagically maybe.

#### 3. Install ktransformers
*Note*: `ktransformers` probably *requires* GPU as it has a [hard requirement on CUDA dependencies](https://github.com/kvcache-ai/ktransformers/issues/337#issuecomment-2661711997) at least to compile.
```
# install deps e.g. `apt-get install build-essential npm` as needed
# install uv https://docs.astral.sh/uv/getting-started/installation/
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo and build it with python
git clone https://github.com/kvcache-ai/ktransformers.git --depth=1
cd ktransformers
git rev-parse --short HEAD # c515cc4
uv venv ./venv --python 3.11
source  venv/bin/activate
uv pip install -r requirements-local_chat.txt
uv pip install setuptools wheel packaging

# REMEMBER WEBSITE DOES NOT YET WORK SO PROBABLY SKIP THIS
# OPTIONAL: first build the webapp so you can use it through browser
cd ktransformers/website/
npm install @vue/cli
npm run build
cd ../..

# there is a *HARD RUNTIME REQUIREMENT* on at least a single *CUDA* GPU w/ 16GB VRAM or more
# might be able to prepend `MAX_JOBS=8 uv pip ...` or some way to speed it up a bit?
uv pip install flash_attn --no-build-isolation

# ONLY IF you have Intel dual socket and >1TB RAM to hold 2x copies of entire model in RAM (one copy per socket)
# Dual socket AMD EPYC NPS0 probably makes this not needed?
# $ export USE_NUMA=1

# finally do the real build
KTRANSFORMERS_FORCE_BUILD=TRUE uv pip install . --no-build-isolation

# DONE, Continue below!

# WIP: If there is an error and you have very new nvcc 12.8 maybe? example error log below.
# this didn't work for me, but might for you. i built it on a different box then copied it all over.
# $ uv pip uninstall torch
# $ uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# if you want to rebuild again, first make clean like so
uv pip uninstall ktransformers
rm -rf ktransformers/ktransformers_ext/build
rm -rf ktransformers/ktransformers_ext/cuda/build
rm -rf ktransformers/ktransformers_ext/cuda/dist
rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info
```

#### 4. OPTIONAL Upgrade to v0.3 preview binary *only* for Intel Xeon CPUs with AMX Extensions
[These binaries will crash on other CPUs even with AVX512 CPU flags e.g. AMD 9950X, Threadripper Pro, etc.](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html#v03-showcase)

Check to see if your Intel Xeon processor and Linux kernel are new enough to support AMX extensions:
```
$ lscpu | grep amx
amx_bf16 avx512_fp16 amx_tile amx_int8
```

*WARNING* Take necesary precautions when running an unknown binary file such as securing your network and files etc.
```
# first backup the website dist e.g.
$ mkdir backups
$ rsync -avh venv/lib/python3.11/site-packages/ktransformers/website ./backups/

# download the binary at your own risk
$ wget https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl

# strange they put the v0.3 in an old v0.1.4 release tag despite it being newer
# https://github.com/kvcache-ai/ktransformers/releases/tag/v0.1.4

# install it over what you just built and installed
$ uv pip install -U ./ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl

# REMEMBER API and WEBSITE DOES NOT YET WORK
# restore the website
$ rsync -avh ./backups/website venv/lib/python3.11/site-packages/ktransformers/
```

It crashes on CPUs without AMX extensions.
```
# web crashes
2025-02-14 16:24:41,253 INFO ktransformers/venv/lib/python3.11/site-packages/ktransformers/server/main.py[29]: Creating SQL tables
2025-02-14 16:24:41,254 INFO ktransformers/venv/lib/python3.11/site-packages/ktransformers/server/api/openai/assistants/assistants.py[75]: Creating default assistant
__AVX512F__
Injecting model as ktransformers.operators.models . KDeepseekV2Model
Injecting model.embed_tokens as default
Injecting model.layers as default
Injecting model.layers.0 as default
Injecting model.layers.0.self_attn as ktransformers.operators.attention . KDeepseekV2Attention

Illegal instruction     (core dumped) ktransformers --gguf_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" --model_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" --cpu_infer 24 --host 127.0.0.1 --port 8080 --web true --no_flash_attn true

# local chat crashes
using custom modeling_xxx.py.
using default_optimize_rule for DeepseekV3ForCausalLM
__AVX512F__
Injecting model as ktransformers.operators.models . KDeepseekV2Model
Injecting model.embed_tokens as default
Injecting model.layers as default
Injecting model.layers.0 as default
Injecting model.layers.0.self_attn as ktransformers.operators.attention . KDeepseekV2Attention
743438 Illegal instruction     (core dumped) python ./ktransformers/local_chat.py --gguf_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" --model_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" --cpu_infer 24 --max_new_tokens 1000 --force_think true
```

Probably because it is compiled for Intel Xeon with AMX Optimization only.

> Intel AMX Optimization – Our AMX-accelerated kernel is meticulously tuned, running several times faster than existing llama.cpp implementations. We plan to open-source this kernel after cleansing and are considering upstream contributions to llama.cpp.

[Reference](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html#some-explanations)


#### 5. Test local chat
```
# local chat test
$ touch p.txt # this might be system_prompt? you can skip it below and seems to work fine
# use cpu_infer equal to number of physical CPU cores on your box (not SMT/hyperthreads)
$ python ./ktransformers/local_chat.py \
    --gguf_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --model_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --prompt_file ./p.txt \
    --cpu_infer 16 \
    --max_new_tokens 1024 \
    --force_think true

Chat: Count from one to ten in Mandarin Chinese.
```

### 6. Test web app
*NOTE*: This does not seem to work as generations will hallucinate immedeately.

This provides a built in web app as well as an API compliant with `litellm` or `OpenWebUI` etc.
```
# server and webpage
$ ktransformers
    --gguf_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --model_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --cpu_infer 16 \
    --force_think true \
    --host 127.0.0.1 \
    --port 8080 \
    --web true

# Open Browser to: http://localhost:8080/web/index.html#/chat
```

## Benchmarks
Check out [level1techs](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826) for more llama.cpp benchmarks including CPU only performance.

#### Hardware
* AMD Ryzen Threadripper PRO 7965WX 24-Cores
* Dual NVIDIA RTX A6000 with 48 GB VRAM each (96GB total)
* Linux Kernel 6.13.0
* Ubuntu 24.04.1 LTS (Noble Numbat)

#### ktransformers@c515cc4
```
$ python ./ktransformers/local_chat.py \
    --gguf_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --model_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --cpu_infer 24 \
    --max_new_tokens 1024 \
    --force_think true

Chat: Count from one to ten in Mandarin Chinese.

...

prompt eval count:    14 token(s)
prompt eval duration: 0.8247685432434082s
prompt eval rate:     16.974459216090978 tokens/s # <--- as high as 60 tokens/s at times
eval count:           750 token(s)
eval duration:        56.33242607116699s
eval rate:            13.31382388985866 tokens/s
```

#### llama.cpp@90e4dba4
```
$ ./build/bin/llama-server \
    --model "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf" \
    --ctx-size 8192 \
    --n-gpu-layers 21 \
    --cache-type-k q4_0 \
    --cache-type-v f16 \
    --parallel 1 \
    --threads 24 \
    --host 127.0.0.1 \
    --port 8080
...

prompt eval time =    8014.15 ms /   226 tokens (   35.46 ms per token,    28.20 tokens per second)
       eval time =   75773.60 ms /   615 tokens (  123.21 ms per token,     8.12 tokens per second)
      total time =   83787.75 ms /   841 tokens
```

## Advanced

#### Advanced 1
Seems like maybe some configs live in `ktransformers/configs/config.yaml` used with `--mode="long_context"`? (This is probably a red herring, see newer stuff below lol)
```bash
$ find . -name config.yaml
./venv/lib/python3.11/site-packages/ktransformers/configs/config.yaml # <--- after running with `--mode="long_context"`
./build/lib.linux-x86_64-cpython-311/ktransformers/configs/config.yaml
./ktransformers/configs/config.yaml
```

#### Advanced 2
For some reason it tries to keep the database on the server side instead of client side? `rm ./venv/lib/python3.11/site-packages/ktransformers/server.db`

#### Advanced 3
MultiGPU and more offload customization available with `--optimize_config_path`
```bash
$ find . -name "DeepSeek-V3-*.yaml"
# use the one in venv/ and ignore build/ and ktransformers/
# e.g. --optimize_config_path ./ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml
# could possibly make a CPU only DeepSek-V3-Chat-cpu.yaml ???
```

#### Advanced 4
Experimental 2x GPU command (not sure everything is actually working)
```
# this crashed, but maybe can increase offload with 1x GPU editing the yaml?
# https://github.com/kvcache-ai/ktransformers/blob/c515cc49a595696fedaca6032e100951c42ad36f/doc/en/multi-gpu-tutorial.md
$ ktransformers \
    --gguf_path "/mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --model_path "/mnt/raid/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --cpu_infer 24 \
    --host 127.0.0.1 \
    --port 8080 \
    --web false \
    --no_flash_attn false \
    --total_context 8192 \
    --cache_q4 true \
    --gpu_split 1,1 \
    --mode="long_context" \
    --optimize_config_path ./venv/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml
    # --optimize_config_path ./venv/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml # single GPU
    # maybe you can make a -cpu-only.yaml and figure out the injection syntax to no longer require GPU?
```

#### Advanced 5
More Advanced Commands

I found the best documentation for how to run ktransformers in [ktransformers PR#382 ceerRep comment](https://github.com/kvcache-ai/ktransformers/pull/382#issue-2856190692)

Possibly the `--mode="long_context"` `config.yaml` is a red herring and it is easier to simply pass those values via CLI arguments.
```bash
# chat
# seems to imply --prompt_file is actually user prompt and there is no way to provide system prompt which is fine as R1 should have none
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 ktransformers/local_chat.py \
    --model_path ~/DeepSeek-R1/ \
    --model_name deepseek \ # <--- this will make open-webui look prettier probably lmao...
    --gguf_path ~/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/ \
    --cpu_infer 64 \
    --max_new_tokens 8192 \
    --cache_lens 32768 \
    --total_context 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_rule_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-4gpu.yaml \ # <--- or specify the single GPU or maybe make your own for CPU only ???
    --force_think \ # <--- only for R1
    --use_cuda_graph \ # <--- not 100% this is actually good yet or not for all configs
    --prompt_file ~/test.txt # <--- can probably omit for interactive one shot no carriage return chat: interface

# all argument comments are above
# API server
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 ktransformers/server/main.py \
    --model_path /home/naivlex/DeepSeek-R1/ \
    --model_name deepseek \
    --gguf_path /home/naivlex/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/ \
    --cpu_infer 64 \
    --max_new_tokens 8192 \
    --cache_lens 32768 \
    --total_context 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml \
    --force_think \
    --use_cuda_graph \
    --host 127.0.0.1 \
    --port 8080
```

#### Advanced 6
Optional open-webui interface

I want a more simplified version of this for CLI which I think exists. I
don't want full `tui` like `textual` but I want actual scrollback
buffer and using [primp](https://github.com/deedy5/primp) to expand
`#https://some.site.com` into prompt.

First install [open-webui](https://github.com/open-webui/open-webui)
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
# i don't know why it downloads onnx models, annoys me lmao...
```

#### Advanced 7
Disk Read IOPs Discussion

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

## References
* [ktransformers github update documentation PR](https://github.com/kvcache-ai/ktransformers/pull/384)
* [ktransformers on unsloth dynamic quants issue](https://github.com/kvcache-ai/ktransformers/issues/311)
* [unsloth r1 gguf ktransformers gh issue](https://github.com/kvcache-ai/ktransformers/issues/186#issuecomment-2659894815)
* [mysterioiusly popular r1 gguf repo with missing files](https://huggingface.co/is210379/DeepSeek-R1-UD-IQ1_S/discussions/1)
* [reddit post of another guy trying ktransformers but video is too long to watch](https://www.reddit.com/r/LocalLLaMA/comments/1ioybsf/comment/mcs1g9n/)
* [reddit post as above comment about the v0.2 vs v0.3 and avx-512-only binary](https://www.reddit.com/r/LocalLLaMA/comments/1ioybsf/comment/mco7x22/)
* [ktransformers deepseek-r1 official install guide](https://kvcache-ai.github.io/ktransformers/en/install.html#installation)
* [ktransformers deepseek-r1 tutorial guide](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html)
* [ktransformers deepseek-r1 faq](https://kvcache-ai.github.io/ktransformers/en/FAQ.html)

## Error Logs
I'm getting build errors on my new ARCH Linux box when trying to build ktransformers. Seems to work okay on my Ubuntu 22.04 box though.
I tried it on Python 3.11 first, and then on 3.12 just to see including updating torch to latest cu128 nightly. No dice.
I linked some build log errors on what may be the same thing as [ktransformers GH Issues #217](https://github.com/kvcache-ai/ktransformers/issues/217)

Seems likely an issue with too new of `nvcc --version`:

* `Build cuda_12.0.r12.0/compiler.32267302_0` works with CUDA 12.8
* `Build cuda_12.8.r12.8/compiler.35404655_0` is *too new* and throws this error:

Might be possible to use a `Dockerfile` to build a python cheese `.whl` file or similar if you can't get it building.
