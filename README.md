Run DeepSeek-R1 671B unsloth quants with ktransformers
===
In early testing ktransformers is running faster than llama.cpp right
now for DeepSeek-R1 GGUF inferencing for some system configurations.

There are some rough edges and not exactly sure how it will pan out but
this will get you started to compare results for yourself.

Keep in mind llama.cpp has some [experimental
branches](https://github.com/ggml-org/llama.cpp/pull/11397#issuecomment-2645973828)
allowing selective expert offload etc so this may all be moot sooner
than later as things are moving so fast.

## NOTES for `ktransformers@c515cc4`

1. There is a *hard runtime requirement* on at least one *CUDA* GPU with about 14GB VRAM for 4k context and [24GB for 8k context in v0.2.1](https://github.com/kvcache-ai/ktransformers?tab=readme-ov-file#-updates). If you want CPU *ONLY* inference, continue using `llama.cpp` currently.
2. The Web and API interface do not work yet. They will hallucinate.
3. Only the local chat is working. It has no memory, just one round of user prompt and assistant response.
4. You must use a single line prompt with local chat as any copy paste '\n' will trigger multiple generations leading to confusing output.

If you have an Intel Xeon with AMX extensions look closely at secton 4 below regarding the `v0.3 binary`.

## INCOMING FIX
I tested an incoming [ktransformers PR #382](https://github.com/kvcache-ai/ktransformers/pull/382#issuecomment-2664109713)
which seems fix the OpenAI compatible API endpoint! With luck it will land in `main soon.`

Initial new benchmark for this experimental PR looks really good at face value:

| engine | pp | tg |
| --- | --- | --- |
| | tok/sec | tok/sec |
| `ktransformers@ee24eb8` | 86.5 | 14.8 |
| `llama.cpp@90e4dba4` | 35.4 | 7.3 |

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

#### 3. Install ktransformers
*Note*: `ktransformers` probably *requires* GPU as it has a [hard requirement on CUDA dependencies](https://github.com/kvcache-ai/ktransformers/issues/337#issuecomment-2661711997) at least to compile.
```
# install deps e.g. `apt-get install build-essential npm` as needed
# install uv https://docs.astral.sh/uv/getting-started/installation/
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo and build it with python
$ git clone https://github.com/kvcache-ai/ktransformers.git --depth=1
$ cd ktransformers
$ git rev-parse --short HEAD # c515cc4
$ uv venv ./venv --python 3.11
$ source  venv/bin/activate
$ uv pip install -r requirements-local_chat.txt
$ uv pip install setuptools wheel packaging

# REMEMBER API and WEBSITE DOES NOT YET WORK SO PROBABLY SKIP THIS
# OPTIONAL: first build the webapp so you can use it through browser
$ cd ktransformers/website/
$ npm install @vue/cli
$ npm run build
$ cd ../..

# there is a *HARD RUNTIME REQUIREMENT* on at least a single *CUDA* GPU w/ 16GB VRAM or more
# might be able to prepend `MAX_JOBS=8 uv pip ...` or some way to speed it up a bit?
$ uv pip install flash_attn --no-build-isolation

# ONLY IF you have dual CPU sockets and >1TB RAM
$ export USE_NUMA=1

# finally do the real build
$ KTRANSFORMERS_FORCE_BUILD=TRUE uv pip install . --no-build-isolation

# DONE, Continue below!

# WIP: If there is an error and you have very new cuda 12.8 maybe try this?
# uv pip uninstall torch
# uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# if you want to rebuild again, first make clean like so
$ rm -rf ktransformers/ktransformers_ext/build
$ rm -rf ktransformers/ktransformers_ext/cuda/build
$ rm -rf ktransformers/ktransformers_ext/cuda/dist
$ rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info
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
1. Seems like maybe some configs live in `ktransformers/configs/config.yaml` used with `--mode="long_context"`?
```bash
$ find . -name config.yaml
./venv/lib/python3.11/site-packages/ktransformers/configs/config.yaml # <--- after running with `--mode="long_context"`
./build/lib.linux-x86_64-cpython-311/ktransformers/configs/config.yaml
./ktransformers/configs/config.yaml
```
2. For some reason it tries to keep the database on the server side instead of client side? `rm ./venv/lib/python3.11/site-packages/ktransformers/server.db`
3. MultiGPU and more offload customization available with `--optimize_config_path`
```bash
$ find . -name "DeepSeek-V3-*.yaml"
# use the one in venv/ and ignore build/ and ktransformers/
# e.g. --optimize_config_path ./ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml
```
4. Full experimental command (not sure everything is actually working, got crash on multi-gpu with 2x CUDAs)
```
# this crashed, but maybe can increase offload with 1x GPU editing the yaml?
# https://github.com/kvcache-ai/ktransformers/blob/c515cc49a595696fedaca6032e100951c42ad36f/doc/en/multi-gpu-tutorial.md
ktransformers \
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
I'm getting this on my new ARCH Linux box when trying to build ktransformers. Seems to work okay on 22.04 box though.
I tried it on Python 3.11 first, and then on 3.12 just to see including updating torch to latest cu128 nightly. No dice.
It may be the same thing as [ktransformers GH Issues #217](https://github.com/kvcache-ai/ktransformers/issues/217)
```
      /mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py:492:
      UserWarning: There are no c++ version bounds defined for CUDA version 12.8
        warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
      Traceback (most recent call last):
        File "<string>", line 11, in <module>
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/build_meta.py",
      line 435, in build_wheel
          return _build(['bdist_wheel'])
                 ^^^^^^^^^^^^^^^^^^^^^^^
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/build_meta.py",
      line 426, in _build
          return self._build_with_temp_dir(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/build_meta.py",
      line 407, in _build_with_temp_dir
          self.run_setup()
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/build_meta.py",
      line 320, in run_setup
          exec(code, locals())
        File "<string>", line 295, in <module>
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/__init__.py",
      line 117, in setup
          return distutils.core.setup(**attrs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/core.py",
      line 186, in setup
          return run_commands(dist)
                 ^^^^^^^^^^^^^^^^^^
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/core.py",
      line 202, in run_commands
          dist.run_commands()
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/dist.py",
      line 983, in run_commands
          self.run_command(cmd)
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/dist.py", line
      999, in run_command
          super().run_command(command)
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/dist.py",
      line 1002, in run_command
          cmd_obj.run()
        File "<string>", line 153, in run
        File
      "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/command/bdist_wheel.py",
      line 379, in run
          self.run_command("build")
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/cmd.py",
      line 339, in run_command
          self.distribution.run_command(command)
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/dist.py", line
      999, in run_command
          super().run_command(command)
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/dist.py",
      line 1002, in run_command
          cmd_obj.run()
        File
      "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/command/build.py",
      line 136, in run
          self.run_command(cmd_name)
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/cmd.py",
      line 339, in run_command
          self.distribution.run_command(command)
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/dist.py", line
      999, in run_command
          super().run_command(command)
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/dist.py",
      line 1002, in run_command
          cmd_obj.run()
        File
      "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/command/build_ext.py",
      line 99, in run
          _build_ext.run(self)
        File
      "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/command/build_ext.py",
      line 365, in run
          self.build_extensions()
        File "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py",
      line 1009, in build_extensions
          build_ext.build_extensions(self)
        File
      "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/command/build_ext.py",
      line 481, in build_extensions
          self._build_extensions_serial()
        File
      "/mnt/astrodata/llm/ktransformers/ktransformers/venv/lib/python3.12/site-packages/setuptools/_distutils/command/build_ext.py",
      line 507, in _build_extensions_serial
          self.build_extension(ext)
        File "<string>", line 285, in build_extension
        File "/home/garm/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib/python3.12/subprocess.py", line 573,
      in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['cmake',
      '/mnt/astrodata/llm/ktransformers/ktransformers/ktransformers/ktransformers_ext',
      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/mnt/astrodata/llm/ktransformers/ktransformers/build/lib.linux-x86_64-cpython-312/',
      '-DPYTHON_EXECUTABLE=/mnt/astrodata/llm/ktransformers/ktransformers/venv/bin/python3',
      '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_NATIVE=ON', '-DEXAMPLE_VERSION_INFO=0.2.1+cu128torch27fancy']' returned
      non-zero exit status 1.

      hint: This usually indicates a problem with the package or the build environment.
```
