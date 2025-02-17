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

Right now on Threadripper Pro with 256GB RAM and ~24GB RAM `ktransformers`
generates just over 11 tok/sec with and `llama.cpp` generates just over 8
tok/sec. This is highly anecdotal as the GPU VRAM is not loaded right
and didn't check context lengths etc.

There are still bugs in ktransformers at least running this unsloth dynamic quant
mentioned below.

## Bugs
1. The Web and API interface do not work yet. They will hallucinate.
2. Only the local chat is working.
3. You must use a single line prompt with local chat as any copy paste '\n' will trigger multiple generations leading to confusing output.

If you have an Intel Xeon with AMX extensions look closely at secton 4.5 below.

Additional references at bottom.

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

#### 4. Install ktransformers
Note `ktransformers` has a [hard requirement on CUDA dependencies](https://github.com/kvcache-ai/ktransformers/issues/337#issuecomment-2661711997) at least to compile.
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

# finally do the real build
$ KTRANSFORMERS_FORCE_BUILD=TRUE uv pip install . --no-build-isolation
# it seems to support flash attention with R1?? maybe?? maybe not??
$ uv pip install flash_attn --no-build-isolation
```

#### 4.5 OPTIONAL Upgrade to v0.3 preview binary *only* for Intel Xeon CPUs with AMX Extensions
[These binaries will crash on other CPUs even with AVX512 CPU flags e.g. AMD 9950X, Threadripper Pro, etc.](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html#v03-showcase)

*WARNING* Take necesary precautions when running an unknown binary file such as securing your network and files etc.
```
# first backup the website dist e.g.
mkdir backups
rsync -avh venv/lib/python3.11/site-packages/ktransformers/website ./backups/

# download the binary at your own risk
wget https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl

# strange they put the v0.3 in an old v0.1.4 release tag despite it being newer
https://github.com/kvcache-ai/ktransformers/releases/tag/v0.1.4

# install it over what you just built and installed
uv pip install -U ./ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl

# REMEMBER API and WEBSITE DOES NOT YET WORK
# restore the website
rsync -avh ./backups/website venv/lib/python3.11/site-packages/ktransformers/
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

prompt eval count:    14 token(s)
prompt eval duration: 0.42368268966674805s
prompt eval rate:     33.04359687437748 tokens/s
eval count:           946 token(s)
eval duration:        81.89759874343872s
eval rate:            11.55101021903636 tokens/s
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

## References
* [ktransformers github update documentation PR](https://github.com/kvcache-ai/ktransformers/pull/384)
* [ktransformers on unsloth dynamic quants issue](https://github.com/kvcache-ai/ktransformers/issues/311)
* [unsloth r1 gguf ktransformers gh issue](https://github.com/kvcache-ai/ktransformers/issues/186#issuecomment-2659894815)
* [mysterioiusly popular r1 gguf repo with missing files](https://huggingface.co/is210379/DeepSeek-R1-UD-IQ1_S/discussions/1)
* [reddit post of another guy trying ktransformers but video is too long to watch](https://www.reddit.com/r/LocalLLaMA/comments/1ioybsf/comment/mcs1g9n/)
* [reddit post as above comment about the v0.2 vs v0.3 and avx-512-only binary](https://www.reddit.com/r/LocalLLaMA/comments/1ioybsf/comment/mco7x22/)
* [ktransformers deepseek-r1 tutorial guide](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html)
* [ktransformers deepseek-r1 faq](https://kvcache-ai.github.io/ktransformers/en/FAQ.html)
