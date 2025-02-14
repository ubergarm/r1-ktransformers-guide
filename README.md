Run DeepSeek-R1 unsloth quants with ktransformers
===
In theory it seems like ktransformers might be faster than llama.cpp right
now for DeepSeek-R1 GGUF CPU/GPU inferencing. There are some rough edges
and not exactly sure how it will pan out but this will get you started
to compare results for yourself.

Keep in mind llama.cpp has some experimental threads allowing selective
expert offload etc so this may all be moot sooner than later as things
are moving so fast.

Right now on Threadripper Pro with 256GB RAM and ~24GB RAM `ktransformers`
generates just over 11 tok/sec and `llama.cpp` generates just over 8
tok/sec. This is highly anecdotal as the GPU VRAM is not loaded right
and didn't check context lengths etc.

This is a quick manic attempt at getting some info out on `ktransformers`
though it doesn't seem fully baked yet haha... Also I notice it seems
to just go off the rails pretty easily as compared to llama.cpp hah..

Also its [unclear what is ktransformers v0.2 vs v0.3](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html#v03-showcase)
I found a way to get the v0.3 binary preview file with the website good luck... lol

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
```
# install deps e.g. `apt-get install build-essential npm` as needed
# install uv https://docs.astral.sh/uv/getting-started/installation/
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo and build it with python
$ git clone https://github.com/kvcache-ai/ktransformers.git --depth=1
$ uv venv ./venv --python 3.11
$ source  venv/bin/activate
$ uv pip install -r requirements-local_chat.txt
$ uv pip install setuptools wheel packaging

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

#### 4.5 Upgrade to v0.3 preview binary jawn
Optional for latest stuff in FAQ.
```
# first backup the website dist e.g.
mkdir backups
rsync -avh venv/lib/python3.11/site-packages/ktransformers/website ./backups/

# download the binary
wget https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl

# install it over what you just built and installed
uv pip install -U ./ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl

# restore the website
rsync -avh ./backups/website venv/lib/python3.11/site-packages/ktransformers/
```

#### 5. Test local chat
```
# local chat test
$ touch p.txt # unsure if this is system prompt or what?
# use cpu_infer equal to number of physical CPU cores on your box (not SMT/hyperthreads)
$ python ./ktransformers/local_chat.py \
    --gguf_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --model_path "/mnt/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/" \
    --prompt_file ./p.txt \
    --cpu_infer 16 \
    --max_new_tokens 1024 \
    --force_think true

prompt eval count:    9 token(s)
prompt eval duration: 0.8497006893157959s
prompt eval rate:     10.591965045064358 tokens/s
eval count:           300 token(s)
eval duration:        26.647690773010254s
eval rate:            11.25801115584285 tokens/s
```

### 6. Test web app
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

## Bugs
```
# ERROR 1
    |   File "/home/w/projects/ktransformers/venv/lib/python3.11/site-packages/ktransformers/server/backend/interfaces/ktransformers.py", line 130, in prefill
    |     self.cache.reset()
    |   File "/home/w/projects/ktransformers/venv/lib/python3.11/site-packages/ktransformers/models/custom_cache.py", line 175, in reset
    |     self.value_cache[layer_idx].zero_()
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AttributeError: 'NoneType' object has no attribute 'zero_'
    +------------------------------------

Edit `venv/lib/python3.11/site-packages/ktransformers/models/custom_cache.py` line 175
# add a line above e.g. `if not layer_idx return`
# e.g.
    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            if not layer_idx: # <--- add this line
                return        # <--- add this line
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

# ERROR 2
# It seems to throw this error if the chat continues
    |   File "/home/w/projects/ktransformers/venv/lib/python3.11/site-packages/ktransformers/server/backend/interfaces/transformers.py", line 187, in format_and_tokenize_input_id
    |     unequal_mask = torch.ne(x,y)
    |                    ^^^^^^^^^^^^^
    | RuntimeError: The size of tensor a (639) must match the size of tensor b (206) at non-singleton dimension 1

# not sure what to do here, might be the `.py` and `.json` files don't line up with the GGUF quant??
# maybe FA issue? no idea...
# for now just delete the chat and start a new chat
```

## v0.3 vs v0.2 ???
```
wget https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
uv pip install -U ./ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
```

## References
* [unsloth r1 gguf ktransformers gh issue](https://github.com/kvcache-ai/ktransformers/issues/186#issuecomment-2659894815)
* [mysterioiusly popular r1 gguf repo with missing files](https://huggingface.co/is210379/DeepSeek-R1-UD-IQ1_S/discussions/1)
* [reddit post of another guy trying ktransformers but video is too long to watch](https://www.reddit.com/r/LocalLLaMA/comments/1ioybsf/comment/mcs1g9n/)
* [ktransformers deepseek-r1 tutorial guide](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html)
* [ktransformers deepseek-r1 faq](https://kvcache-ai.github.io/ktransformers/en/FAQ.html)
