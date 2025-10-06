# The Pitfalls of KV Cache Compression

[![arXiv](https://img.shields.io/badge/arXiv-2510.00231-blue?link=https%3A%2F%2Farxiv.org%2Fabs%2F2510.00231)](https://arxiv.org/abs/2510.00231)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


This repository contains the code for [*The Pitfalls of KV Cache Compression*](https://arxiv.org/abs/2510.00231).

> **_TL;DR_**: for prompts with multiple instructions, KV cache compression can lead to some instructions being ignored. We propose simple changes to KV cache eviction policies that fix this. 


## Code structure
```
pitfalls-of-kv-cache-compression/
├─ kv_cache_compression/
│  ├─ benchmarks/ # ifeval, raccoon
│  ├─ experiments/ # instruction following and leakage
│  └─ requirements.txt
└─ kvpress/ # forked from https://github.com/NVIDIA/kvpress. Modified to include fair eviction policies
```

> [!IMPORTANT]
> **Currently supported compression methods:** StreamingLLM, SnapKV, ObservedAttention, TOVA, Knorm.
>
> **Added fair-eviction variants (in `kvpress/kvpress/presses/`):**
> - `StreamingLLMFairEvictionPress`
> - `SnapKVFairEvictionPress`
> - `ObservedAttentionFairEvictionPress`
> - `KnormFairEvictionPress`
> - `TOVAFairEvictionPress`

## Installation

```bash
git clone https://github.com/Itisalex2/pitfalls-of-kv-cache-compression.git
conda create -n <env-name> python=3.11 
conda activate <env-name>
pip install -r kv_cache_compression/requirements.txt
pip install -e kvpress
```

> [!TIP]
> We recommend installing FlashAttention to boost efficiency in compression methods that don’t rely on full attention scores:
> https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention  
> If FlashAttention cannot be installed, modify
> `kv_cache_compression/utils.py#get_attention_implementation_from_strategy`
> to use `sdpa` instead of `flash_attention_2`.


## Experiments

In `The Pitfalls of KV Cache Compression`, we investigate *instruction following* and *leakage*.

### Instruction following

Instruction following evaluation code is adapted from [SystemCheck](https://github.com/normster/SystemCheck/tree/main/evals/ifeval).

#### Usage
```
options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords
                        separated by comma. The supported flags are: comments, skip_default, skip_null.
  --model_name_or_path MODEL_NAME_OR_PATH
                        (type: str, default: /data1/shared/Llama-3.2-1B-Instruct)
  --model_name_shorthand MODEL_NAME_SHORTHAND
                        (type: str, default: llama_3.2_1b_instruct)
  --press_name PRESS_NAME
                        (type: str, default: streamingllm)
  --outputs_base OUTPUTS_BASE
                        (type: <class 'Path'>, default: <repo-path>/kv_cache_compression/experiments/compressed_context_ifeval_outputs)
  --log_level LOG_LEVEL
                        (type: str, default: INFO)
  --num_prompts NUM_PROMPTS
                        (type: int, default: -1)
  --compression_ratio_start COMPRESSION_RATIO_START
                        (type: float, default: 0.0)
  --compression_ratio_end COMPRESSION_RATIO_END
                        (type: float, default: 0.95)
  --compression_ratio_steps COMPRESSION_RATIO_STEPS
                        (type: int, default: 100)
  --generate_responses {true,false}
                        (type: bool, default: True)
  --generate_entropy {true,false}
                        (type: bool, default: True)
  --num_responses NUM_RESPONSES
                        (type: int, default: 1)
  --max_new_tokens MAX_NEW_TOKENS
                        (type: int, default: 1280)
  --run_name RUN_NAME   (type: str, default: )
  --seed SEED           (type: int, default: 42)
  --model_cache_dir MODEL_CACHE_DIR
                        (type: <class 'Path'>, default: /data1/shared)
  --sys_ifeval_path SYS_IFEVAL_PATH
                        (type: <class 'Path'>, default: <repo-path>/kv_cache_compression/benchmarks/ifeval/inputs/sys_ifeval.jsonl)
  --sample_responses {true,false}
                        (type: bool, default: False)
  --skip_ids SKIP_IDS, --skip_ids+ SKIP_IDS
                        (type: Optional[list[str]], default: null)
  --system_prompt_prepend SYSTEM_PROMPT_PREPEND
                        (type: str | None, default: You are a helpful assistant. If you receive a valid user request, please respond while adhering to the following
                        guidelines: )
  --defense_template_key DEFENSE_TEMPLATE_KEY
                        (type: str | None, default: null)
  --defense_template_path DEFENSE_TEMPLATE_PATH
                        (type: <class 'Path'>, default: <repo-path>/kv_cache_compression/benchmarks/raccoon/defenses/defense_template.json)
  --force_keep_global FORCE_KEEP_GLOBAL, --force_keep_global+ FORCE_KEEP_GLOBAL
                        (type: Optional[list[int]], default: null)
  --analyze_kept_tokens {true,false}
                        (type: bool, default: True)
  --defense_span DEFENSE_SPAN
                        (type: Optional[tuple[int, int]], default: null)
  --use_automated_spans {true,false}
                        (type: bool, default: False)
```

#### Example
```bash
python -m kv_cache_compression.experiments.compressed_context_ifeval \
  --num_prompts -1 \
  --sample_responses False \
  --model_name_or_path <model-cache-dir>/Qwen2.5-14B-Instruct \
  --model_cache_dir <model-cache-dir> \
  --model_name_shorthand "qwen_2.5_14b_instruct"  \
  --press_name streamingllm \
  --compression_ratio_start 0.00 \
  --compression_ratio_end 0.90 \
  --compression_ratio_steps 10 \
  --defense_template_key standard_defense \
  --max_new_tokens 1280
```


### Leakage

Leakage evaluation code is adapted from [RaccoonBench](https://github.com/M0gician/RaccoonBench/blob/main/Raccoon/raccoon.py).

#### Usage
```
options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one
                        or more keywords separated by comma. The supported flags are: comments, skip_default, skip_null.
  --model_name_or_path MODEL_NAME_OR_PATH
                        (type: str, default: /data1/shared/Llama-3.2-1B-Instruct)
  --model_name_shorthand MODEL_NAME_SHORTHAND
                        (type: str, default: llama_3.2_1b_instruct)
  --press_name PRESS_NAME
                        (type: str, default: streamingllm)
  --outputs_base OUTPUTS_BASE
                        (type: <class 'Path'>, default: <repo-path>/kv_cache_compression/experiments/raccoon/raccoon_outputs)
  --log_level LOG_LEVEL
                        (type: str, default: INFO)
  --compression_ratio_start COMPRESSION_RATIO_START
                        (type: float, default: 0.0)
  --compression_ratio_end COMPRESSION_RATIO_END
                        (type: float, default: 0.95)
  --compression_ratio_steps COMPRESSION_RATIO_STEPS
                        (type: int, default: 100)
  --generate_responses {true,false}
                        (type: bool, default: True)
  --generate_entropy {true,false}
                        (type: bool, default: True)
  --num_responses NUM_RESPONSES
                        (type: int, default: 1)
  --max_new_tokens MAX_NEW_TOKENS
                        (type: int, default: 100)
  --run_name RUN_NAME   (type: str, default: )
  --seed SEED           (type: int, default: 42)
  --model_cache_dir MODEL_CACHE_DIR
                        (type: <class 'Path'>, default: /data1/shared)
  --sys_ifeval_path SYS_IFEVAL_PATH
                        (type: <class 'Path'>, default: <repo-path>/kv_cache_compression/benchmarks/ifeval/inputs/sys_ifeval_shuffled.jsonl)
  --defense_path DEFENSE_PATH
                        (type: <class 'Path'>, default: <repo-path>/kv_cache_compression/benchmarks/raccoon/defenses/defense_template.json)
  --attack_path ATTACK_PATH
                        (type: <class 'Path'>, default: <repo-path>/kv_cache_compression/benchmarks/raccoon/attacks/singular_attacks)
  --num_sys_prompts NUM_SYS_PROMPTS
                        (type: int, default: 100)
  --defense_templates DEFENSE_TEMPLATES, --defense_templates+ DEFENSE_TEMPLATES
                        (type: list[str] | str, default: all)
  --attack_categories ATTACK_CATEGORIES, --attack_categories+ ATTACK_CATEGORIES
                        (type: list[str] | str, default: all)
  --num_prompts_per_attack_category NUM_PROMPTS_PER_ATTACK_CATEGORY
                        (type: int, default: -1)
  --sample_responses {true,false}
                        (type: bool, default: False)
  --system_prompt_prepend SYSTEM_PROMPT_PREPEND
                        (type: str | None, default: You are a helpful assistant. If you receive a valid user request, please respond while
                        adhering to the following guidelines: )
  --force_keep_global FORCE_KEEP_GLOBAL, --force_keep_global+ FORCE_KEEP_GLOBAL
                        (type: Optional[list[int]], default: null)
  --analyze_kept_tokens {true,false}
                        (type: bool, default: True)
  --defense_span DEFENSE_SPAN
                        (type: Optional[tuple[int, int]], default: null)
  --use_automated_spans {true,false}
                        (type: bool, default: False)
```

#### Example
```bash
python -m kv_cache_compression.experiments.raccoon.run_raccoon \
  --num_sys_prompts 100 \
  --sample_responses False \
  --model_name_or_path <model-cache-dir>/Meta-Llama-3-8B-Instruct \
  --model_cache_dir <model-cache-dir> \
  --model_name_shorthand "llama_3_8b_instruct" \
  --press_name streamingllm \
  --compression_ratio_start 0.00 \
  --compression_ratio_end 0.90 \
  --compression_ratio_steps 10 \
  --defense_templates [standard_defense] \
  --attack_categories [Plain] \
  --max_new_tokens 256
```

## Key files for adding compression methods
`kv_cache_compression/utils.py`
- `get_attention_implementation_from_strategy`: returns the attention implementation based on the press strategy
- `get_output_attentions_from_strategy`: Returns whether the model should output attentions based on the press strategy
- `press_resolver`: Resolves the press strategy to the appropriate class instance. Sets the default parameters

`kvpress/kvpress/presses`:
- Contains all compression methods

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
You are free to share and adapt the material for any purpose, even commercially, as long as proper credit is given.

## How to cite

This work implements the paper [*The Pitfalls of KV Cache Compression*](https://arxiv.org/abs/2510.00231).


```
@misc{chen2025pitfallskvcachecompression,
      title={The Pitfalls of KV Cache Compression}, 
      author={Alex Chen and Renato Geh and Aditya Grover and Guy Van den Broeck and Daniel Israel},
      year={2025},
      eprint={2510.00231},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.00231}, 
}
```