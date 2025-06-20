from dataclasses import dataclass
from pathlib import Path

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.cache_utils import DynamicCache
from kvpress import ScorerPress
import torch.nn.functional as F


@dataclass
class CompressedContextConfig:
    model_name_or_path: str = "/data1/shared/Llama-3.2-1B-Instruct"
    model_name_shorthand: str = (
        "llama_3.2_1b_instruct"  # Used for the generated run name
    )
    press_name: str = "streamingllm"
    outputs_base: Path = (
        Path(__file__).resolve().parent / "compressed_context_evaluation_outputs"
    )
    log_level: str = "INFO"
    system_prompt_column: str = "system_prompt"
    prompt_column: str = "prompt"
    target_column: str = "target"
    keywords_column: str = "keywords"
    refusal: str = "I can't"
    num_prompts: int = 1  # -1 means all prompts will be used
    compression_ratio_start: float = 0.00
    compression_ratio_end: float = 0.95
    compression_ratio_steps: int = 100
    generate_responses: bool = True
    generate_keywords_percentages: bool = True
    generate_entropy: bool = True  # Entropy analysis
    num_responses: int = 1
    max_new_tokens: int = 100  # Maximum number of new tokens to generate

    run_name: str = ""  # Custom run name, if empty will be generated automatically
    seed: int = 42
    model_cache_dir: Path = Path("/data1/shared")


@dataclass
class DatasetEntry:
    context: str
    prompt: str
    target: str
    keywords: list[str]


@torch.no_grad()
def compressed_context_loss(
    model: PreTrainedModel,
    messages_ids: torch.Tensor,
    context_len: int,
    context_cache: DynamicCache,
    target_ids: torch.Tensor,
) -> float:
    """
    Computes the compressed context loss for a given model, context, post-context
    messages and target.
    The loss is defined as the cross-entropy loss of the model's predictions
    for the target given the compressed context and post-context messages.
    """

    assert messages_ids.shape[0] == 1, (
        "Only batch size 1 is supported for compressed context loss."
    )
    assert target_ids.shape[0] == 1, "Only batch size 1 is supported for target_ids."

    context_kv_cache = context_cache
    # Restore context cache after generating each loss.
    # Ref: https://github.com/NVIDIA/kvpress/blob/main/kvpress/pipeline.py
    context_cache_seq_lengths = [
        context_cache.get_seq_length(layer_idx)
        for layer_idx in range(len(context_cache))  # type:ignore
    ]

    post_context_ids = messages_ids[:, context_len:]
    post_context_len = post_context_ids.shape[1]
    post_context_and_target_ids = torch.cat([post_context_ids, target_ids], dim=1)
    target_len = target_ids.shape[1]
    position_ids = torch.arange(
        context_len, context_len + post_context_len + target_len, device=model.device
    ).unsqueeze(0)

    outputs = model(
        input_ids=post_context_and_target_ids,
        past_key_values=context_kv_cache,
        position_ids=position_ids,
    )

    logits = outputs.logits[
        :,
        post_context_len - 1 : -1,
    ]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

    # Reset cache to its state before generating the loss
    # Ref: https://github.com/NVIDIA/kvpress/blob/main/kvpress/pipeline.py
    context_kv_cache.key_cache = [
        context_kv_cache.key_cache[layer_idx][:, :, :sequence_length]
        for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
    ]
    context_kv_cache.value_cache = [
        context_kv_cache.value_cache[layer_idx][:, :, :sequence_length]
        for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
    ]
    if hasattr(context_kv_cache, "_quantized_key_cache"):
        context_kv_cache._quantized_key_cache = [  # type:ignore
            context_kv_cache._quantized_key_cache[layer_idx][:, :, :sequence_length]  # type:ignore
            for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
        ]
        context_kv_cache._quantized_value_cache = [  # type:ignore
            context_kv_cache._quantized_value_cache[layer_idx][:, :, :sequence_length]  # type:ignore
            for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
        ]

    return loss.item()


@torch.no_grad()
def get_logits_entropy(
    logits: torch.Tensor,
    logits_mask: torch.Tensor,
) -> float:
    """Compute the average entropy of the logits over the non-masked tokens."""
    assert logits.dim() == 3  # (batch_size, seq_len, vocab_size)
    assert logits.size(0) == 1  # batch_size == 1
    assert logits_mask.dim() == 2  # (batch_size, seq_len)
    assert logits_mask.size(0) == 1  # batch_size == 1

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    entropy = entropy * logits_mask
    entropy_mean = entropy.sum(dim=1) / logits_mask.sum(dim=1).clamp(min=1.0)

    return entropy_mean.item()


@torch.no_grad()
def get_context_cache(
    model: PreTrainedModel,
    context_ids: torch.Tensor,
    press: ScorerPress,
) -> DynamicCache:
    assert context_ids.shape[0] == 1, "Only batch size 1 is supported for context_ids."
    with press(model):
        past_key_values = model(context_ids).past_key_values
    return past_key_values


@torch.no_grad()
def generate_from_compressed_context(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    messages_ids: torch.Tensor,
    context_len: int,
    context_cache: DynamicCache,
    do_sample: bool = True,
    temperature: float = 1.0,
    max_new_tokens: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert messages_ids.shape[0] == 1, (
        "Only batch size 1 is supported for context_and_prompt_ids."
    )

    past_key_values = context_cache

    # Restore context cache after generating each answer.
    # Ref: https://github.com/NVIDIA/kvpress/blob/main/kvpress/pipeline.py
    context_cache_seq_lengths = [
        context_cache.get_seq_length(layer_idx)
        for layer_idx in range(len(context_cache))  # type:ignore
    ]

    post_context_ids = messages_ids[:, context_len:]
    post_context_len = post_context_ids.shape[1]

    position_ids = torch.arange(
        context_len, context_len + post_context_len, device=model.device
    ).unsqueeze(0)
    generated_ids = torch.empty((1, 0), dtype=torch.long, device=model.device)
    generated_logits = torch.empty((1, 0, model.config.vocab_size), device=model.device)
    generated_logits_non_special_tokens_mask = torch.empty(
        (1, 0), dtype=torch.bool, device=model.device
    )

    input_ids = post_context_ids

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            position_ids=position_ids,
        )

        if do_sample:
            probabilities = F.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            sample_id = torch.multinomial(probabilities, num_samples=1)
        else:
            sample_id = outputs.logits[:, -1:].argmax(-1)

        generated_ids = torch.cat([generated_ids, sample_id], dim=1)
        generated_logits = torch.cat([generated_logits, outputs.logits[:, -1:]], dim=1)
        generated_logits_non_special_tokens_mask = torch.cat(
            [
                generated_logits_non_special_tokens_mask,
                (sample_id != tokenizer.eos_token_id)  # type:ignore
                & (sample_id != tokenizer.pad_token_id),  # type:ignore
            ],
            dim=1,
        )

        if sample_id.item() == tokenizer.eos_token_id:  # type:ignore
            break

        input_ids = sample_id
        position_ids = position_ids[:, -1:] + 1

    # Reset cache to its state before generation
    # Ref: https://github.com/NVIDIA/kvpress/blob/main/kvpress/pipeline.py
    past_key_values.key_cache = [
        past_key_values.key_cache[layer_idx][:, :, :sequence_length]
        for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
    ]
    past_key_values.value_cache = [
        past_key_values.value_cache[layer_idx][:, :, :sequence_length]
        for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
    ]
    if hasattr(past_key_values, "_quantized_key_cache"):
        past_key_values._quantized_key_cache = [  # type:ignore
            past_key_values._quantized_key_cache[layer_idx][:, :, :sequence_length]  # type:ignore
            for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
        ]
        past_key_values._quantized_value_cache = [  # type:ignore
            past_key_values._quantized_value_cache[layer_idx][:, :, :sequence_length]  # type:ignore
            for layer_idx, sequence_length in enumerate(context_cache_seq_lengths)
        ]

    return generated_ids, generated_logits, generated_logits_non_special_tokens_mask
