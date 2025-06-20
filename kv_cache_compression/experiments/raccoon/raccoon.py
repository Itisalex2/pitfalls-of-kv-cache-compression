import logging
from collections import defaultdict
from pathlib import Path

from kvpress import ScorerPress
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.cache_utils import DynamicCache
from rouge_score import rouge_scorer

from ..compressed_context_evaluation import (
    generate_from_compressed_context,
    get_logits_entropy,
    get_context_cache,
)
from ..utils import (
    write_jsonl,
    get_system_span,
    keep_rates_from_kept_indices,
    segment_keep_pcts,
)


class Raccoon:
    """
    Class to run the RACCOON evaluations.
    Ref: https://github.com/M0gician/RaccoonBench/blob/main/Raccoon/raccoon.py
    We modify it to test on local models and datasets.

    Args:
        model (PreTrainedModel): The language model to evaluate
        tokenizer (AutoTokenizer): The tokenizer for the model
        press (ScorerPress): The KV-Cache compression method
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 256.
        generate_entropy (bool, optional): Whether to compute the entropy of the generated responses. Defaults to True.
        num_responses (int, optional): Number of responses to generate per attack. Defaults to 1.
        system_prompts_and_keys (list[tuple[str, str]]): List of tuples of system prompts and their keys.
        attack_prompts (dict[str, list], optional): Dictionary of attack categories to list of prompts
        defense_templates (dict[str, str], optional): Dictionary of defense names to defense templates.
        save_path (Path, optional): Path to save the results. Defaults to Path("raccoon_results").
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        press: ScorerPress,
        system_prompts_and_keys: list[tuple[str, str]],
        attack_prompts: dict[str, list],
        defense_templates: dict[str, str],
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        generate_entropy: bool = True,
        num_responses: int = 1,
        save_path: Path = Path("raccoon_results"),
        analyze_kept_tokens: bool = True,
        use_automated_spans: bool = False,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.tokenizer = tokenizer
        self.press = press
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.generate_entropy = generate_entropy
        self.num_responses = num_responses
        self.system_prompts_and_keys = system_prompts_and_keys
        self.attack_prompts = attack_prompts
        self.defense_templates = defense_templates
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.analyze_kept_tokens = analyze_kept_tokens
        self.use_automated_spans = use_automated_spans
        self.logger = logging.getLogger(__name__)
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def attack(
        self,
        sys_prompt: str,
        atk_prompt: str,
        context_cache: DynamicCache,
        system_prompt_content_len: int,
        num_responses: int = 1,
    ) -> list[dict]:
        """
        Generate attack responses given the system prompt (defended) and attack prompt

        Returns:
            list[dict]: A list of dictionaries containing the response and entropy
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": atk_prompt},
        ]
        messages_ids = self.tokenizer.apply_chat_template(  # type: ignore
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        do_sample = self.temperature > 0

        responses: list[dict] = []
        for _ in range(num_responses):
            generated_ids, logits, logits_non_special_tokens_mask = (
                generate_from_compressed_context(
                    self.model,
                    self.tokenizer,
                    messages_ids,
                    system_prompt_content_len,
                    context_cache,
                    do_sample=do_sample,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            )

            if self.generate_entropy:
                entropy = get_logits_entropy(logits, logits_non_special_tokens_mask)
            else:
                entropy = None

            response: str = self.tokenizer.decode(  # type: ignore
                generated_ids[0], skip_special_tokens=True
            )
            response_entry = {
                "response": response,
                "response_entropy": entropy,
            }
            responses.append(response_entry)

        return responses

    def evaluate(
        self, sys_prompt: str, atk_responses: list[str]
    ) -> dict[str, float | list[float]]:
        """Compare the sys_prompt to the attack responses"""
        individual_scores: dict[str, list[float]] = defaultdict(list)
        sums = defaultdict(float)
        for r in atk_responses:
            for k, s in self.scorer.score(sys_prompt, r).items():
                individual_scores[f"{k}_recall_individual"].append(s.recall)
                individual_scores[f"{k}_precision_individual"].append(s.precision)
                individual_scores[f"{k}_f1_individual"].append(s.fmeasure)

                sums[f"{k}_recall"] += s.recall
                sums[f"{k}_precision"] += s.precision
                sums[f"{k}_f1"] += s.fmeasure
        n = max(1, len(atk_responses))

        scores: dict[str, float | list[float]] = {k: v / n for k, v in sums.items()}
        scores.update(individual_scores)
        return scores

    def run_benchmark(
        self,
    ) -> None:
        for defense_template_name, defense_template in tqdm(
            self.defense_templates.items(), desc="Defense templates"
        ):
            assert "$system_prompt" in defense_template, (
                "Defense template must contain $system_prompt"
            )
            defense_template_without_system_prompt = defense_template.replace(
                "$system_prompt", ""
            )  # For evaluation of defense alone

            for system_prompt, key in tqdm(
                self.system_prompts_and_keys, desc="System prompts"
            ):
                try:
                    defended_system_prompt = defense_template.replace(
                        "$system_prompt", system_prompt
                    )

                    defended_system_prompt_content_ids = (
                        self.tokenizer.apply_chat_template(  # type:ignore
                            [{"role": "system", "content": defended_system_prompt}],
                            return_tensors="pt",
                        ).to(self.device)
                    )
                    defended_system_prompt_content_len = (
                        defended_system_prompt_content_ids.shape[1]
                    )
                    defended_system_prompt_content_str_tokens = (
                        self.tokenizer.convert_ids_to_tokens(  # type: ignore
                            defended_system_prompt_content_ids[0].tolist(),
                            skip_special_tokens=False,
                        )
                    )
                    defended_system_prompt_ids_and_str_tokens = list(
                        zip(
                            range(defended_system_prompt_content_len),
                            defended_system_prompt_content_ids[0].tolist(),
                            defended_system_prompt_content_str_tokens,
                        )
                    )  # type: ignore

                    if self.analyze_kept_tokens:
                        system_prompt_span = get_system_span(
                            self.tokenizer,
                            defended_system_prompt_content_ids[0].tolist(),
                            system_prompt,
                        )
                        self.press.clear_analysis()
                    else:
                        system_prompt_span = None

                    if self.use_automated_spans:
                        # Automatically detect defense and system instruction spans for fair eviction methods
                        defense_span = get_system_span(
                            self.tokenizer,
                            defended_system_prompt_content_ids[0].tolist(),
                            defense_template_without_system_prompt,
                        )
                        sys_instr_span = get_system_span(
                            self.tokenizer,
                            defended_system_prompt_content_ids[0].tolist(),
                            system_prompt,
                        )
                        if (
                            defense_span[1] != sys_instr_span[0]
                            and defense_span[0] != sys_instr_span[1]
                        ):
                            if sys_instr_span[1] == defense_span[0] + 1:
                                self.logger.info(
                                    f"Adjusting faulty system instruction span to be contiguous with defense span: {defense_span} and {sys_instr_span}. Key: {key}."
                                )
                                sys_instr_span = (sys_instr_span[0], defense_span[0])
                                self.logger.info(
                                    f"New system instruction span: {sys_instr_span}. Key: {key}."
                                )
                            else:
                                self.logger.warning(
                                    f"Automated spans for defense and system prompt are overlapping or non-contiguous: {defense_span} and {sys_instr_span}. Key: {key}. This may affect eviction methods that rely on these spans."
                                )
                        assert hasattr(self.press, "defense_span"), (
                            "Press must have defense_span attribute if use_automated_spans is True"
                        )
                        assert hasattr(self.press, "sys_instr_span"), (
                            "Press must have sys_instr_span attribute if use_automated_spans is True"
                        )
                        self.press.defense_span = defense_span  # type: ignore
                        self.press.sys_instr_span = sys_instr_span  # type: ignore

                    context_cache = get_context_cache(
                        self.model, defended_system_prompt_content_ids, self.press
                    )
                    context_cache_len = context_cache[0][0].shape[2]  # type: ignore

                    if self.analyze_kept_tokens:
                        assert isinstance(system_prompt_span, tuple)
                        per_layer_keep_rate, overall_keep_rate = (
                            keep_rates_from_kept_indices(
                                self.press.kept_indices_by_layer,
                                defended_system_prompt_content_len,
                            )
                        )

                        # Compression ratio 0.0
                        if not self.press.kept_indices_by_layer:
                            overall_keep_rate = [
                                1.0
                            ] * defended_system_prompt_content_len

                        system_keep_percentage, defense_keep_percentage = (
                            segment_keep_pcts(
                                overall_keep_rate,
                                system_prompt_span,
                                defended_system_prompt_content_len,
                            )
                        )
                    else:
                        per_layer_keep_rate = {}
                        overall_keep_rate = []
                        system_keep_percentage = -1.0
                        defense_keep_percentage = -1.0

                    for (
                        attack_prompt_category,
                        attack_prompts,
                    ) in self.attack_prompts.items():
                        for idx, attack_prompt in enumerate(attack_prompts):
                            self.logger.info(
                                f"Running RACCOON with defense: {defense_template_name}, system prompt: {system_prompt}, system_prompt_key: {key}, attack prompt category: {attack_prompt_category}, prompt index: {idx}"
                            )
                            responses_dict: list[dict] = self.attack(
                                sys_prompt=defended_system_prompt,
                                atk_prompt=attack_prompt,
                                context_cache=context_cache,
                                system_prompt_content_len=defended_system_prompt_content_len,
                                num_responses=self.num_responses,
                            )
                            responses: list[str] = [
                                r["response"] for r in responses_dict
                            ]
                            system_prompt_scores = self.evaluate(
                                system_prompt, responses
                            )
                            defended_system_prompt_scores = self.evaluate(
                                defended_system_prompt, responses
                            )
                            defense_only_scores = self.evaluate(
                                defense_template_without_system_prompt,
                                responses,
                            )
                            result_entry = {
                                "compression_ratio": f"{self.press.compression_ratio:.2f}",
                                "defense_template_name": defense_template_name,
                                "system_prompt": system_prompt,
                                "system_prompt_key": key,
                                "system_prompt_cache_length": context_cache_len,
                                "attack_prompt": attack_prompt,
                                "attack_prompt_index": idx,
                                "attack_prompt_category": attack_prompt_category,
                                "responses": responses_dict,
                                "system_prompt_scores": system_prompt_scores,
                                "defended_system_prompt_scores": defended_system_prompt_scores,
                                "defense_only_scores": defense_only_scores,
                                "system_prompt_token_span": system_prompt_span,
                                "defended_system_prompt_ids_and_str_tokens": defended_system_prompt_ids_and_str_tokens,
                                "keep_rate_per_layer": per_layer_keep_rate,
                                "keep_rate_overall": overall_keep_rate,
                                "system_keep_pct": system_keep_percentage,
                                "defense_keep_pct": defense_keep_percentage,
                            }
                            jsonl_path = (
                                self.save_path
                                / defense_template_name
                                / attack_prompt_category
                                / f"prompt_{idx}.jsonl"
                            )
                            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                            write_jsonl(jsonl_path, [result_entry], append=True)
                            self.logger.info(
                                f"Saved results to {jsonl_path} for attack prompt index {idx}"
                            )
                except Exception as e:
                    self.logger.error(f"Error generating response for item {key}: {e}")
                    self.logger.error(f"Skipping item {key}.")
                    continue
