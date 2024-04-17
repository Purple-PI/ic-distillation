from dataclasses import dataclass
from typing import Optional

import hydra
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from ic_distillation.config import TrainConfig
from ic_distillation.decoding import (
    ContextAwareDecoderConfig,
    ContrastiveDecoder,
    ContrastiveDecoderConfig,
    PMIDecoder,
    PMIDecoderConfig,
)
from ic_distillation.utils import get_dtype, get_env


@dataclass
class ContextDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def format_prompt_str(self, x):

        prompt_str = f"{x['context']}.\n\nQuestion:\n{x['question']}\n\nResponse:\n"

        return prompt_str

    def __call__(self, batch, return_tensors=None):
        formatted_batch = []
        if return_tensors is None:
            return_tensors = self.return_tensors

        for x in batch:
            formatted_batch.append(self.format_prompt_str(x))

        features = self.tokenizer(
            formatted_batch,
            padding=True,
            return_tensors=return_tensors,
            add_special_tokens=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=True,
            max_length=self.max_length,
        )

        return features


@dataclass
class NoContextDataCollator(ContextDataCollator):
    def format_prompt_str(self, x):
        prompt_str = f"Question:\n{x['question']}\n\nResponse:\n"
        return prompt_str


class ExpertGuidedDataCollator:
    def __init__(self, main_data_collator, *args):
        self.main_data_collator = main_data_collator
        self.weak_data_collators = args

    def __call__(self, batch):
        main_input = self.main_data_collator(batch)
        weak_inputs = [
            weak_collator(batch) for weak_collator in self.weak_data_collators
        ]
        return main_input, weak_inputs


def load_expert_guided_decoder(config, main_model, load_model=True):
    main_config = config.model
    noise_config = config.secondary

    assert (
        noise_config is not None
    ), "You need to supply either a noise model or a weak model."

    if load_model:
        if noise_config.path == main_config.path:
            print(f"Using same backbone for noise model")
            noise_model = main_model
        else:
            CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
            model_path = CHECKPOINT_PATH / config.noise.path
            dtype = get_dtype(config.eval.dtype)
            attn_implementation = config.generation.attn_implementation
            tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
            noise_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                pad_token_id=tokenizer.pad_token_id,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation=attn_implementation,
            )

    else:
        noise_model = None

    if load_model:
        model_type = config.model_type
        if model_type == "contrastive":
            expert_guided_config = ContrastiveDecoderConfig(
                vocab_size=main_model.config.vocab_size,
                decoder_start_token_id=main_model.config.decoder_start_token_id,
                alpha=config.generation.contrastive_alpha,
            )
            model_class = ContrastiveDecoder
        elif model_type == "pmi":
            expert_guided_config = PMIDecoderConfig(
                vocab_size=main_model.config.vocab_size,
                decoder_start_token_id=main_model.config.decoder_start_token_id,
                lambd=config.generation.pmi_lambda,
                tau=config.generation.pmi_tau,
            )
            model_class = PMIDecoder
        elif model_type == "context-aware":
            expert_guided_config = ContextAwareDecoderConfig(
                vocab_size=main_model.config.vocab_size,
                decoder_start_token_id=main_model.config.decoder_start_token_id,
                alpha=config.generation.context_aware_alpha,
            )
            model_class = ContrastiveDecoder
        else:
            raise ValueError(f"Model type {model_type} not recognized")

        expert_guided_decoder = model_class(
            main_model, noise_model, config=expert_guided_config
        )
    else:
        expert_guided_decoder = None
    return expert_guided_decoder


def get_model_and_tokenizer(cfg):
    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")

    model_path = CHECKPOINT_PATH / cfg.model.path
    tokenizer_path = model_path

    dtype = get_dtype(cfg.generation.dtype)

    attn_implementation = cfg.generation.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        pad_token_id=tokenizer.pad_token_id,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )

    # model = torch.compile(model, mode="reduce-overhead")

    return model, tokenizer


def load_data_collator_model_tokenizer(config):
    model, tokenizer = get_model_and_tokenizer(config)
    base_data_collator = ContextDataCollator(tokenizer=tokenizer)
    secondary_data_collator = NoContextDataCollator(tokenizer=tokenizer)

    assert (
        config.secondary is not None
    ), "Noise model must be provided in guided decoding system."
    model = load_expert_guided_decoder(config, model)

    data_collator = ExpertGuidedDataCollator(
        base_data_collator, secondary_data_collator
    )

    return data_collator, model, tokenizer


@hydra.main(config_path="../config", config_name="config.yaml")
def main(cfg):
    load_dotenv()
    dataset = [
        {
            "context": "Current year is 2025 and Trump won the elections.",
            "question": "Who is the president of the US?",
        },
        {
            "context": "The year is 2024 and after a close race, Kamala Harris has been inaugurated as the President.",
            "question": "Who is the president of the US?",
        },
    ]

    contrastive_data_collator, contrastive_model, tokenizer = (
        load_data_collator_model_tokenizer(cfg)
    )

    base_model = contrastive_model.main_model
    base_data_collator = contrastive_data_collator.main_data_collator

    # Running base generation

    base_inputs = base_data_collator(dataset)
    base_inputs = {
        "input_ids": base_inputs["input_ids"].to(base_model.device),
        "attention_mask": base_inputs["attention_mask"].to(base_model.device),
    }
    gen_config = {
        "max_new_tokens": 10,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    base_outputs = base_model.generate(
        output_scores=True,
        return_dict_in_generate=True,
        **gen_config,
        **base_inputs,
    )
    print(
        "Base model output:",
        tokenizer.batch_decode(base_outputs.sequences.cpu(), skip_special_tokens=True),
    )

    contrastive_inputs, weak_contrastive_inputs = contrastive_data_collator(dataset)
    contrastive_inputs = {
        "input_ids": contrastive_inputs["input_ids"].to(base_model.device),
        "attention_mask": contrastive_inputs["attention_mask"].to(base_model.device),
        "weak_inputs": [
            {
                "input_ids": weak_batch["input_ids"].to(base_model.device),
                "attention_mask": weak_batch["attention_mask"].to(base_model.device),
            }
            for weak_batch in weak_contrastive_inputs
        ],
    }

    contrastive_outputs = contrastive_model.generate(
        output_scores=True,
        return_dict_in_generate=True,
        **gen_config,
        **contrastive_inputs,
    )

    print(
        "Contrastive model output:",
        tokenizer.batch_decode(
            contrastive_outputs.sequences.cpu(), skip_special_tokens=True
        ),
    )


if __name__ == "__main__":
    main()
