from ast import Dict
from dataclasses import dataclass
from string import Template
from typing import Any, Optional

import numpy as np
from datasets import Dataset
from numpy.random import Generator
from transformers import PreTrainedTokenizerBase


@dataclass
class ICLDataCollator:
    tokenizer: PreTrainedTokenizerBase
    dataset: Dataset
    sampler: Generator
    max_length: int = 4096
    template: Dict[str, str]
    n_icl: int = 1
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __post_init__(self):
        self.indices_range = np.arange(len(self.dataset))
        self.header = self.template.get("header", "")
        self.instruction_template = Template(self.template.get("instruction", ""))
        self.input_template = Template(self.template.get("input", ""))
        self.answer_template = Template(self.template.get("answer", ""))

    def sample_ic_examples(self):
        random_indices = self.sampler.choice(
            self.indices_range, self.n_icl, replace=False
        )
        examples = self.dataset.select(random_indices)
        return examples

    def format_prompt_str(self, x, add_answer=True):
        header = self.header
        instruction = self.instruction_template.safe_substitute(
            instruction=x["instruction"]
        )
        if x["input"] is not None and x["input"] != "":
            input_str = self.input_template.safe_substitute(input=x["input"])
        else:
            input_str = ""
        prompt_str = header + instruction + input_str
        if add_answer:
            answer = self.answer_template.safe_substitute(answer=x["answer"])
            prompt_str = prompt_str + answer
        return prompt_str

    def format_prompt_example(self, x):
        examples = self.sample_ic_examples()
        prompt_str = "\n\n".join(
            [self.format_prompt_str(e, add_answer=True) for e in examples]
        )
        current_instruction = self.format_prompt_str(x, add_answer=False)
        prompt_str = prompt_str + "\n\n" + current_instruction
        input_ids = self.tokenizer.encode(
            prompt_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )
        icl_input_ids = [self.tokenizer.bos_token_id] + input_ids
        current_instruction_ids = self.tokenizer.encode(
            current_instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )

        return icl_input_ids, current_instruction_ids

    def __call__(self, batch, return_tensors=None):
        icl_batch_formatted = []
        current_batch_formatted = []
        for x in batch:
            icl_input_ids, current_instruction_ids = self.format_prompt_example(x)
            icl_batch_formatted.append({"input_ids": icl_input_ids})
            current_batch_formatted.append({"input_ids": current_instruction_ids})

        return icl_batch_formatted, current_batch_formatted
