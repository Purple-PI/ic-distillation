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
    template: str
    n_icl: int = 1
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __post_init__(self):
        self.indices_range = np.arange(len(self.dataset))
        self.template = Template(self.template)

    def sample_ic_examples(self):
        random_indices = self.sampler.choice(
            self.indices_range, self.n_icl, replace=False
        )
        examples = self.dataset.select(random_indices)
        return examples

    def format_prompt_example(self, x):
        examples = self.sample_ic_examples()
        prompt_str = "\n\n".join([self.template.safe_substitute(**e) for e in examples])

        prompt_str = (
            prompt_str
            + "\n\n"
            + self.template.safe_substitute(
                instruction=x["instruction"], input=x["input"]
            )
        )
        input_ids = self.tokenizer.encode(
            prompt_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )
        input_ids = [self.tokenizer.bos_token_id] + input_ids

        return input_ids

    def __call__(self, batch, return_tensors=None):
        batch_formatted = [{"input_ids": self.format_prompt_example(x)} for x in batch]
        main_features = self.tokenizer.pad(
            batch_formatted,
            padding=True,
            truncation=True,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return main_features
