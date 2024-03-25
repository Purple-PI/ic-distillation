# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/ppo.py \
    --log_with=wandb
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hydra
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from ictrainer import ICTrainer
from omegaconf import OmegaConf
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, HfArgumentParser, pipeline, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import PPOConfig

from ic_distillation.utils import get_env

tqdm.pandas()


@hydra.main(config_path="../config", config_name="config.yaml")
def main(cfg):

    # We retrieve the dataloader by calling the `build_dataset` function.
    DATA_PATH = get_env("DATA_PATH")
    dataset = load_from_disk(DATA_PATH / cfg.data_path)

    # set seed before initializing value head for deterministic eval
    set_seed(cfg.training.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    if not cfg.peft.use_peft:
        ref_model = AutoModel.from_pretrained(cfg.model.path)
        device_map = None
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}

    model = AutoModel.from_pretrained(
        ppo_config.model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
    )

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        elif is_npu_available():
            device = "npu:0"
        else:
            device = (
                0 if torch.cuda.is_available() else "cpu"
            )  # to avoid a `pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    task, model_name = ppo_config.reward_model.split(":")

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }

    for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        query_ref_tensors = [torch.cat([q, c, r]) for q, c, r in zip(queries, contexts)]

    # sample examples from training set

    # 1 Generate response given query
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        generate_ref_response=True,
        **generation_kwargs
    )
    # 2 Create Incontext queries
    # query_ref <= query + example

    batch["response"] = tokenizer.batch_decode(response_tensors)

    texts = [q + r for q, r in zip(batch["query"], batch["response"])]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, query_ref_tensors)
    # ppo_trainer.log_stats(stats, batch, columns_to_log=["query", "response", "ref_response", "ref_rewards"])


if __name__ == "__main__":
    main()
