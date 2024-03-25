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
from typing import Optional
from omegaconf import OmegaConf
from pathlib import Path
import hydra
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModel

from ictrainer import  ICTrainer
from trl.trainer import PPOConfig, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available


tqdm.pandas()



@hydra.main(config_path="./config", config_name="llama_async.yaml")
def main(cfg):
    

    def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one shoul
        customize this function to train the model on its own dataset.

        Args:
            query_dataset (`str`):s
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # load imdb with datasets
        ds = load_dataset(query_dataset, split="train")
    
        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds


    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(ppo_config, ppo_config.query_dataset)


    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}


    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    if not args.use_peft:
        ref_model = AutoModel.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
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
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

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
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
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
        query_ref_tensors = [torch.cat([q, c, r]) for q, c ,r in zip(queries, contexts)]


    #sample examples from training set
    
    # 1 Generate response given query
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    # 2 Create Incontext queries 
    # query_ref <= query + example




    batch["response"] = tokenizer.batch_decode(response_tensors)

    texts = [q + r for q, r in zip(batch["query"], batch["response"])]



    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, query_ref_tensors)
    # ppo_trainer.log_stats(stats, batch, columns_to_log=["query", "response", "ref_response", "ref_rewards"])



if __name__=="__main__":
    main()
