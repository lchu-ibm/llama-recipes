# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import time

import fire
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from transformers import (
    LlamaForCausalLM,
    LlamaConfig
)

from configs import fsdp_config, train_config

from utils.config_utils import (
    update_config,
)
from utils.train_utils import (
    setup,
    setup_environ_flags,
    get_policies
)

from model_checkpointing import checkpoint_handler


def main(**kwargs):

    update_config((train_config, fsdp_config), **kwargs)

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)
    setup_environ_flags(rank)

    start = time.perf_counter()
    llama_config = LlamaConfig.from_pretrained(train_config.model_name)
    with torch.device("meta"):
        model = LlamaForCausalLM(llama_config)
    end = time.perf_counter()
    if rank == 0:
        print(f"cpu model loading time = {end-start:.4f}\n")

    mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)

    start = time.perf_counter()
    model = FSDP(
        model,
        auto_wrap_policy= wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False),
    )
    end = time.perf_counter()
    if rank == 0:
        print(f"FSDP construction time = {end-start:.4f}\n")

    start = time.perf_counter()
    checkpoint_handler.load_model_sharded(model, rank, train_config)
    end = time.perf_counter()
    if rank == 0:
        print(f"sharded checkpoint loading time = {end - start:.4f}\n")


if __name__ == "__main__":
    fire.Fire(main)
