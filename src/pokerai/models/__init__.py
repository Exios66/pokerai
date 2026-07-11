"""Model constructors."""

from __future__ import annotations

import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from pokerai.config import GPT2Hyperparams, N_EMBD, N_HEAD, N_LAYER, N_POSITIONS


class Bigram(nn.Module):
    """Next-token table: row i = logits for the token after i."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        return self.table(idx)


def build_gpt2(
    tokenizer: PreTrainedTokenizerFast,
    hp: GPT2Hyperparams | None = None,
) -> GPT2LMHeadModel:
    hp = hp or GPT2Hyperparams()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=hp.n_positions,
        n_embd=hp.n_embd,
        n_layer=hp.n_layer,
        n_head=hp.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(config)


def default_gpt2_config_kwargs() -> dict:
    return {
        "n_positions": N_POSITIONS,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
    }
