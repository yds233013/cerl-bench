"""Loading the policy, and the LoRA adapter that is the only thing trained."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cerl_rl import protocol


def pick_device() -> torch.device:
    """MPS or nothing.

    Silently falling back to CPU would turn a two-hour budget into an
    unfinishable run and would misreport what was measured, so an unavailable
    MPS is an error rather than a fallback.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available; this pilot is sized for Apple Silicon and "
            "will not silently train on CPU",
        )
    return torch.device("mps")


def load_tokenizer() -> Any:
    tok = AutoTokenizer.from_pretrained(
        protocol.MODEL_ID, revision=protocol.MODEL_REVISION,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(device: torch.device, *, lora_rank: int = 8) -> Any:
    """Base weights frozen in bf16; a small LoRA adapter in fp32 on top.

    bf16 for the base halves resident memory on a machine that is already deep
    into swap. The adapter stays fp32 because it is what the optimizer touches,
    and bf16 optimizer states on a rank-8 adapter buy nothing while costing
    numerical headroom.
    """
    model = AutoModelForCausalLM.from_pretrained(
        protocol.MODEL_ID,
        revision=protocol.MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,           # deterministic rollouts; nothing to regularize at this size
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.to(device)
    return model


def adapter_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
