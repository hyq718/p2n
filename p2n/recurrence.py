"""Parameter-free recurrence attached only when --p2n is enabled.

The official GPTModel, TransformerBlock, TE layers, loss, optimizer, and
training schedule are retained. Hooks capture the input to the core and
replace its warm output with K additional shared-weight Jacobi updates.
"""

from __future__ import annotations

import random
import torch


def core_range(num_layers, start=None, end=None):
    """Return a zero-based half-open interval from one-based inclusive CLI bounds."""
    if (start is None) != (end is None):
        raise ValueError("Specify both --p2n-core-start and --p2n-core-end")
    if start is None:
        if num_layers < 3:
            raise ValueError("The default middle-third core requires at least three layers")
        start, end = num_layers // 3 + 1, 2 * num_layers // 3
    if not 1 <= start <= end <= num_layers:
        raise ValueError(f"Core must satisfy 1 <= start <= end <= {num_layers}")
    return start - 1, end


def shift_previous(states, tokens, eod_id):
    """Shift [sequence, batch, hidden] right; reset after an EOD token."""
    shifted = torch.cat((torch.zeros_like(states[:1]), states[:-1]), dim=0)
    if eod_id >= 0:
        valid = (tokens[:, :-1] != eod_id).transpose(0, 1).unsqueeze(-1)
        shifted = torch.cat((shifted[:1], shifted[1:] * valid), dim=0)
    return shifted


def enable_p2n(model, *, start, end, eod_id, seed, iteration_provider):
    """Attach the recurrence to an official GPTModel without changing its parameters.

    Supported pretraining scope: dense BF16 GPT, TP=PP=CP=1, no recomputation,
    no dropout, and no inference cache. Data parallelism uses official Megatron
    DDP and its optimizer. Unsupported features fail instead of silently changing
    the recurrence. Vanilla never calls this function and retains upstream support.
    """
    config = model.config
    if any(getattr(config, name) != 1 for name in (
        "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"
    )):
        raise ValueError("P2N currently requires TP=PP=CP=1")
    if config.recompute_granularity is not None or config.fp8 or config.cpu_offloading:
        raise ValueError("P2N requires BF16/FP32, no activation recomputation or CPU offloading")
    if config.hidden_dropout or config.attention_dropout:
        raise ValueError("P2N currently requires zero hidden and attention dropout")
    if getattr(config, "num_moe_experts", None) is not None:
        raise ValueError("P2N currently supports dense models only")
    if getattr(config, "enable_cuda_graph", False) or getattr(config, "external_cuda_graph", False):
        raise ValueError("P2N does not support CUDA graph capture")
    decoder = model.decoder
    if hasattr(decoder, "_p2n_handles"):
        raise ValueError("P2N is already enabled")
    if not 0 <= start < end <= len(decoder.layers):
        raise ValueError("Invalid core layer range")
    state = {"inside_update": False, "tokens": None, "core_input": None, "kwargs": None}

    def capture_tokens(_module, inputs, kwargs):
        tokens = kwargs.get("input_ids", inputs[0] if inputs else None)
        if tokens is None or tokens.ndim != 2:
            raise ValueError("P2N requires [batch, sequence] token IDs")
        if kwargs.get("inference_context") is not None or kwargs.get("inference_params") is not None:
            raise ValueError("P2N implements pretraining, not cached autoregressive inference")
        if kwargs.get("packed_seq_params") is not None:
            raise ValueError("P2N does not yet support THD packed sequence input")
        state["tokens"] = tokens

    def capture_prefix(_module, _inputs, kwargs):
        if not state["inside_update"]:
            state["core_input"] = kwargs["hidden_states"]
            state["kwargs"] = {key: value for key, value in kwargs.items() if key != "hidden_states"}

    def update_core(_module, _inputs, _kwargs, output):
        if state["inside_update"]:
            return None
        if state["tokens"] is None or state["core_input"] is None:
            raise RuntimeError("P2N input capture was not executed")
        k = random.Random(seed + iteration_provider()).choice((2, 3)) if model.training else 3
        decoder.p2n_last_k = k
        hidden, context = output
        state["inside_update"] = True
        try:
            for _ in range(k):
                hidden = state["core_input"] + shift_previous(hidden, state["tokens"], eod_id)
                for layer in decoder.layers[start:end]:
                    kwargs = dict(state["kwargs"])
                    kwargs["context"] = context
                    hidden, context = layer(hidden_states=hidden, **kwargs)
            return hidden, context
        finally:
            state.update(inside_update=False, core_input=None, tokens=None, kwargs=None)

    decoder._p2n_handles = [
        model.register_forward_pre_hook(capture_tokens, with_kwargs=True),
        decoder.layers[start].register_forward_pre_hook(capture_prefix, with_kwargs=True),
        decoder.layers[end - 1].register_forward_hook(update_core, with_kwargs=True),
    ]
    decoder.p2n_core_range = (start, end)
