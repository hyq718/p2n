"""Run with one GPU: real TE/FlashAttention output, gradient and kernel checks."""
from pathlib import Path
import sys
import json
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/Megatron-LM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from p2n.recurrence import enable_p2n


def build_model():
    config = TransformerConfig(num_layers=6, hidden_size=128, ffn_hidden_size=512,
        num_attention_heads=2, num_query_groups=1, kv_channels=64,
        normalization="RMSNorm", qk_layernorm=True, layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        add_bias_linear=False, add_qkv_bias=False, hidden_dropout=0, attention_dropout=0,
        params_dtype=torch.bfloat16, bf16=True, attention_backend=AttnBackend.flash,
        gradient_accumulation_fusion=False, bias_activation_fusion=True,
        bias_dropout_fusion=True, apply_rope_fusion=True, cross_entropy_loss_fusion=True)
    return GPTModel(config, get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
                    vocab_size=128, max_sequence_length=64, parallel_output=True,
                    share_embeddings_and_output_weights=False, position_embedding_type="rope",
                    rotary_base=1000000).cuda()


torch.cuda.set_device(0)
dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29796", rank=0, world_size=1)
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
torch.manual_seed(42)
model_parallel_cuda_manual_seed(42)
tokens = torch.randint(1, 128, (2, 64), device="cuda")
tokens[:, 7] = 0
positions = torch.arange(64, device="cuda").expand(2, -1)
results = []
for iteration, expected_k in [(0, 2), (2, 3)]:
    model, reference = build_model(), build_model()
    reference.load_state_dict(model.state_dict())
    keys = set(model.state_dict())
    enable_p2n(model, start=2, end=4, eod_id=0, seed=42, iteration_provider=lambda: iteration)
    assert set(model.state_dict()) == keys

    def reference_forward(decoder, hidden_states, attention_mask, **kwargs):
        def layers(hidden, begin, end):
            for layer in decoder.layers[begin:end]:
                hidden, _ = layer(hidden_states=hidden, attention_mask=attention_mask, **kwargs)
            return hidden
        prefix = layers(hidden_states, 0, 2)
        hidden = layers(prefix, 2, 4)
        for _ in range(expected_k):
            feedback = torch.zeros_like(hidden)
            feedback[1:] = hidden[:-1] * (tokens[:, :-1] != 0).T.unsqueeze(-1)
            hidden = layers(prefix + feedback, 2, 4)
        hidden = layers(hidden, 4, 6)
        return decoder.final_layernorm(hidden)

    reference.decoder.forward = MethodType(reference_forward, reference.decoder)
    actual = model(tokens, positions, None)
    expected = reference(tokens, positions, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert model.decoder.p2n_last_k == expected_k
    results.append({"k": expected_k, "outputs_equal": True, "all_parameter_gradients_equal": True})
    del model, reference

vanilla = build_model()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA]) as profile:
    vanilla(tokens, positions, None).float().sum().backward()
    torch.cuda.synchronize()
kernel_names = [event.name for event in profile.events() if "flash" in event.name.lower()]
assert kernel_names, "No FlashAttention kernel was observed"
print(json.dumps({"gpu_checks": results, "flash_attention_observed": True,
                  "flash_kernel_examples": sorted(set(kernel_names))[:4]}), flush=True)
parallel_state.destroy_model_parallel()
dist.destroy_process_group()
