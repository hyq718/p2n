from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from p2n.recurrence import core_range, enable_p2n, shift_previous


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.calls = 0

    def forward(self, *, hidden_states, context=None, **kwargs):
        self.calls += 1
        return torch.tanh(self.linear(hidden_states)), context


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                      context_parallel_size=1, recompute_granularity=None,
                                      fp8=None, cpu_offloading=False, hidden_dropout=0,
                                      attention_dropout=0)
        self.embedding = nn.Embedding(10, 4)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(Layer() for _ in range(6))
        self.head = nn.Linear(4, 10)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids).transpose(0, 1)
        for layer in self.decoder.layers:
            hidden, _ = layer(hidden_states=hidden)
        return self.head(hidden)


def test_layer_range_and_boundary_shift():
    assert core_range(6) == (2, 4)
    assert core_range(12) == (4, 8)
    assert core_range(21) == (7, 14)
    assert core_range(6, 1, 6) == (0, 6)
    for args in [(6, 0, 3), (6, 3, None), (6, 5, 2), (6, 1, 7)]:
        with pytest.raises(ValueError):
            core_range(*args)
    tokens = torch.tensor([[2, 3, 0, 4, 5]])
    states = torch.arange(5.0).reshape(5, 1, 1).requires_grad_()
    shifted = shift_previous(states, tokens, 0)
    assert shifted.flatten().tolist() == [0, 0, 1, 0, 3]
    shifted.sum().backward()
    assert states.grad.flatten().tolist() == [1, 1, 0, 1, 0]


@pytest.mark.parametrize("bounds", [(2, 4), (0, 6), (1, 5)])
@pytest.mark.parametrize("iteration", [0, 2])  # seed 42 selects K=2 and K=3 respectively
def test_hook_recurrence_matches_reference_outputs_and_all_parameter_gradients(bounds, iteration):
    torch.manual_seed(7)
    model = Model()
    reference = deepcopy(model)
    before = set(model.state_dict())
    begin, end = bounds
    enable_p2n(model, start=begin, end=end, eod_id=0, seed=42,
               iteration_provider=lambda: iteration)
    assert set(model.state_dict()) == before  # no new parameters or checkpoint keys
    tokens = torch.tensor([[2, 0, 3, 4, 5], [4, 5, 0, 6, 7]])
    actual = model(tokens)
    k = model.decoder.p2n_last_k
    assert k in (2, 3)
    hidden = reference.embedding(tokens).transpose(0, 1)
    for layer in reference.decoder.layers[:begin]:
        hidden, _ = layer(hidden_states=hidden)
    prefix = hidden
    for layer in reference.decoder.layers[begin:end]:
        hidden, _ = layer(hidden_states=hidden)
    for _ in range(k):
        feedback = torch.zeros_like(hidden)
        feedback[1:] = hidden[:-1] * (tokens[:, :-1] != 0).transpose(0, 1).unsqueeze(-1)
        hidden = prefix + feedback
        for layer in reference.decoder.layers[begin:end]:
            hidden, _ = layer(hidden_states=hidden)
    for layer in reference.decoder.layers[end:]:
        hidden, _ = layer(hidden_states=hidden)
    expected = reference.head(hidden)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (_, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad)
    assert [layer.calls for layer in model.decoder.layers] == [
        k + 1 if begin <= index < end else 1 for index in range(6)
    ]
    model.eval()
    model(tokens)
    assert model.decoder.p2n_last_k == 3
