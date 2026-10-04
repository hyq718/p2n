"""Official Megatron-LM GPT pretraining, optionally enabling P2N."""

from pathlib import Path
import sys

UPSTREAM = Path(__file__).resolve().parent / "vendor" / "Megatron-LM"
if not (UPSTREAM / "pretrain_gpt.py").is_file():
    raise RuntimeError("Initialize the pinned dependency with git submodule update --init")
sys.path.insert(0, str(UPSTREAM))

import pretrain_gpt as upstream
from megatron.training import get_args, get_tokenizer, print_rank_0, pretrain
from megatron.training import global_vars, checkpointing
from megatron.core.enums import ModelType
from p2n.recurrence import core_range, enable_p2n


def extra_arguments(parser):
    group = parser.add_argument_group("P2N")
    group.add_argument("--p2n", action="store_true", help="Enable shared-core P2N training")
    group.add_argument("--p2n-core-start", type=int, help="First core layer, one-based inclusive")
    group.add_argument("--p2n-core-end", type=int, help="Last core layer, one-based inclusive")
    group.add_argument("--eod-id", type=int, default=None,
                       help="Override NullTokenizer's EOD ID for existing pretokenized data")
    return parser


def model_provider(*args, **kwargs):
    model = upstream.model_provider(*args, **kwargs)
    options = get_args()
    if options.p2n:
        start, end = core_range(options.num_layers, options.p2n_core_start, options.p2n_core_end)
        enable_p2n(model, start=start, end=end, eod_id=get_tokenizer().eod,
                   seed=options.seed, iteration_provider=lambda: get_args().curr_iteration)
        print_rank_0(f"P2N core layers {start + 1}..{end}; warm pass + K=2/3; eval K=3")
    elif options.p2n_core_start is not None or options.p2n_core_end is not None:
        raise ValueError("Core layer bounds require --p2n")
    count = sum(parameter.numel() for parameter in model.parameters())
    print_rank_0(f"MODEL_PARAMETERS={count}; P2N={options.p2n}; ATTENTION={options.attention_backend}")
    return model


def install_tokenizer_metadata_override():
    original = global_vars.build_tokenizer

    def build_tokenizer(options):
        tokenizer = original(options)
        if options.eod_id is not None:
            if options.tokenizer_type != "NullTokenizer":
                raise ValueError("--eod-id is only for NullTokenizer with pretokenized data")
            if not 0 <= options.eod_id < tokenizer.vocab_size:
                raise ValueError("EOD ID must be inside the tokenizer vocabulary")
            tokenizer._eod_id = options.eod_id
        return tokenizer

    global_vars.build_tokenizer = build_tokenizer


def install_checkpoint_validation():
    original = checkpointing.check_checkpoint_args

    def check_checkpoint_args(saved):
        options = get_args()
        if bool(getattr(saved, "p2n", False)) != options.p2n:
            raise ValueError("Checkpoint P2N mode differs from this run; use --finetune to initialize weights")
        if options.p2n:
            previous = core_range(saved.num_layers, getattr(saved, "p2n_core_start", None),
                                  getattr(saved, "p2n_core_end", None))
            current = core_range(options.num_layers, options.p2n_core_start, options.p2n_core_end)
            if previous != current or getattr(saved, "eod_id", None) != options.eod_id:
                raise ValueError("Checkpoint P2N core range or EOD ID differs from this run")
        original(saved)

    checkpointing.check_checkpoint_args = check_checkpoint_args


if __name__ == "__main__":
    install_tokenizer_metadata_override()
    install_checkpoint_validation()
    upstream.train_valid_test_datasets_provider.is_distributed = True
    pretrain(upstream.train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder, upstream.forward_step,
             args_defaults={"tokenizer_type": "NullTokenizer"},
             extra_args_provider=extra_arguments)
