import argparse
import os
import time
from pathlib import Path

import tensorrt_llm
from tensorrt_llm._deprecation import emit_engine_arch_deprecation
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.build import parallel_build
from tensorrt_llm.llmapi.kv_cache_type import KVCacheType
from tensorrt_llm.logger import logger
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.plugin import PluginConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_OUTPUT_ROOT = Path(
    os.getenv(
        "QWEN_MODEL_OUTPUT_ROOT",
        str(REPO_ROOT / "models" / "qwen"),
    )
)
DEFAULT_TLLM_CHECKPOINT_DIR = MODEL_OUTPUT_ROOT / "tllm_checkpoint"
DEFAULT_ENGINE_DIR = MODEL_OUTPUT_ROOT / "engine"
DEFAULT_HF_MODEL_DIR = MODEL_OUTPUT_ROOT / "hf_model"
DEFAULT_HIDDEN_STATE_ENGINE_DIR = MODEL_OUTPUT_ROOT / "hidden_state_engine"


def _import_hidden_state_builder():
    try:
        from trt_hidden_state_engine import build_hidden_state_engine
    except ImportError:
        from examples.models.qwen.trt_hidden_state_engine import build_hidden_state_engine

    return build_hidden_state_engine


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine_kind",
        type=str,
        default="generation",
        choices=["generation", "hidden_state"],
        help="Build a standard generation engine or a hidden-state engine that exposes last_hidden_state_output.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DEFAULT_TLLM_CHECKPOINT_DIR),
        help="The directory path that contains TensorRT-LLM checkpoint files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the serialized TensorRT-LLM engine files. Defaults to models/qwen/engine "
        "for generation builds and models/qwen/hidden_state_engine for hidden-state builds.",
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default=str(DEFAULT_HF_MODEL_DIR),
        help="Hugging Face Qwen model directory. Used when --engine_kind hidden_state.",
    )
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--max_output_len", type=int, default=128)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum total sequence length. Defaults to max_input_len + max_output_len.",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=None,
        help="Maximum number of batched input tokens after padding removal. Defaults to max_input_len.",
    )
    parser.add_argument("--max_beam_width", type=int, default=1)
    parser.add_argument("--max_prompt_embedding_table_size", type=int, default=0)
    parser.add_argument(
        "--kv_cache_type",
        type=str,
        default="paged",
        choices=["paged", "continuous", "disabled"],
        help="KV cache type for the built engine.",
    )
    parser.add_argument(
        "--remove_input_padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable input-padding removal when building a hidden-state engine. Enabled by default because it is "
        "the stable path for the Qwen hidden-state engine build in this environment.",
    )
    parser.add_argument(
        "--gpt_attention_plugin",
        type=str,
        default="auto",
        help="GPT attention plugin dtype for hidden-state engine builds. Use 'none' to disable it.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for hidden-state builds.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--gather_context_logits", action="store_true", default=False)
    parser.add_argument("--gather_generation_logits", action="store_true", default=False)
    return parser.parse_args()


def _to_kv_cache_type(name: str) -> KVCacheType:
    mapping = {
        "paged": KVCacheType.PAGED,
        "continuous": KVCacheType.CONTINUOUS,
        "disabled": KVCacheType.DISABLED,
    }
    return mapping[name]


def main():
    emit_engine_arch_deprecation("build_trtllm.py")
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    logger.set_level(args.log_level)

    default_output_dir = DEFAULT_HIDDEN_STATE_ENGINE_DIR if args.engine_kind == "hidden_state" else DEFAULT_ENGINE_DIR
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.engine_kind == "hidden_state":
        hf_model_dir = Path(args.hf_model_dir)
        if not hf_model_dir.exists():
            raise FileNotFoundError(
                f"Hugging Face model directory not found for hidden-state engine build: {hf_model_dir}"
            )

        build_hidden_state_engine = _import_hidden_state_builder()
        gpt_attention_plugin = None if args.gpt_attention_plugin == "none" else args.gpt_attention_plugin
        tik = time.time()
        engine_dir = build_hidden_state_engine(
            str(hf_model_dir),
            args.max_input_len,
            max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
            dtype=args.dtype,
            remove_input_padding=args.remove_input_padding,
            gpt_attention_plugin=gpt_attention_plugin,
            engine_dir=output_dir,
        )
        tok = time.time()
        elapsed = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
        print(f"qwen_hf_model_dir={hf_model_dir.resolve()}")
        print(f"qwen_hidden_state_engine_dir={engine_dir.resolve()}")
        print(f"Total time of building hidden-state engine: {elapsed}")
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"TensorRT-LLM checkpoint config not found: {config_path}")

    model_config = PretrainedConfig.from_json_file(str(config_path))
    max_seq_len = args.max_seq_len or (args.max_input_len + args.max_output_len)
    max_num_tokens = args.max_num_tokens or args.max_input_len

    tik = time.time()
    build_config = BuildConfig(
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        max_beam_width=args.max_beam_width,
        max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
        kv_cache_type=_to_kv_cache_type(args.kv_cache_type),
        gather_context_logits=args.gather_context_logits,
        gather_generation_logits=args.gather_generation_logits,
        strongly_typed=True,
        plugin_config=PluginConfig(),
    )
    parallel_build(
        model_config,
        str(checkpoint_dir),
        build_config,
        str(output_dir),
        workers=args.workers,
        log_level=args.log_level,
    )
    tok = time.time()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"qwen_tllm_checkpoint_dir={checkpoint_dir.resolve()}")
    print(f"qwen_tllm_engine_dir={output_dir.resolve()}")
    print(f"Total time of building all engines: {elapsed}")


if __name__ == "__main__":
    main()
