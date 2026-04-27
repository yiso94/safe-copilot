import argparse
import json
import os
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tensorrt_llm
import torch
from tensorrt_llm import AutoModelForCausalLM
from tensorrt_llm._deprecation import emit_engine_arch_deprecation
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_OUTPUT_ROOT = Path(
    os.getenv(
        "QWEN_MODEL_OUTPUT_ROOT",
        str(REPO_ROOT / "models" / "qwen"),
    )
)
DEFAULT_HF_OUTPUT_DIR = MODEL_OUTPUT_ROOT / "hf_model"
DEFAULT_TLLM_CHECKPOINT_DIR = MODEL_OUTPUT_ROOT / "tllm_checkpoint"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, required=True)
    parser.add_argument(
        "--hf_output_dir",
        type=str,
        default=str(DEFAULT_HF_OUTPUT_DIR),
        help="Where to save the extracted Hugging Face Qwen submodel when model_dir points to a multimodal model.",
    )
    parser.add_argument("--tp_size", type=int, default=1, help="N-way tensor parallelism size")
    parser.add_argument("--pp_size", type=int, default=1, help="N-way pipeline parallelism size")
    parser.add_argument("--cp_size", type=int, default=1, help="N-way context parallelism size")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="The data type for the converted TRT-LLM checkpoint.",
    )
    parser.add_argument("--load_model_on_cpu", action="store_true")
    parser.add_argument(
        "--use_parallel_embedding",
        action="store_true",
        default=False,
        help="Enable embedding parallelism.",
    )
    parser.add_argument(
        "--embedding_sharding_dim",
        type=int,
        default=0,
        choices=[0, 1],
        help="Shard embeddings on vocab dim (0) or hidden dim (1).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_TLLM_CHECKPOINT_DIR),
        help="The path to save the TensorRT-LLM checkpoint.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="The number of workers for converting checkpoint in parallel.",
    )
    parser.add_argument(
        "--moe_tp_size",
        type=int,
        default=-1,
        help="N-way tensor parallelism size for MoE, default is tp_size.",
    )
    parser.add_argument(
        "--moe_ep_size",
        type=int,
        default=-1,
        help="N-way expert parallelism size for MoE, default is 1.",
    )
    return parser.parse_args()


def _local_files_only(model_dir: str) -> bool:
    return Path(model_dir).exists()


def _load_hf_config(model_dir: str):
    return AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=_local_files_only(model_dir),
    )


def _is_qwen_language_model(config) -> bool:
    if getattr(config, "llm_config", None) is not None:
        return False
    model_type = getattr(config, "model_type", "")
    if isinstance(model_type, str) and model_type.startswith("qwen"):
        return True
    architectures = getattr(config, "architectures", None) or []
    return any("Qwen" in architecture and "CausalLM" in architecture for architecture in architectures)


def _patch_rope_scaling(config_path: Path) -> None:
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    rope = config.get("rope_scaling")
    if isinstance(rope, dict) and rope.get("type") == "dynamic" and "alpha" not in rope:
        rope["alpha"] = rope.get("factor", 1.0)
        config_path.write_text(json.dumps(config, indent=2))


def _resolve_tokenizer_source(model_dir: str, llm_config=None) -> str:
    tokenizer_source = os.getenv("QWEN_TOKENIZER_SOURCE")
    if tokenizer_source:
        return tokenizer_source

    name_or_path = getattr(llm_config, "_name_or_path", None)
    if name_or_path:
        candidate = Path(name_or_path)
        if candidate.exists():
            return str(candidate)

        model_path = Path(model_dir)
        if model_path.exists():
            relative_candidate = (model_path / name_or_path).resolve()
            if relative_candidate.exists():
                return str(relative_candidate)

    return model_dir


def _load_tokenizer_from_source(source: str, llm_config=None):
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": _local_files_only(source),
    }
    if llm_config is not None:
        kwargs["config"] = llm_config

    try:
        return AutoTokenizer.from_pretrained(source, fix_mistral_regex=True, **kwargs)
    except TypeError:
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(source, **kwargs)


def resolve_language_model_dir(model_dir: str, hf_output_dir: str) -> str:
    config = _load_hf_config(model_dir)
    if _is_qwen_language_model(config):
        model_path = Path(model_dir)
        if model_path.exists():
            _patch_rope_scaling(model_path / "config.json")
            return str(model_path.resolve())
        return model_dir

    if getattr(config, "llm_config", None) is None:
        raise ValueError(f"{model_dir} is neither a standalone Qwen causal LM nor a multimodal config with llm_config.")

    output_dir = Path(hf_output_dir)
    config_path = output_dir / "config.json"
    if config_path.exists():
        _patch_rope_scaling(config_path)
        return str(output_dir.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=_local_files_only(model_dir),
    ).eval()
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise ValueError(f"Model loaded from {model_dir} does not expose a language_model attribute.")
    language_model.save_pretrained(output_dir)

    tokenizer_source = _resolve_tokenizer_source(model_dir, config.llm_config)
    try:
        tokenizer = _load_tokenizer_from_source(tokenizer_source, llm_config=config.llm_config)
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:
        warnings.warn(
            "Tokenizer assets could not be resolved locally, so only the language model weights/config were saved. "
            f"Set QWEN_TOKENIZER_SOURCE if you need tokenizer files alongside {output_dir}. Original error: {exc}"
        )

    del model
    release_gc()
    _patch_rope_scaling(config_path)
    return str(output_dir.resolve())


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    del args
    return QuantConfig()


def args_to_build_options(args):
    return {
        "use_parallel_embedding": args.use_parallel_embedding,
        "embedding_sharding_dim": args.embedding_sharding_dim,
        "load_model_on_cpu": args.load_model_on_cpu,
    }


def convert_and_save_hf(args, hf_model_dir: str):
    world_size = args.tp_size * args.pp_size
    override_fields = {}
    override_fields.update(args_to_build_options(args))
    quant_config = args_to_quant_config(args)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            moe_tp_size=args.moe_tp_size,
            moe_ep_size=args.moe_ep_size,
        )
        qwen = AutoModelForCausalLM.from_hugging_face(
            hf_model_dir,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            **override_fields,
        )
        qwen.config.mapping.cp_size = args.cp_size
        qwen.config.mapping.attn_tp_size = -1
        qwen.config.mapping.attn_cp_size = -1
        qwen.config.mapping.world_size *= args.cp_size
        qwen.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del qwen

    execute(args.workers, [convert_and_save_rank] * world_size, args)
    release_gc()


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    traceback.print_exc()
                    exceptions.append(exc)
            assert len(exceptions) == 0, "Checkpoint conversion failed, please check error log."


def main():
    emit_engine_arch_deprecation("convert_checkpoint.py")
    print(tensorrt_llm.__version__)
    args = parse_arguments()

    if args.moe_tp_size == -1 and args.moe_ep_size == -1:
        args.moe_tp_size = args.tp_size
        args.moe_ep_size = 1
    elif args.moe_tp_size == -1:
        args.moe_tp_size = args.tp_size // args.moe_ep_size
    elif args.moe_ep_size == -1:
        args.moe_ep_size = args.tp_size // args.moe_tp_size
    assert args.moe_tp_size * args.moe_ep_size == args.tp_size, "moe_tp_size * moe_ep_size must equal tp_size"

    tik = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_model_dir = resolve_language_model_dir(args.model_dir, args.hf_output_dir)
    logger.info(f"Using Qwen Hugging Face source: {hf_model_dir}")
    convert_and_save_hf(args, hf_model_dir)

    tok = time.time()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"qwen_hf_model_dir={hf_model_dir}")
    print(f"qwen_tllm_checkpoint_dir={output_dir.resolve()}")
    print(f"Total time of converting checkpoints: {elapsed}")


if __name__ == "__main__":
    main()
