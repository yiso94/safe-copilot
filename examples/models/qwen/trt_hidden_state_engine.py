import os
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tensorrt_llm import AutoModelForCausalLM, BuildConfig, build
from tensorrt_llm.builder import Engine
from tensorrt_llm.functional import Tensor as TrtTensor
from tensorrt_llm.llmapi.kv_cache_type import KVCacheType
from tensorrt_llm.plugin.plugin import PluginConfig
from tensorrt_llm.runtime import GenerationSession
from tensorrt_llm.runtime import SamplingConfig as RuntimeSamplingConfig
from tensorrt_llm.runtime.generation import RuntimeTensor, _prepare_input_ids
from tensorrt_llm.runtime.model_runner import _engine_config_to_model_config
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_ROOT = Path(
    os.getenv(
        "QWEN_TRT_HIDDEN_STATE_ENGINE_ROOT",
        str(REPO_ROOT / "models" / "qwen" / "hidden_state_engine"),
    )
)


def _str_dtype_to_torch(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise TypeError(f"Unsupported dtype string: {dtype}")


def _existing_engine_matches(
    engine_dir: Path,
    *,
    dtype: str,
    max_input_len: int,
    max_prompt_embedding_table_size: int,
    remove_input_padding: bool,
    gpt_attention_plugin: str | None,
) -> bool:
    config_path = engine_dir / "config.json"
    engine_path = engine_dir / "rank0.engine"
    if not config_path.exists() or not engine_path.exists():
        return False

    try:
        import json

        config = json.loads(config_path.read_text())
    except Exception:
        return False

    build_config = config.get("build_config", {})
    plugin_config = build_config.get("plugin_config", {})
    existing_plugin = plugin_config.get("gpt_attention_plugin")

    return (
        config.get("pretrained_config", {}).get("dtype") == dtype
        and
        int(build_config.get("max_input_len", -1)) == int(max_input_len)
        and int(build_config.get("max_prompt_embedding_table_size", 0)) == int(max_prompt_embedding_table_size)
        and bool(plugin_config.get("remove_input_padding", False)) == bool(remove_input_padding)
        and existing_plugin == gpt_attention_plugin
    )


def engine_dir_for(
    model_id: str,
    max_input_len: int,
    max_prompt_embedding_table_size: int = 0,
    *,
    dtype: str = "float16",
    remove_input_padding: bool = True,
    gpt_attention_plugin: str | None = "auto",
) -> Path:
    safe_model_id = model_id.replace("/", "__")
    suffix = f"{dtype}__max_input_len_{max_input_len}"
    if max_prompt_embedding_table_size > 0:
        suffix += f"__max_prompt_embedding_table_size_{max_prompt_embedding_table_size}"
    if not remove_input_padding:
        suffix += "__remove_input_padding_false"
    if gpt_attention_plugin is None:
        suffix += "__gpt_attention_plugin_none"
    return ENGINE_ROOT / safe_model_id / suffix


def resolve_hf_dir(model_id: str) -> str:
    local_path = Path(model_id)
    if local_path.exists():
        return str(local_path.resolve())
    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return snapshot_download(model_id)


def build_hidden_state_engine(
    model_id: str,
    max_input_len: int,
    max_prompt_embedding_table_size: int = 0,
    *,
    dtype: str = "float16",
    remove_input_padding: bool = True,
    gpt_attention_plugin: str | None = "auto",
    engine_dir: str | Path | None = None,
) -> Path:
    hf_dir = resolve_hf_dir(model_id)
    if engine_dir is None:
        engine_dir_path = engine_dir_for(
            model_id,
            max_input_len,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            dtype=dtype,
            remove_input_padding=remove_input_padding,
            gpt_attention_plugin=gpt_attention_plugin,
        )
    else:
        engine_dir_path = Path(engine_dir)

    if _existing_engine_matches(
        engine_dir_path,
        dtype=dtype,
        max_input_len=max_input_len,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        remove_input_padding=remove_input_padding,
        gpt_attention_plugin=gpt_attention_plugin,
    ):
        return engine_dir_path

    engine_dir_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_hugging_face(hf_dir, dtype=dtype)
    original_forward = model.forward

    def forward_with_hidden_outputs(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if model.config.mapping.is_last_pp_rank() and isinstance(outputs, tuple) and len(outputs) == 3:
            _, _, last_hidden_state = outputs
            if isinstance(last_hidden_state, TrtTensor):
                last_hidden_state.mark_output("last_hidden_state_output", model.config.dtype)
        return outputs

    model.forward = forward_with_hidden_outputs

    plugin_config = PluginConfig(
        remove_input_padding=remove_input_padding,
        gpt_attention_plugin=gpt_attention_plugin,
    )
    build_config = BuildConfig(
        max_batch_size=1,
        opt_batch_size=1,
        max_input_len=max_input_len,
        max_seq_len=max_input_len,
        max_num_tokens=max_input_len,
        max_beam_width=1,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        kv_cache_type=KVCacheType.DISABLED,
        strongly_typed=True,
        plugin_config=plugin_config,
    )
    engine = build(model, build_config)
    engine.save(str(engine_dir_path))
    return engine_dir_path


def _prepare_prompt_table_inputs(
    *,
    batch_size: int,
    prompt_embedding_table: torch.Tensor | None,
    prompt_tasks: str | torch.Tensor | None,
    max_prompt_embedding_table_size: int,
    hidden_size: int,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if max_prompt_embedding_table_size == 0:
        return None, None, None

    if prompt_embedding_table is not None:
        prompt_embedding_table = prompt_embedding_table.to(
            device="cuda",
            dtype=model_dtype,
        ).contiguous()
        if prompt_embedding_table.dim() == 3:
            _, task_vocab_size, prompt_hidden_size = prompt_embedding_table.shape
        elif prompt_embedding_table.dim() == 2:
            task_vocab_size, prompt_hidden_size = prompt_embedding_table.shape
        else:
            raise ValueError("prompt_embedding_table must have shape [tasks, vocab, hidden] or [vocab, hidden]")
        prompt_vocab_size = torch.tensor([task_vocab_size], dtype=torch.int32, device="cuda")
        prompt_embedding_table = prompt_embedding_table.reshape(-1, prompt_hidden_size).contiguous()
    else:
        prompt_embedding_table = torch.empty(
            [1, hidden_size],
            dtype=model_dtype,
            device="cuda",
        )
        prompt_vocab_size = torch.zeros([1], dtype=torch.int32, device="cuda")

    if prompt_tasks is None:
        tasks = torch.zeros([batch_size], dtype=torch.int32, device="cuda")
    elif isinstance(prompt_tasks, str):
        tasks = torch.tensor(
            [int(task) for task in prompt_tasks.split(",")],
            dtype=torch.int32,
            device="cuda",
        )
    else:
        tasks = prompt_tasks.to(device="cuda", dtype=torch.int32).contiguous()

    if tasks.numel() != batch_size:
        raise ValueError(f"Number of prompt tasks ({tasks.numel()}) must match batch size ({batch_size})")

    return prompt_embedding_table, tasks, prompt_vocab_size


def get_last_hidden_state_tensorrt_llm_from_input_ids(
    model_id: str,
    batch_input_ids: Sequence[torch.Tensor],
    max_input_len: int,
    *,
    prompt_embedding_table: torch.Tensor | None = None,
    prompt_tasks: str | torch.Tensor | None = None,
    max_prompt_embedding_table_size: int = 0,
    engine_dir: str | Path | None = None,
) -> torch.Tensor:
    if not batch_input_ids:
        raise ValueError("batch_input_ids must not be empty")
    if len(batch_input_ids) != 1:
        raise NotImplementedError("The hidden-state TensorRT-LLM helper currently supports batch size 1 only")

    engine_dir_path = build_hidden_state_engine(
        model_id,
        max_input_len,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        engine_dir=engine_dir,
    )

    prepared_batch_input_ids = [
        input_ids.to(dtype=torch.int32, device="cuda").contiguous() for input_ids in batch_input_ids
    ]
    input_ids, context_lengths = _prepare_input_ids(prepared_batch_input_ids)
    host_context_lengths = context_lengths.cpu()

    engine = Engine.from_dir(str(engine_dir_path))
    model_config = _engine_config_to_model_config(engine.config)
    mapping = engine.config.pretrained_config.mapping
    session = GenerationSession(model_config, engine.engine, mapping, debug_mode=True)
    model_dtype = _str_dtype_to_torch(str(engine.config.pretrained_config.dtype))

    sampling_config = RuntimeSamplingConfig(
        end_id=1,
        pad_id=0,
        max_new_tokens=1,
    )

    batch_size = len(prepared_batch_input_ids)
    max_context_length = int(context_lengths.max().item())
    session.setup(
        batch_size=batch_size,
        max_context_length=max_context_length,
        max_new_tokens=1,
        beam_width=1,
    )
    session._GenerationSession__setup_decoder(input_ids, sampling_config, host_context_lengths)

    model_inputs = session._prepare_context_inputs(
        batch_size=batch_size,
        context_lengths=context_lengths,
        host_context_lengths=host_context_lengths,
        use_gpt_attention_plugin=session.use_gpt_attention_plugin,
        remove_input_padding=session.remove_input_padding,
        max_context_length=max_context_length,
        input_ids=input_ids,
        pad_id=sampling_config.pad_id,
        eos_id=sampling_config.end_id,
    )

    prompt_embedding_table, tasks, prompt_vocab_size = _prepare_prompt_table_inputs(
        batch_size=batch_size,
        prompt_embedding_table=prompt_embedding_table,
        prompt_tasks=prompt_tasks,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        hidden_size=session.hidden_size,
        model_dtype=model_dtype,
    )

    dummy_cache_indirection = torch.zeros(
        (batch_size, 1, session.max_seq_length),
        dtype=torch.int32,
        device="cuda",
    )
    ctx_tensors = session._get_context_shape_buffer(
        input_ids=input_ids,
        context_lengths=context_lengths,
        host_context_lengths=host_context_lengths,
        position_ids=model_inputs.get("position_ids"),
        last_token_ids=model_inputs.get("last_token_ids"),
        attention_mask=model_inputs.get("attention_mask", None),
        cross_attention_mask=None,
        cache_indirection=dummy_cache_indirection,
        kv_cache_block_offsets=None,
        host_kv_cache_block_offsets=None,
        cross_kv_cache_block_offsets=None,
        host_cross_kv_cache_block_offsets=None,
        hidden_states_input=None,
        prompt_embedding_table=prompt_embedding_table,
        tasks=tasks,
        prompt_vocab_size=prompt_vocab_size,
        encoder_output=None,
        encoder_input_lengths=None,
        host_runtime_perf_knobs=model_inputs.get("host_runtime_perf_knobs", None),
        host_context_progress=torch.tensor([0], dtype=torch.int64),
        skip_cross_attn_blocks=None,
        language_adapter_routings=None,
    )

    total_tokens = int(context_lengths.sum().item())
    output_tensor_names = getattr(session.runtime, "output_tensor_names", [])
    if "last_hidden_state_output" in output_tensor_names and "last_hidden_state_output" not in ctx_tensors:
        ctx_tensors["last_hidden_state_output"] = RuntimeTensor.from_torch(
            "last_hidden_state_output",
            torch.empty(
                (total_tokens, session.hidden_size),
                dtype=model_dtype,
                device="cuda",
            ),
        )
    if "last_token_hidden_state_output" in output_tensor_names and "last_token_hidden_state_output" not in ctx_tensors:
        ctx_tensors["last_token_hidden_state_output"] = RuntimeTensor.from_torch(
            "last_token_hidden_state_output",
            torch.empty(
                (batch_size, session.hidden_size),
                dtype=model_dtype,
                device="cuda",
            ),
        )

    context = session.runtime.ctx_context
    session.runtime._set_tensors(context, ctx_tensors)
    engine_io_names = {session.runtime.engine.get_tensor_name(i) for i in range(session.runtime.engine.num_io_tensors)}
    for name, tensor in ctx_tensors.items():
        if name not in engine_io_names:
            continue
        if context.get_tensor_address(name) != tensor.data:
            try:
                if list(context.get_tensor_shape(name)) != list(tensor.shape):
                    context.set_input_shape(name, tensor.shape)
            except Exception:
                pass
            context.set_tensor_address(name, tensor.data)
    ok = session.runtime._run(context, torch.cuda.current_stream().cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT-LLM hidden-state engine execution failed")
    torch.cuda.synchronize()

    last_hidden_state = ctx_tensors["last_hidden_state_output"].to_torch()
    return last_hidden_state.reshape(batch_size, -1, session.hidden_size)


def get_last_hidden_state_tensorrt_llm(
    model_id: str,
    prompt: str,
    max_input_len: int,
    *,
    engine_dir: str | Path | None = None,
) -> torch.Tensor:
    hf_dir = resolve_hf_dir(model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
        add_special_tokens=True,
    )

    return get_last_hidden_state_tensorrt_llm_from_input_ids(
        model_id,
        [encoded["input_ids"][0]],
        max_input_len,
        engine_dir=engine_dir,
    )
