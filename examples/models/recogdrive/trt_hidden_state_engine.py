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
from tensorrt_llm.runtime.generation import _prepare_input_ids
from tensorrt_llm.runtime.model_runner import _engine_config_to_model_config
from transformers import AutoTokenizer

ENGINE_ROOT = Path(
    os.getenv(
        "TRT_HIDDEN_STATE_ENGINE_ROOT",
        "/workspaces/safe-copilot/models/trt_llm_hidden_state_engines",
    )
)


def engine_dir_for(
    model_id: str,
    max_input_len: int,
    max_prompt_embedding_table_size: int = 0,
    *,
    remove_input_padding: bool = True,
    gpt_attention_plugin: str | None = "auto",
) -> Path:
    safe_model_id = model_id.replace("/", "__")
    suffix = f"max_input_len_{max_input_len}"
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
    remove_input_padding: bool = True,
    gpt_attention_plugin: str | None = "auto",
) -> Path:
    hf_dir = resolve_hf_dir(model_id)
    engine_dir = engine_dir_for(
        model_id,
        max_input_len,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        remove_input_padding=remove_input_padding,
        gpt_attention_plugin=gpt_attention_plugin,
    )
    if (engine_dir / "config.json").exists() and (engine_dir / "rank0.engine").exists():
        return engine_dir

    engine_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_hugging_face(hf_dir, dtype="float16")
    original_forward = model.forward

    def forward_with_hidden_outputs(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if model.config.mapping.is_last_pp_rank() and isinstance(outputs, tuple) and len(outputs) == 3:
            _, last_token_hidden_state, last_hidden_state = outputs
            if isinstance(last_token_hidden_state, TrtTensor):
                last_token_hidden_state.mark_output("last_token_hidden_state_output", model.config.dtype)
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
        plugin_config=plugin_config,
    )
    engine = build(model, build_config)
    engine.save(str(engine_dir))
    return engine_dir


def _prepare_prompt_table_inputs(
    *,
    batch_size: int,
    prompt_embedding_table: torch.Tensor | None,
    prompt_tasks: str | torch.Tensor | None,
    max_prompt_embedding_table_size: int,
    hidden_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if max_prompt_embedding_table_size == 0:
        return None, None, None

    if prompt_embedding_table is not None:
        prompt_embedding_table = prompt_embedding_table.to(
            device="cuda",
            dtype=torch.float16,
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
            dtype=torch.float16,
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
) -> torch.Tensor:
    if not batch_input_ids:
        raise ValueError("batch_input_ids must not be empty")
    if len(batch_input_ids) != 1:
        raise NotImplementedError("The hidden-state TensorRT-LLM helper currently supports batch size 1 only")

    engine_dir = build_hidden_state_engine(
        model_id,
        max_input_len,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
    )

    prepared_batch_input_ids = [
        input_ids.to(dtype=torch.int32, device="cuda").contiguous() for input_ids in batch_input_ids
    ]
    input_ids, context_lengths = _prepare_input_ids(prepared_batch_input_ids)
    host_context_lengths = context_lengths.cpu()

    engine = Engine.from_dir(str(engine_dir))
    model_config = _engine_config_to_model_config(engine.config)
    mapping = engine.config.pretrained_config.mapping
    session = GenerationSession(model_config, engine.engine, mapping, debug_mode=True)

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

    context = session.runtime.ctx_context
    session.runtime._set_tensors(context, ctx_tensors)
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
    )
